//! Main volume renderer: wgpu pipeline creation and render execution.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{DMat3, DMat4, DVec2, DVec3, DVec4};
use half::f16;
use volren_core::{
    camera::Camera,
    render_params::{BlendMode, ClipPlane, VolumeRenderParams},
    reslice::{SlicePlane, ThickSlabMode, ThickSlabParams},
    transfer_function::{OpacityTransferFunction, TransferFunctionLut},
    volume::{DynVolume, VolumeInfo},
    window_level::WindowLevel,
};

use crate::{
    texture::GpuVolumeTexture,
    uniforms::{blend_mode as bm, VolumeUniforms},
};

const VOLUME_SHADER_SRC: &str = concat!(
    include_str!("shaders/common.wgsl"),
    "\n",
    include_str!("shaders/fullscreen_quad.wgsl"),
    "\n",
    include_str!("shaders/gradient.wgsl"),
    "\n",
    include_str!("shaders/shading.wgsl"),
    "\n",
    include_str!("shaders/volume_raycast.wgsl"),
);
const RESLICE_SHADER_SRC: &str = include_str!("shaders/reslice.wgsl");
const CROSSHAIR_SHADER_SRC: &str = include_str!("shaders/crosshair.wgsl");
const BLIT_SHADER_SRC: &str = include_str!("shaders/blit_rgba.wgsl");

/// Rectangular sub-region of the render target, in pixels.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    /// Horizontal offset from the left edge.
    pub x: u32,
    /// Vertical offset from the top edge.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Viewport {
    /// Create a viewport covering the full render target.
    #[must_use]
    pub fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }
}

/// Errors that can occur during rendering.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// No volume has been uploaded yet.
    #[error("no volume data uploaded — call `set_volume()` first")]
    NoVolume,
    /// No transfer-function data has been uploaded yet.
    #[error("no render parameters uploaded — call `set_render_params()` first")]
    NoTransferFunction,
    /// Viewport has zero area.
    #[error("viewport has zero area")]
    ZeroViewport,
}

/// Parameters for rendering crosshair overlay lines on a 2D slice viewport.
#[derive(Debug, Clone)]
pub struct CrosshairParams {
    /// Normalised position in `[0, 1] × [0, 1]` on the slice.
    pub position: [f32; 2],
    /// Line colour for the horizontal line (RGBA, 0–1).
    pub horizontal_color: [f32; 4],
    /// Line colour for the vertical line (RGBA, 0–1).
    pub vertical_color: [f32; 4],
    /// Line thickness in pixels.
    pub thickness: f32,
}

impl Default for CrosshairParams {
    fn default() -> Self {
        Self {
            position: [0.5, 0.5],
            horizontal_color: [1.0, 0.0, 0.0, 1.0],
            vertical_color: [0.0, 1.0, 0.0, 1.0],
            thickness: 1.0,
        }
    }
}

/// Patient orientation labels for the orientation marker.
#[derive(Debug, Clone)]
pub struct OrientationLabels {
    /// Label for the right direction (+X).
    pub right: String,
    /// Label for the left direction (−X).
    pub left: String,
    /// Label for the anterior direction (+Y).
    pub anterior: String,
    /// Label for the posterior direction (−Y).
    pub posterior: String,
    /// Label for the superior direction (+Z).
    pub superior: String,
    /// Label for the inferior direction (−Z).
    pub inferior: String,
}

impl Default for OrientationLabels {
    fn default() -> Self {
        Self {
            right: "R".into(),
            left: "L".into(),
            anterior: "A".into(),
            posterior: "P".into(),
            superior: "S".into(),
            inferior: "I".into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VolumeMetadata {
    world_to_volume: [[f32; 4]; 4],
    volume_to_world: [[f32; 4]; 4],
    dimensions: [f32; 3],
    spacing: [f32; 3],
    scalar_range: [f32; 2],
}

impl VolumeMetadata {
    fn from_volume(volume: &DynVolume) -> Self {
        let dimensions = volume.dimensions().as_dvec3();
        let spacing = volume.spacing();
        let direction = mat4_from_direction(volume.direction());
        let scale = DVec3::new(
            (dimensions.x - 1.0).max(1.0) * spacing.x,
            (dimensions.y - 1.0).max(1.0) * spacing.y,
            (dimensions.z - 1.0).max(1.0) * spacing.z,
        );
        let volume_to_world =
            DMat4::from_translation(volume.origin()) * direction * DMat4::from_scale(scale);
        let world_to_volume = volume_to_world.inverse();
        let (scalar_min, scalar_max) = volume.scalar_range();

        Self {
            world_to_volume: world_to_volume.as_mat4().to_cols_array_2d(),
            volume_to_world: volume_to_world.as_mat4().to_cols_array_2d(),
            dimensions: [
                dimensions.x as f32,
                dimensions.y as f32,
                dimensions.z as f32,
            ],
            spacing: [spacing.x as f32, spacing.y as f32, spacing.z as f32],
            scalar_range: [scalar_min as f32, scalar_max as f32],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SliceUniforms {
    world_to_volume: [[f32; 4]; 4],
    slice_origin: [f32; 4],
    slice_right: [f32; 4],
    slice_up: [f32; 4],
    slice_normal: [f32; 4],
    slice_extent: [f32; 4],
    window_level: [f32; 4],
    slab_params: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CrosshairUniforms {
    position: [f32; 4],
    horizontal_color: [f32; 4],
    vertical_color: [f32; 4],
    viewport: [f32; 4],
}

/// A fully GPU-resident volume renderer.
///
/// The renderer stores the uploaded 3D texture plus the metadata needed for
/// raycasting and reslicing. Call [`VolumeRenderer::set_render_params`] after
/// uploading a volume to bake the active transfer functions.
pub struct VolumeRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    volume_pipeline: wgpu::RenderPipeline,
    volume_bind_group_layout: wgpu::BindGroupLayout,
    volume_uniform_buffer: wgpu::Buffer,

    slice_pipeline: wgpu::RenderPipeline,
    slice_bind_group_layout: wgpu::BindGroupLayout,
    slice_uniform_buffer: wgpu::Buffer,

    crosshair_pipeline: wgpu::RenderPipeline,
    crosshair_uniform_buffer: wgpu::Buffer,
    crosshair_bind_group: wgpu::BindGroup,

    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,

    lut_texture: wgpu::Texture,
    lut_view: wgpu::TextureView,
    lut_sampler: wgpu::Sampler,
    gradient_lut_texture: wgpu::Texture,
    gradient_lut_view: wgpu::TextureView,
    gradient_lut_sampler: wgpu::Sampler,

    volume_texture: Option<GpuVolumeTexture>,
    volume_bind_group: Option<wgpu::BindGroup>,
    slice_bind_group: Option<wgpu::BindGroup>,
    volume_metadata: Option<VolumeMetadata>,
    has_render_params: bool,

    output_format: wgpu::TextureFormat,
    viewport_size: (u32, u32),
}

impl VolumeRenderer {
    /// Create a new renderer for the given device and output format.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        Self::from_arc(
            Arc::new(device.clone()),
            Arc::new(queue.clone()),
            output_format,
        )
    }

    /// Create a renderer from `Arc`-wrapped device and queue.
    #[must_use]
    pub fn from_arc(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let volume_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volren_volume_shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUME_SHADER_SRC.into()),
        });
        let slice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volren_reslice_shader"),
            source: wgpu::ShaderSource::Wgsl(RESLICE_SHADER_SRC.into()),
        });
        let crosshair_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volren_crosshair_shader"),
            source: wgpu::ShaderSource::Wgsl(CROSSHAIR_SHADER_SRC.into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volren_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER_SRC.into()),
        });

        let volume_bind_group_layout = Self::create_volume_bind_group_layout(&device);
        let slice_bind_group_layout = Self::create_slice_bind_group_layout(&device);
        let crosshair_bind_group_layout = Self::create_crosshair_bind_group_layout(&device);
        let blit_bind_group_layout = Self::create_blit_bind_group_layout(&device);

        let volume_pipeline = Self::create_pipeline(
            &device,
            &volume_shader,
            &volume_bind_group_layout,
            output_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        let slice_pipeline = Self::create_pipeline(
            &device,
            &slice_shader,
            &slice_bind_group_layout,
            output_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        let crosshair_pipeline = Self::create_pipeline(
            &device,
            &crosshair_shader,
            &crosshair_bind_group_layout,
            output_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        let blit_pipeline = Self::create_pipeline(
            &device,
            &blit_shader,
            &blit_bind_group_layout,
            output_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        let volume_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volren_volume_uniforms"),
            size: std::mem::size_of::<VolumeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let slice_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volren_slice_uniforms"),
            size: std::mem::size_of::<SliceUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let crosshair_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volren_crosshair_uniforms"),
            size: std::mem::size_of::<CrosshairUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let crosshair_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volren_crosshair_bind_group"),
            layout: &crosshair_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: crosshair_uniform_buffer.as_entire_binding(),
            }],
        });

        let (lut_texture, lut_view, lut_sampler) = Self::create_lut_texture(&device, 4096);
        let (gradient_lut_texture, gradient_lut_view, gradient_lut_sampler) =
            Self::create_lut_texture(&device, 1024);
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volren_blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            device,
            queue,
            volume_pipeline,
            volume_bind_group_layout,
            volume_uniform_buffer,
            slice_pipeline,
            slice_bind_group_layout,
            slice_uniform_buffer,
            crosshair_pipeline,
            crosshair_uniform_buffer,
            crosshair_bind_group,
            blit_pipeline,
            blit_bind_group_layout,
            blit_sampler,
            lut_texture,
            lut_view,
            lut_sampler,
            gradient_lut_texture,
            gradient_lut_view,
            gradient_lut_sampler,
            volume_texture: None,
            volume_bind_group: None,
            slice_bind_group: None,
            volume_metadata: None,
            has_render_params: false,
            output_format,
            viewport_size: (0, 0),
        }
    }

    /// The texture format this renderer outputs into.
    #[must_use]
    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    /// Upload (or replace) volume data as a 3D GPU texture.
    pub fn set_volume(&mut self, volume: &DynVolume, linear_interpolation: bool) {
        self.volume_texture = Some(GpuVolumeTexture::upload(
            &self.device,
            &self.queue,
            volume,
            linear_interpolation,
        ));
        self.volume_metadata = Some(VolumeMetadata::from_volume(volume));
        self.rebuild_bind_groups();
    }

    /// Upload a baked transfer-function LUT to the GPU.
    pub fn set_transfer_function(&mut self, lut: &TransferFunctionLut) {
        let (texture, view, sampler) = Self::create_lut_texture(&self.device, lut.lut_size());
        let f16_bytes = f32_slice_to_f16_bytes(lut.as_rgba_f32());
        self.queue.write_texture(
            texture.as_image_copy(),
            &f16_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(lut.lut_size() * 4 * 2),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: lut.lut_size(),
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.lut_texture = texture;
        self.lut_view = view;
        self.lut_sampler = sampler;
        self.has_render_params = true;
        self.rebuild_bind_groups();
    }

    /// Bake and upload transfer functions from the current render parameters.
    ///
    /// # Errors
    /// Returns [`RenderError::NoVolume`] if no volume metadata has been uploaded yet.
    pub fn set_render_params(&mut self, params: &VolumeRenderParams) -> Result<(), RenderError> {
        let metadata = self.volume_metadata.ok_or(RenderError::NoVolume)?;
        let lut = TransferFunctionLut::bake(
            &params.color_tf,
            &params.opacity_tf,
            f64::from(metadata.scalar_range[0]),
            f64::from(metadata.scalar_range[1]),
            4096,
        );
        self.set_transfer_function(&lut);
        let gradient_tf = params
            .gradient_opacity_tf
            .clone()
            .unwrap_or_else(opaque_unit_ramp);
        self.upload_gradient_lut(&gradient_tf);
        Ok(())
    }

    /// Handle viewport resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.viewport_size = (width, height);
    }

    /// Create an off-screen render target texture.
    #[must_use]
    pub fn create_render_target(&self, width: u32, height: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volren_offscreen_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.output_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Render the volume into the given color attachment.
    ///
    /// The caller owns the command encoder and submits it.
    ///
    /// # Errors
    /// Returns [`RenderError`] if prerequisites are missing or the viewport is invalid.
    pub fn render_volume(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        params: &VolumeRenderParams,
        viewport: Viewport,
    ) -> Result<(), RenderError> {
        let metadata = self.volume_metadata.ok_or(RenderError::NoVolume)?;
        let bind_group = self
            .volume_bind_group
            .as_ref()
            .ok_or(RenderError::NoVolume)?;
        validate_viewport(viewport)?;
        if !self.has_render_params {
            return Err(RenderError::NoTransferFunction);
        }

        let uniforms = self.build_uniforms(camera, metadata, params, viewport);
        self.queue.write_buffer(
            &self.volume_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volren_volume_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.volume_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );
        pass.draw(0..6, 0..1);
        Ok(())
    }

    /// Render the volume into a newly-created off-screen texture.
    ///
    /// # Errors
    /// Propagates the same errors as [`Self::render_volume`].
    pub fn render_volume_to_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        params: &VolumeRenderParams,
        width: u32,
        height: u32,
    ) -> Result<wgpu::Texture, RenderError> {
        let texture = self.create_render_target(width, height);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_volume(
            encoder,
            &view,
            camera,
            params,
            Viewport::full(width, height),
        )?;
        Ok(texture)
    }

    /// Render the volume at reduced resolution and upscale the result into `target`.
    ///
    /// This is intended for interactive manipulation where responsiveness matters
    /// more than final image quality. Passing `1` disables downsampling.
    ///
    /// # Errors
    /// Propagates the same errors as [`Self::render_volume`].
    pub fn render_volume_interactive(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        params: &VolumeRenderParams,
        viewport: Viewport,
        downsample_factor: u32,
    ) -> Result<(), RenderError> {
        validate_viewport(viewport)?;
        let factor = downsample_factor.max(1);
        if factor == 1 {
            return self.render_volume(encoder, target, camera, params, viewport);
        }

        let lod_width = (viewport.width / factor).max(1);
        let lod_height = (viewport.height / factor).max(1);
        let texture = self.create_render_target(lod_width, lod_height);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.render_volume(
            encoder,
            &view,
            camera,
            params,
            Viewport::full(lod_width, lod_height),
        )?;
        self.blit_texture_view(encoder, target, viewport, &view);
        Ok(())
    }

    /// Render a 2D reslice (MPR slice) into the given color attachment.
    ///
    /// # Errors
    /// Returns [`RenderError::NoVolume`] when no volume has been uploaded.
    pub fn render_slice(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        slice_plane: &SlicePlane,
        window_level: &WindowLevel,
        viewport: Viewport,
        thick_slab: Option<&ThickSlabParams>,
    ) -> Result<(), RenderError> {
        let metadata = self.volume_metadata.ok_or(RenderError::NoVolume)?;
        let bind_group = self
            .slice_bind_group
            .as_ref()
            .ok_or(RenderError::NoVolume)?;
        validate_viewport(viewport)?;

        let uniforms = self.build_slice_uniforms(metadata, slice_plane, window_level, thick_slab);
        self.queue
            .write_buffer(&self.slice_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volren_slice_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.slice_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );
        pass.draw(0..6, 0..1);
        Ok(())
    }

    /// Render a slice into a newly-created off-screen texture.
    ///
    /// # Errors
    /// Propagates the same errors as [`Self::render_slice`].
    pub fn render_slice_to_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slice_plane: &SlicePlane,
        window_level: &WindowLevel,
        width: u32,
        height: u32,
        thick_slab: Option<&ThickSlabParams>,
    ) -> Result<wgpu::Texture, RenderError> {
        let texture = self.create_render_target(width, height);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_slice(
            encoder,
            &view,
            slice_plane,
            window_level,
            Viewport::full(width, height),
            thick_slab,
        )?;
        Ok(texture)
    }

    /// Render crosshair overlay lines on a slice viewport.
    ///
    /// # Errors
    /// Returns [`RenderError::ZeroViewport`] for an empty viewport.
    pub fn render_crosshair(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        viewport: Viewport,
        crosshair: &CrosshairParams,
    ) -> Result<(), RenderError> {
        validate_viewport(viewport)?;
        let uniforms = CrosshairUniforms {
            position: [
                crosshair.position[0],
                crosshair.position[1],
                crosshair.thickness,
                0.0,
            ],
            horizontal_color: crosshair.horizontal_color,
            vertical_color: crosshair.vertical_color,
            viewport: [viewport.width as f32, viewport.height as f32, 0.0, 0.0],
        };
        self.queue.write_buffer(
            &self.crosshair_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volren_crosshair_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.crosshair_pipeline);
        pass.set_bind_group(0, &self.crosshair_bind_group, &[]);
        pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );
        pass.draw(0..6, 0..1);
        Ok(())
    }

    /// Render an orientation marker in the given viewport.
    ///
    /// The marker is generated on the CPU as a small RGBA image, then composited
    /// over the target with a lightweight textured-quad pass.
    ///
    /// # Errors
    /// Returns [`RenderError::ZeroViewport`] for an empty viewport.
    pub fn render_orientation_marker(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        viewport: Viewport,
        labels: &OrientationLabels,
    ) -> Result<(), RenderError> {
        validate_viewport(viewport)?;
        let width = viewport.width.max(1);
        let height = viewport.height.max(1);
        let image = build_orientation_marker_image(width, height, camera, labels);
        self.blit_rgba8(encoder, target, viewport, width, height, &image);
        Ok(())
    }

    fn build_uniforms(
        &self,
        camera: &Camera,
        metadata: VolumeMetadata,
        params: &VolumeRenderParams,
        viewport: Viewport,
    ) -> VolumeUniforms {
        let aspect = f64::from(viewport.width) / f64::from(viewport.height.max(1));
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let mvp = (proj * view).as_mat4();
        let inv_mvp = (proj * view).inverse().as_mat4();

        let blend_mode = match params.blend_mode {
            BlendMode::Composite => bm::COMPOSITE,
            BlendMode::MaximumIntensity => bm::MAXIMUM_INTENSITY,
            BlendMode::MinimumIntensity => bm::MINIMUM_INTENSITY,
            BlendMode::AverageIntensity => bm::AVERAGE_INTENSITY,
            BlendMode::Additive => bm::ADDITIVE,
            BlendMode::Isosurface { .. } => bm::ISOSURFACE,
            _ => bm::COMPOSITE,
        };
        let (window_center, window_width) = params.window_level.map_or_else(
            || {
                let wl = WindowLevel::from_scalar_range(
                    f64::from(metadata.scalar_range[0]),
                    f64::from(metadata.scalar_range[1]),
                );
                (wl.center as f32, wl.width as f32)
            },
            |wl| (wl.center as f32, wl.width as f32),
        );
        let (shading_enabled, ambient, diffuse, specular, specular_power) =
            if let Some(shading) = params.shading {
                (
                    1u32,
                    shading.ambient,
                    shading.diffuse,
                    shading.specular,
                    shading.specular_power,
                )
            } else {
                (0u32, 0.0, 0.0, 0.0, 0.0)
            };
        let (clip_planes, num_clip_planes) = combined_clip_planes(params);
        let iso_value = match params.blend_mode {
            BlendMode::Isosurface { iso_value } => iso_value as f32,
            _ => 0.0,
        };
        let camera_position = camera.position().as_vec3();

        VolumeUniforms {
            mvp: mvp.to_cols_array_2d(),
            inv_mvp: inv_mvp.to_cols_array_2d(),
            world_to_volume: metadata.world_to_volume,
            volume_to_world: metadata.volume_to_world,
            dimensions: [
                metadata.dimensions[0],
                metadata.dimensions[1],
                metadata.dimensions[2],
                0.0,
            ],
            spacing: [
                metadata.spacing[0],
                metadata.spacing[1],
                metadata.spacing[2],
                0.0,
            ],
            scalar_range: [
                metadata.scalar_range[0],
                metadata.scalar_range[1],
                iso_value,
                0.0,
            ],
            step_size: params.step_size_factor.max(1e-3),
            opacity_correction: 1.0 / params.step_size_factor.max(1e-3),
            blend_mode,
            shading_enabled,
            ambient,
            diffuse,
            specular,
            specular_power,
            light_position: [camera_position.x, camera_position.y, camera_position.z, 0.0],
            camera_position: [camera_position.x, camera_position.y, camera_position.z, 0.0],
            window_center,
            window_width,
            num_clip_planes,
            _pad0: 0,
            clip_planes,
            background: params.background,
        }
    }

    fn build_slice_uniforms(
        &self,
        metadata: VolumeMetadata,
        slice_plane: &SlicePlane,
        window_level: &WindowLevel,
        thick_slab: Option<&ThickSlabParams>,
    ) -> SliceUniforms {
        let slab_mode = thick_slab.map_or(ThickSlabMode::Mip, |params| params.mode);
        let (half_thickness, num_samples) = thick_slab.map_or((0.0f32, 1u32), |params| {
            (params.half_thickness as f32, params.num_samples.max(1))
        });

        SliceUniforms {
            world_to_volume: metadata.world_to_volume,
            slice_origin: [
                slice_plane.origin.x as f32,
                slice_plane.origin.y as f32,
                slice_plane.origin.z as f32,
                0.0,
            ],
            slice_right: [
                slice_plane.right.x as f32,
                slice_plane.right.y as f32,
                slice_plane.right.z as f32,
                0.0,
            ],
            slice_up: [
                slice_plane.up.x as f32,
                slice_plane.up.y as f32,
                slice_plane.up.z as f32,
                0.0,
            ],
            slice_normal: [
                slice_plane.normal().x as f32,
                slice_plane.normal().y as f32,
                slice_plane.normal().z as f32,
                0.0,
            ],
            slice_extent: [
                slice_plane.width as f32,
                slice_plane.height as f32,
                half_thickness,
                0.0,
            ],
            window_level: [
                window_level.center as f32,
                window_level.width as f32,
                0.0,
                0.0,
            ],
            slab_params: [thick_slab_mode_code(slab_mode), num_samples, 0, 0],
        }
    }

    fn upload_gradient_lut(&mut self, tf: &OpacityTransferFunction) {
        let resolution = 1024u32;
        let f32_bytes = bake_opacity_lut_bytes(tf, resolution);
        let f32_slice: &[f32] = bytemuck::cast_slice(&f32_bytes);
        let f16_bytes = f32_slice_to_f16_bytes(f32_slice);
        let (texture, view, sampler) = Self::create_lut_texture(&self.device, resolution);
        self.queue.write_texture(
            texture.as_image_copy(),
            &f16_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(resolution * 4 * 2),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: resolution,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.gradient_lut_texture = texture;
        self.gradient_lut_view = view;
        self.gradient_lut_sampler = sampler;
        self.rebuild_bind_groups();
    }

    fn rebuild_bind_groups(&mut self) {
        let Some(volume_texture) = self.volume_texture.as_ref() else {
            return;
        };

        self.volume_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volren_volume_bind_group"),
            layout: &self.volume_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.volume_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&volume_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.gradient_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.gradient_lut_sampler),
                },
            ],
        }));

        self.slice_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volren_slice_bind_group"),
            layout: &self.slice_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.slice_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&volume_texture.sampler),
                },
            ],
        }));
    }

    fn blit_rgba8(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        viewport: Viewport,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volren_blit_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.blit_texture_view(encoder, target, viewport, &view);
    }

    fn blit_texture_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        viewport: Viewport,
        source_view: &wgpu::TextureView,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volren_blit_bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volren_blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );
        pass.draw(0..6, 0..1);
    }

    fn create_volume_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volren_volume_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                texture_bgl_entry(1, wgpu::TextureViewDimension::D3),
                sampler_bgl_entry(2),
                texture_bgl_entry(3, wgpu::TextureViewDimension::D1),
                sampler_bgl_entry(4),
                texture_bgl_entry(5, wgpu::TextureViewDimension::D1),
                sampler_bgl_entry(6),
            ],
        })
    }

    fn create_slice_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volren_slice_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                texture_bgl_entry(1, wgpu::TextureViewDimension::D3),
                sampler_bgl_entry(2),
            ],
        })
    }

    fn create_crosshair_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volren_crosshair_bgl"),
            entries: &[uniform_bgl_entry(0)],
        })
    }

    fn create_blit_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volren_blit_bgl"),
            entries: &[texture_bgl_entry_2d(0), sampler_bgl_entry(1)],
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        bind_group_layout: &wgpu::BindGroupLayout,
        output_format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volren_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volren_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_lut_texture(
        device: &wgpu::Device,
        size: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volren_lut"),
            size: wgpu::Extent3d {
                width: size,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D1),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volren_lut_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        (texture, view, sampler)
    }
}

fn validate_viewport(viewport: Viewport) -> Result<(), RenderError> {
    if viewport.width == 0 || viewport.height == 0 {
        Err(RenderError::ZeroViewport)
    } else {
        Ok(())
    }
}

fn opaque_unit_ramp() -> OpacityTransferFunction {
    let mut tf = OpacityTransferFunction::new();
    tf.add_point(0.0, 1.0);
    tf.add_point(1.0, 1.0);
    tf
}

fn bake_opacity_lut_bytes(tf: &OpacityTransferFunction, resolution: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((resolution * 4) as usize);
    for i in 0..resolution {
        let t = if resolution <= 1 {
            0.0
        } else {
            f64::from(i) / f64::from(resolution - 1)
        };
        let opacity = tf.evaluate(t) as f32;
        rgba.extend_from_slice(&[opacity, opacity, opacity, 1.0]);
    }
    bytemuck::cast_slice(&rgba).to_vec()
}

/// Convert an f32 slice to packed f16 (little-endian) bytes for `Rgba16Float` upload.
fn f32_slice_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &val in data {
        bytes.extend_from_slice(&f16::from_f32(val).to_le_bytes());
    }
    bytes
}

fn combined_clip_planes(params: &VolumeRenderParams) -> ([[f32; 4]; 6], u32) {
    let mut planes = params.clip_planes.clone();
    if let Some(bounds) = params.cropping_bounds {
        planes.extend([
            ClipPlane::from_point_and_normal(DVec3::new(bounds.min.x, 0.0, 0.0), DVec3::X),
            ClipPlane::from_point_and_normal(DVec3::new(bounds.max.x, 0.0, 0.0), DVec3::NEG_X),
            ClipPlane::from_point_and_normal(DVec3::new(0.0, bounds.min.y, 0.0), DVec3::Y),
            ClipPlane::from_point_and_normal(DVec3::new(0.0, bounds.max.y, 0.0), DVec3::NEG_Y),
            ClipPlane::from_point_and_normal(DVec3::new(0.0, 0.0, bounds.min.z), DVec3::Z),
            ClipPlane::from_point_and_normal(DVec3::new(0.0, 0.0, bounds.max.z), DVec3::NEG_Z),
        ]);
    }

    let mut packed = [[0.0f32; 4]; 6];
    for (index, plane) in planes.iter().take(6).enumerate() {
        let eq = plane.equation;
        packed[index] = [eq.x as f32, eq.y as f32, eq.z as f32, eq.w as f32];
    }
    (packed, planes.len().min(6) as u32)
}

fn thick_slab_mode_code(mode: ThickSlabMode) -> u32 {
    match mode {
        ThickSlabMode::Mip => 0,
        ThickSlabMode::MinIp => 1,
        ThickSlabMode::Mean => 2,
        _ => 0,
    }
}

fn mat4_from_direction(direction: DMat3) -> DMat4 {
    DMat4::from_cols(
        direction.x_axis.extend(0.0),
        direction.y_axis.extend(0.0),
        direction.z_axis.extend(0.0),
        DVec4::W,
    )
}

fn uniform_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_bgl_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_bgl_entry_2d(binding: u32) -> wgpu::BindGroupLayoutEntry {
    texture_bgl_entry(binding, wgpu::TextureViewDimension::D2)
}

fn sampler_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn build_orientation_marker_image(
    width: u32,
    height: u32,
    camera: &Camera,
    labels: &OrientationLabels,
) -> Vec<u8> {
    let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];
    let center = DVec2::new(f64::from(width) * 0.5, f64::from(height) * 0.5);
    let radius = f64::from(width.min(height)) * 0.28;
    let view = camera.view_matrix();

    let axes = [
        (DVec3::X, [255, 80, 80, 255], labels.right.as_str()),
        (-DVec3::X, [128, 40, 40, 220], labels.left.as_str()),
        (DVec3::Y, [80, 255, 80, 255], labels.anterior.as_str()),
        (-DVec3::Y, [40, 128, 40, 220], labels.posterior.as_str()),
        (DVec3::Z, [80, 160, 255, 255], labels.superior.as_str()),
        (-DVec3::Z, [40, 80, 128, 220], labels.inferior.as_str()),
    ];

    for (axis, color, label) in axes {
        let projected = project_axis(view, axis);
        if projected.length_squared() < 1e-8 {
            continue;
        }
        let end = center + projected.normalize() * radius;
        draw_line(&mut pixels, width, height, center, end, color);
        draw_text(
            &mut pixels,
            width,
            height,
            end + projected.normalize() * 6.0,
            label,
            color,
        );
    }

    draw_disc(
        &mut pixels,
        width,
        height,
        center,
        2.5,
        [255, 255, 255, 255],
    );
    pixels
}

fn project_axis(view: DMat4, axis: DVec3) -> DVec2 {
    let camera_space = view.transform_vector3(axis);
    DVec2::new(camera_space.x, -camera_space.y)
}

fn draw_line(pixels: &mut [u8], width: u32, height: u32, start: DVec2, end: DVec2, color: [u8; 4]) {
    let delta = end - start;
    let steps = delta.length().ceil().max(1.0) as u32;
    for step in 0..=steps {
        let t = f64::from(step) / f64::from(steps.max(1));
        let point = start + delta * t;
        alpha_plot(
            pixels,
            width,
            height,
            point.x.round() as i32,
            point.y.round() as i32,
            color,
        );
    }
}

fn draw_disc(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    center: DVec2,
    radius: f64,
    color: [u8; 4],
) {
    let min_x = (center.x - radius).floor() as i32;
    let max_x = (center.x + radius).ceil() as i32;
    let min_y = (center.y - radius).floor() as i32;
    let max_y = (center.y + radius).ceil() as i32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = f64::from(x) - center.x;
            let dy = f64::from(y) - center.y;
            if dx * dx + dy * dy <= radius * radius {
                alpha_plot(pixels, width, height, x, y, color);
            }
        }
    }
}

fn draw_text(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    position: DVec2,
    text: &str,
    color: [u8; 4],
) {
    let mut cursor_x = position.x.round() as i32;
    let cursor_y = position.y.round() as i32;
    for ch in text.chars() {
        draw_char(pixels, width, height, cursor_x, cursor_y, ch, color);
        cursor_x += 6;
    }
}

fn draw_char(pixels: &mut [u8], width: u32, height: u32, x: i32, y: i32, ch: char, color: [u8; 4]) {
    let glyph = glyph_rows(ch);
    for (row_index, row_bits) in glyph.iter().enumerate() {
        for col in 0..5 {
            if (row_bits >> (4 - col)) & 1 == 1 {
                alpha_plot(pixels, width, height, x + col, y + row_index as i32, color);
            }
        }
    }
}

fn alpha_plot(pixels: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let index = ((y as u32 * width + x as u32) * 4) as usize;
    let src_a = f32::from(color[3]) / 255.0;
    let dst_a = f32::from(pixels[index + 3]) / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    let blend = |src: u8, dst: u8| -> u8 {
        if out_a <= f32::EPSILON {
            0
        } else {
            (((f32::from(src) * src_a) + (f32::from(dst) * dst_a * (1.0 - src_a))) / out_a)
                .round()
                .clamp(0.0, 255.0) as u8
        }
    };

    pixels[index] = blend(color[0], pixels[index]);
    pixels[index + 1] = blend(color[1], pixels[index + 1]);
    pixels[index + 2] = blend(color[2], pixels[index + 2]);
    pixels[index + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

fn glyph_rows(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        _ => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_table_contains_expected_rows() {
        assert_eq!(glyph_rows('R')[0], 0b11110);
        assert_eq!(glyph_rows('I')[6], 0b11111);
    }

    #[test]
    fn thick_slab_mode_codes_are_stable() {
        assert_eq!(thick_slab_mode_code(ThickSlabMode::Mip), 0);
        assert_eq!(thick_slab_mode_code(ThickSlabMode::MinIp), 1);
        assert_eq!(thick_slab_mode_code(ThickSlabMode::Mean), 2);
    }
}

#[cfg(all(test, feature = "snapshot-tests"))]
mod gpu_smoke_tests {
    use super::*;
    use std::sync::mpsc;

    use glam::{DMat3, DVec3, UVec3};
    use volren_core::{Volume, VolumeRenderParams};

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()
        })
    }

    fn small_volume() -> DynVolume {
        let mut data = vec![0u16; 16 * 16 * 16];
        data[8 + 8 * 16 + 8 * 16 * 16] = 2048;
        Volume::from_data(
            data,
            UVec3::new(16, 16, 16),
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        )
        .expect("valid test volume")
        .into()
    }

    fn sphere_volume() -> DynVolume {
        let dims = UVec3::new(32, 32, 32);
        let center = DVec3::new(15.5, 15.5, 15.5);
        let radius = 9.0;
        let mut data = vec![0u16; (dims.x * dims.y * dims.z) as usize];

        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let index = (z * dims.x * dims.y + y * dims.x + x) as usize;
                    let point = DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                    data[index] = if (point - center).length() <= radius {
                        2048
                    } else {
                        0
                    };
                }
            }
        }

        Volume::from_data(data, dims, DVec3::ONE, DVec3::ZERO, DMat3::IDENTITY, 1)
            .expect("valid sphere volume")
            .into()
    }

    fn read_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
        let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volren_test_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let (sender, receiver) = mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _ = device.poll(wgpu::MaintainBase::Wait);
        receiver.recv().expect("map callback").expect("map success");

        let mapped = buffer.slice(..).get_mapped_range();
        let mut pixels = vec![0u8; (unpadded_bytes_per_row * height) as usize];
        for row in 0..height as usize {
            let src_offset = row * padded_bytes_per_row as usize;
            let dst_offset = row * unpadded_bytes_per_row as usize;
            pixels[dst_offset..dst_offset + unpadded_bytes_per_row as usize]
                .copy_from_slice(&mapped[src_offset..src_offset + unpadded_bytes_per_row as usize]);
        }
        drop(mapped);
        buffer.unmap();
        pixels
    }

    fn checksum(bytes: &[u8]) -> u64 {
        bytes.iter().enumerate().fold(0u64, |acc, (index, value)| {
            acc.wrapping_add((index as u64 + 1) * u64::from(*value))
        })
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn render_volume_smoke_test() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let mut renderer = VolumeRenderer::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
        let volume = small_volume();
        renderer.set_volume(&volume, true);
        renderer
            .set_render_params(&VolumeRenderParams::default())
            .expect("render params upload");

        let camera = Camera::new_perspective(DVec3::new(0.0, 0.0, 50.0), DVec3::ZERO, 30.0);
        let texture = renderer.create_render_target(64, 64);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer
            .render_volume(
                &mut encoder,
                &view,
                &camera,
                &VolumeRenderParams::default(),
                Viewport::full(64, 64),
            )
            .expect("volume render");
        queue.submit(std::iter::once(encoder.finish()));
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn render_sphere_snapshot_checksum() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let mut renderer = VolumeRenderer::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
        let volume = sphere_volume();
        renderer.set_volume(&volume, true);
        let params = VolumeRenderParams::default();
        renderer
            .set_render_params(&params)
            .expect("render params upload");

        let camera = Camera::new_perspective(DVec3::new(0.0, 0.0, 60.0), DVec3::ZERO, 30.0);
        let texture = renderer.create_render_target(64, 64);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer
            .render_volume(
                &mut encoder,
                &view,
                &camera,
                &params,
                Viewport::full(64, 64),
            )
            .expect("volume render");
        queue.submit(std::iter::once(encoder.finish()));

        let pixels = read_texture(&device, &queue, &texture, 64, 64);
        let image_checksum = checksum(&pixels);
        eprintln!("sphere checksum: {image_checksum}");
        assert!(image_checksum > 0, "rendered sphere should not be empty");
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn render_slice_and_crosshair_smoke_test() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let mut renderer = VolumeRenderer::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
        let volume = small_volume();
        renderer.set_volume(&volume, true);
        renderer
            .set_render_params(&VolumeRenderParams::default())
            .expect("render params upload");

        let texture = renderer.create_render_target(64, 64);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer
            .render_slice(
                &mut encoder,
                &view,
                &SlicePlane::axial(0.0, 32.0),
                &WindowLevel::from_scalar_range(0.0, 2048.0),
                Viewport::full(64, 64),
                None,
            )
            .expect("slice render");
        renderer
            .render_crosshair(
                &mut encoder,
                &view,
                Viewport::full(64, 64),
                &CrosshairParams::default(),
            )
            .expect("crosshair render");
        queue.submit(std::iter::once(encoder.finish()));
    }
}
