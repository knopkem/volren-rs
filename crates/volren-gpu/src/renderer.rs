//! Main volume renderer: wgpu pipeline creation and render execution.

use std::sync::Arc;

use volren_core::{
    camera::Camera,
    render_params::{BlendMode, VolumeRenderParams},
    reslice::SlicePlane,
    transfer_function::TransferFunctionLut,
    volume::{DynVolume, VolumeInfo},
    window_level::WindowLevel,
};

use crate::{
    texture::GpuVolumeTexture,
    uniforms::{blend_mode as bm, VolumeUniforms},
};

const SHADER_SRC: &str = include_str!("shaders/volume_raycast.wgsl");

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
        Self { x: 0, y: 0, width, height }
    }
}

/// Errors that can occur during rendering.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// No volume has been uploaded yet.
    #[error("no volume data uploaded — call set_volume() first")]
    NoVolume,
    /// No transfer function LUT has been uploaded.
    #[error("no transfer function LUT uploaded — call set_transfer_function() first")]
    NoTransferFunction,
    /// Viewport has zero area.
    #[error("viewport has zero area")]
    ZeroViewport,
}

/// Parameters for rendering crosshair overlay lines on a 2D slice viewport.
#[derive(Debug, Clone)]
pub struct CrosshairParams {
    /// Normalised position in `[0, 1]` × `[0, 1]` on the slice.
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

/// A fully GPU-resident volume renderer.
///
/// # Usage
/// 1. Create with [`VolumeRenderer::from_arc`] (preferred) or [`VolumeRenderer::new`].
/// 2. Call [`VolumeRenderer::set_volume`] whenever the volume changes.
/// 3. Call [`VolumeRenderer::set_transfer_function`] after LUT baking.
/// 4. Call [`VolumeRenderer::render_volume`] each frame, passing your own encoder.
pub struct VolumeRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    uniform_buffer: wgpu::Buffer,
    lut_texture: wgpu::Texture,
    lut_view: wgpu::TextureView,
    lut_sampler: wgpu::Sampler,

    volume_texture: Option<GpuVolumeTexture>,
    bind_group: Option<wgpu::BindGroup>,
    has_lut: bool,

    output_format: wgpu::TextureFormat,
    viewport_size: (u32, u32),
}

impl VolumeRenderer {
    /// Create a renderer from `Arc`-wrapped device and queue (preferred).
    pub fn from_arc(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volren_raycast_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout = Self::create_bind_group_layout(&device);
        let pipeline = Self::create_pipeline(&device, &shader, &bind_group_layout, output_format);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volren_uniforms"),
            size: std::mem::size_of::<VolumeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (lut_texture, lut_view, lut_sampler) = Self::create_lut_texture(&device, 256);

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            lut_texture,
            lut_view,
            lut_sampler,
            volume_texture: None,
            bind_group: None,
            has_lut: false,
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
        let tex = GpuVolumeTexture::upload(&self.device, &self.queue, volume, linear_interpolation);
        self.volume_texture = Some(tex);
        self.rebuild_bind_group();
    }

    /// Upload a baked transfer function LUT to the GPU.
    pub fn set_transfer_function(&mut self, lut: &TransferFunctionLut) {
        let (lut_texture, lut_view, lut_sampler) =
            Self::create_lut_texture(&self.device, lut.lut_size());
        self.queue.write_texture(
            lut_texture.as_image_copy(),
            lut.as_bytes(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(lut.lut_size() * 4 * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d { width: lut.lut_size(), height: 1, depth_or_array_layers: 1 },
        );
        self.lut_texture = lut_texture;
        self.lut_view = lut_view;
        self.lut_sampler = lut_sampler;
        self.has_lut = true;
        self.rebuild_bind_group();
    }

    /// Handle viewport resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.viewport_size = (width, height);
    }

    /// Render the volume into the given color attachment.
    ///
    /// The caller owns the command encoder and submits it.
    ///
    /// # Errors
    /// Returns [`RenderError`] if prerequisites are missing or viewport is zero-size.
    pub fn render_volume(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        volume_info: &dyn VolumeInfo,
        params: &VolumeRenderParams,
        viewport: Viewport,
    ) -> Result<(), RenderError> {
        if self.volume_texture.is_none() {
            return Err(RenderError::NoVolume);
        }
        if !self.has_lut {
            return Err(RenderError::NoTransferFunction);
        }
        if viewport.width == 0 || viewport.height == 0 {
            return Err(RenderError::ZeroViewport);
        }
        let bind_group = self.bind_group.as_ref().ok_or(RenderError::NoVolume)?;

        let uniforms = self.build_uniforms(camera, volume_info, params, viewport);
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        {
            let [r, g, b, a] = params.background;
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volren_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: r as f64,
                            g: g as f64,
                            b: b as f64,
                            a: a as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.draw(0..6, 0..1);
        }

        Ok(())
    }

    /// Render a 2D reslice (MPR slice).
    ///
    /// Placeholder — the dedicated reslice shader will be implemented in a future phase.
    #[allow(unused_variables)]
    pub fn render_slice(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        slice_plane: &SlicePlane,
        window_level: &WindowLevel,
        viewport: Viewport,
    ) -> Result<(), RenderError> {
        // TODO: Phase 3 — implement reslice shader pipeline
        Ok(())
    }

    /// Render an orientation marker in a corner of the viewport.
    ///
    /// Placeholder for Phase 6.
    #[allow(unused_variables)]
    pub fn render_orientation_marker(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        viewport: Viewport,
        labels: &OrientationLabels,
    ) -> Result<(), RenderError> {
        // TODO: Phase 6
        Ok(())
    }

    /// Render crosshair overlay lines on a slice viewport.
    ///
    /// Placeholder for Phase 6.
    #[allow(unused_variables)]
    pub fn render_crosshair(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        viewport: Viewport,
        crosshair: &CrosshairParams,
    ) -> Result<(), RenderError> {
        // TODO: Phase 6
        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn build_uniforms(
        &self,
        camera: &Camera,
        volume_info: &dyn VolumeInfo,
        params: &VolumeRenderParams,
        viewport: Viewport,
    ) -> VolumeUniforms {
        use glam::{Mat4, Vec3};

        let aspect = viewport.width as f64 / viewport.height as f64;
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let mvp = (proj * view).as_mat4();

        let dims = volume_info.dimensions().as_vec3();
        let spacing = volume_info.spacing().as_vec3();
        let origin = volume_info.origin().as_vec3();

        let world_to_vol = Mat4::from_scale(Vec3::ONE / (dims * spacing))
            * Mat4::from_translation(-origin);
        let vol_to_world = world_to_vol.inverse();

        let blend_mode_u32 = match params.blend_mode {
            BlendMode::Composite => bm::COMPOSITE,
            BlendMode::MaximumIntensity => bm::MAXIMUM_INTENSITY,
            BlendMode::MinimumIntensity => bm::MINIMUM_INTENSITY,
            BlendMode::AverageIntensity => bm::AVERAGE_INTENSITY,
            BlendMode::Additive => bm::ADDITIVE,
            BlendMode::Isosurface { .. } => bm::ISOSURFACE,
            _ => bm::COMPOSITE,
        };

        let (shading_enabled, ambient, diffuse, specular, specular_power) =
            if let Some(s) = &params.shading {
                (1u32, s.ambient, s.diffuse, s.specular, s.specular_power)
            } else {
                (0u32, 0.0, 0.0, 0.0, 0.0)
            };

        let cam_pos = camera.position().as_vec3();
        let light_pos = cam_pos;

        let mut clip_planes = [[0.0f32; 4]; 6];
        for (i, plane) in params.clip_planes.iter().take(6).enumerate() {
            let eq = plane.equation;
            clip_planes[i] = [eq.x as f32, eq.y as f32, eq.z as f32, eq.w as f32];
        }

        let (window_center, window_width) = if let Some(ref wl) = params.window_level {
            (wl.center as f32, wl.width as f32)
        } else {
            (0.5, 1.0)
        };

        let iso_value = if let BlendMode::Isosurface { iso_value } = params.blend_mode {
            iso_value as f32
        } else {
            0.0
        };

        let opacity_correction = 1.0 / params.step_size_factor;

        VolumeUniforms {
            mvp: mvp.to_cols_array_2d(),
            world_to_volume: world_to_vol.to_cols_array_2d(),
            volume_to_world: vol_to_world.to_cols_array_2d(),
            dimensions: [dims.x, dims.y, dims.z, 0.0],
            spacing: [spacing.x, spacing.y, spacing.z, 0.0],
            scalar_range: [0.0, 1.0, iso_value, 0.0],
            step_size: params.step_size_factor,
            opacity_correction,
            blend_mode: blend_mode_u32,
            shading_enabled,
            ambient,
            diffuse,
            specular,
            specular_power,
            light_position: [light_pos.x, light_pos.y, light_pos.z, 0.0],
            camera_position: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
            window_center,
            window_width,
            num_clip_planes: params.clip_planes.len().min(6) as u32,
            _pad0: 0,
            clip_planes,
        }
    }

    fn rebuild_bind_group(&mut self) {
        let Some(ref vol_tex) = self.volume_texture else {
            return;
        };

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volren_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&vol_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&vol_tex.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                },
            ],
        });
        self.bind_group = Some(bg);
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volren_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        bgl: &wgpu::BindGroupLayout,
        output_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volren_pipeline_layout"),
            bind_group_layouts: &[bgl],
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            size: wgpu::Extent3d { width: size, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            format: wgpu::TextureFormat::Rgba32Float,
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
