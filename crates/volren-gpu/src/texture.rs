//! GPU volume texture management.

use half::f16;
use volren_core::volume::{DynVolume, VolumeInfo};

/// The GPU texture format used for uploaded volumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTextureFormat {
    /// Single-channel 16-bit floating-point texture.
    R16Float,
}

impl GpuTextureFormat {
    /// Convert to the corresponding `wgpu::TextureFormat`.
    #[must_use]
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::R16Float => wgpu::TextureFormat::R16Float,
        }
    }
}

/// A 3D volume texture on the GPU.
#[allow(dead_code)]
pub(crate) struct GpuVolumeTexture {
    /// The wgpu texture handle.
    pub texture: wgpu::Texture,
    /// The texture view used in bind groups.
    pub view: wgpu::TextureView,
    /// The sampler (linear or nearest).
    pub sampler: wgpu::Sampler,
    /// The underlying GPU texture format.
    pub format: wgpu::TextureFormat,
    /// Texture dimensions in voxels.
    pub dimensions: glam::UVec3,
}

impl GpuVolumeTexture {
    /// Allocate an empty 3D GPU texture for the given volume dimensions.
    pub fn allocate_empty(
        device: &wgpu::Device,
        dimensions: glam::UVec3,
        linear: bool,
    ) -> Self {
        let format = GpuTextureFormat::R16Float.to_wgpu();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volren_volume_3d"),
            size: wgpu::Extent3d {
                width: dimensions.x,
                height: dimensions.y,
                depth_or_array_layers: dimensions.z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let filter = if linear {
            wgpu::FilterMode::Linear
        } else {
            wgpu::FilterMode::Nearest
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volren_volume_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            format,
            dimensions,
        }
    }

    /// Upload a [`DynVolume`] to the GPU as a 3D texture.
    ///
    /// All scalar types are converted to half-precision floating point so the
    /// shader can sample a single consistent texture format.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &DynVolume,
        linear: bool,
    ) -> Self {
        let dims = volume.dimensions();
        let texture = Self::allocate_empty(device, dims, linear);
        let voxels = to_f16_bits(volume);
        write_f16_bits(queue, &texture.texture, texture.dimensions, 0, dims.z, &voxels);
        texture
    }

    /// Update one Z slice of the 3D texture from signed 16-bit voxels.
    pub fn update_i16_slice(&self, queue: &wgpu::Queue, z_index: u32, pixels: &[i16]) {
        let converted: Vec<u16> = pixels
            .iter()
            .map(|&value| f16::from_f32(f32::from(value)).to_bits())
            .collect();
        write_f16_bits(queue, &self.texture, self.dimensions, z_index, 1, &converted);
    }
}

fn write_f16_bits(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    dimensions: glam::UVec3,
    start_z: u32,
    depth: u32,
    voxels: &[u16],
) {
    let slice_len = (dimensions.x * dimensions.y) as usize;
    let max_chunk_depth =
        ((4 * 1024 * 1024) / (slice_len.max(1) * std::mem::size_of::<u16>())).max(1) as u32;
    let bytes_per_row = dimensions.x * std::mem::size_of::<u16>() as u32;

    for chunk_start in (0..depth).step_by(max_chunk_depth as usize) {
        let chunk_depth = (depth - chunk_start).min(max_chunk_depth);
        let start = chunk_start as usize * slice_len;
        let end = start + chunk_depth as usize * slice_len;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: start_z + chunk_start,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&voxels[start..end]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(dimensions.y),
            },
            wgpu::Extent3d {
                width: dimensions.x,
                height: dimensions.y,
                depth_or_array_layers: chunk_depth,
            },
        );
    }
}

fn to_f16_bits(volume: &DynVolume) -> Vec<u16> {
    match volume {
        DynVolume::U8(v) => v
            .data()
            .iter()
            .map(|&value| f16::from_f32(f32::from(value)).to_bits())
            .collect(),
        DynVolume::I8(v) => v
            .data()
            .iter()
            .map(|&value| f16::from_f32(f32::from(value)).to_bits())
            .collect(),
        DynVolume::U16(v) => v
            .data()
            .iter()
            .map(|&value| f16::from_f32(value as f32).to_bits())
            .collect(),
        DynVolume::I16(v) => v
            .data()
            .iter()
            .map(|&value| f16::from_f32(f32::from(value)).to_bits())
            .collect(),
        DynVolume::U32(v) => v
            .data()
            .iter()
            .map(|&value| f16::from_f32(value as f32).to_bits())
            .collect(),
        DynVolume::I32(v) => v
            .data()
            .iter()
            .map(|&value| f16::from_f32(value as f32).to_bits())
            .collect(),
        DynVolume::F32(v) => v
            .data()
            .iter()
            .map(|&value| f16::from_f32(value).to_bits())
            .collect(),
        DynVolume::F64(v) => v
            .data()
            .iter()
            .map(|&value| f16::from_f32(value as f32).to_bits())
            .collect(),
        _ => unreachable!("all current DynVolume variants are handled"),
    }
}
