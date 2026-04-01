//! GPU volume texture management.

use volren_core::volume::DynVolume;

/// Describes how a volume's scalars are represented in the GPU texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTextureFormat {
    /// 8-bit unsigned normalised.
    R8Unorm,
    /// 16-bit unsigned normalised.
    R16Unorm,
    /// 16-bit signed integer.
    R16Sint,
    /// 32-bit float.
    R32Float,
}

impl GpuTextureFormat {
    /// Select the appropriate texture format for the given volume's scalar type.
    pub fn for_volume(vol: &DynVolume) -> Self {
        match vol {
            DynVolume::U8(_) => Self::R8Unorm,
            DynVolume::I8(_) | DynVolume::U16(_) => Self::R16Unorm,
            DynVolume::I16(_) => Self::R16Sint,
            _ => Self::R32Float,
        }
    }

    /// Convert to the corresponding `wgpu::TextureFormat`.
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::R8Unorm => wgpu::TextureFormat::R8Unorm,
            Self::R16Unorm => wgpu::TextureFormat::R16Unorm,
            Self::R16Sint => wgpu::TextureFormat::R16Sint,
            Self::R32Float => wgpu::TextureFormat::R32Float,
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
}

impl GpuVolumeTexture {
    /// Upload a [`DynVolume`] to the GPU as a 3D texture.
    ///
    /// Existing GPU state is released when this value is dropped.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &DynVolume,
        linear: bool,
    ) -> Self {
        use volren_core::volume::VolumeInfo;

        let dims = volume.dimensions();
        let format = GpuTextureFormat::for_volume(volume);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volren_volume_3d"),
            size: wgpu::Extent3d {
                width: dims.x,
                height: dims.y,
                depth_or_array_layers: dims.z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: format.to_wgpu(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let bytes = volume.as_bytes();
        let bytes_per_component = volume.bytes_per_component();

        queue.write_texture(
            texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims.x * bytes_per_component as u32),
                rows_per_image: Some(dims.y),
            },
            wgpu::Extent3d {
                width: dims.x,
                height: dims.y,
                depth_or_array_layers: dims.z,
            },
        );

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

        Self { texture, view, sampler }
    }
}
