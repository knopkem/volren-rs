//! GPU uniform buffer layout for the volume raycaster.

use bytemuck::{Pod, Zeroable};

/// Uniform data uploaded to the GPU every frame.
///
/// Must match the `@group(0) @binding(0)` uniform in `volume_raycast.wgsl`.
///
/// **Memory layout rules** (for wgpu/WGSL alignment):
/// - All `[f32; 4]` fields are 16 bytes each (vec4 / mat4 columns)
/// - Pad to 16-byte alignment throughout
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct VolumeUniforms {
    /// Model-view-projection matrix (column-major, f32).
    pub mvp: [[f32; 4]; 4],
    /// Inverse model-view-projection matrix for screen-to-world ray unprojection.
    pub inv_mvp: [[f32; 4]; 4],
    /// World-to-volume-texture-space matrix.
    pub world_to_volume: [[f32; 4]; 4],
    /// Volume-to-world matrix (inverse of world_to_volume).
    pub volume_to_world: [[f32; 4]; 4],

    /// Volume dimensions `(nx, ny, nz, 0)`.
    pub dimensions: [f32; 4],
    /// Voxel spacing `(sx, sy, sz, 0)` in world units.
    pub spacing: [f32; 4],
    /// Scalar range `(min, max, 0, 0)` for normalising scalar values.
    pub scalar_range: [f32; 4],

    /// Ray-march step size in texture-space units.
    pub step_size: f32,
    /// Opacity correction factor for step-size–independent compositing.
    pub opacity_correction: f32,
    /// Blend mode index (matches the `BlendMode` enum).
    pub blend_mode: u32,
    /// 1 if shading is enabled, 0 otherwise.
    pub shading_enabled: u32,

    /// Phong ambient term.
    pub ambient: f32,
    /// Phong diffuse term.
    pub diffuse: f32,
    /// Phong specular term.
    pub specular: f32,
    /// Phong specular power (shininess).
    pub specular_power: f32,

    /// Light position in world space `(x, y, z, 0)`.
    pub light_position: [f32; 4],
    /// Camera position in world space `(x, y, z, 0)`.
    pub camera_position: [f32; 4],

    /// Window center for W/L mapping.
    pub window_center: f32,
    /// Window width for W/L mapping.
    pub window_width: f32,
    /// Number of active clip planes.
    pub num_clip_planes: u32,
    /// Padding.
    pub _pad0: u32,

    /// Up to 6 world-space clip planes `(nx, ny, nz, d)` each.
    pub clip_planes: [[f32; 4]; 6],
    /// Background colour used when no sample contributes.
    pub background: [f32; 4],
}

/// Blend mode constants that map to shader integer codes.
pub mod blend_mode {
    /// Front-to-back alpha compositing.
    pub const COMPOSITE: u32 = 0;
    /// Maximum intensity projection.
    pub const MAXIMUM_INTENSITY: u32 = 1;
    /// Minimum intensity projection.
    pub const MINIMUM_INTENSITY: u32 = 2;
    /// Average intensity.
    pub const AVERAGE_INTENSITY: u32 = 3;
    /// Additive.
    pub const ADDITIVE: u32 = 4;
    /// Isosurface.
    pub const ISOSURFACE: u32 = 5;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_are_pod() {
        let u = VolumeUniforms::zeroed();
        let _bytes: &[u8] = bytemuck::bytes_of(&u);
    }

    #[test]
    fn uniforms_size_is_multiple_of_16() {
        assert_eq!(std::mem::size_of::<VolumeUniforms>() % 16, 0);
    }
}
