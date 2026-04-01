//! Render parameters: blend modes, shading, interpolation, clip planes.

use crate::math::Aabb;
use crate::transfer_function::{ColorTransferFunction, OpacityTransferFunction};
use crate::window_level::WindowLevel;
use glam::DVec4;

// ── BlendMode ─────────────────────────────────────────────────────────────────

/// Compositing algorithm used during GPU raycasting.
///
/// # VTK Equivalent
/// `vtkGPUVolumeRayCastMapper::SetBlendMode`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum BlendMode {
    /// Front-to-back alpha compositing (default for anatomical rendering).
    #[default]
    Composite,
    /// Maximum intensity projection — displays the brightest voxel along each ray.
    MaximumIntensity,
    /// Minimum intensity projection.
    MinimumIntensity,
    /// Mean intensity along the ray.
    AverageIntensity,
    /// Additive accumulation (unweighted).
    Additive,
    /// Render the isosurface at a given scalar value using Phong shading.
    Isosurface {
        /// The scalar value defining the isosurface.
        iso_value: f64,
    },
}

// ── Interpolation ─────────────────────────────────────────────────────────────

/// Texture sampling interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Interpolation {
    /// Nearest-neighbour (box) interpolation.
    Nearest,
    /// Trilinear interpolation (default, smoother).
    #[default]
    Linear,
}

// ── ShadingParams ─────────────────────────────────────────────────────────────

/// Phong shading parameters used in composite and isosurface modes.
///
/// Gradient-magnitude based shading matches VTK's `vtkVolumeProperty::ShadeOn`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadingParams {
    /// Ambient term (0–1).
    pub ambient: f32,
    /// Diffuse term (0–1).
    pub diffuse: f32,
    /// Specular term (0–1).
    pub specular: f32,
    /// Specular power (shininess, e.g. 10–100).
    pub specular_power: f32,
}

impl Default for ShadingParams {
    fn default() -> Self {
        Self {
            ambient: 0.1,
            diffuse: 0.7,
            specular: 0.2,
            specular_power: 10.0,
        }
    }
}

// ── ClipPlane ─────────────────────────────────────────────────────────────────

/// An oriented half-space clip plane in world space.
///
/// Points with `plane · (pos, 1)` < 0 are clipped (not rendered).
/// Up to 6 planes are supported simultaneously by the GPU shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipPlane {
    /// Plane equation as `(nx, ny, nz, d)` where `nx·x + ny·y + nz·z + d = 0`.
    /// `(nx, ny, nz)` should be a unit normal.
    pub equation: DVec4,
}

impl ClipPlane {
    /// Create a clip plane from a point on the plane and an outward normal.
    ///
    /// Points on the normal-facing side of the plane are **kept**; points
    /// behind are clipped.
    #[must_use]
    pub fn from_point_and_normal(point: glam::DVec3, normal: glam::DVec3) -> Self {
        let n = normal.normalize();
        let d = -n.dot(point);
        Self {
            equation: DVec4::new(n.x, n.y, n.z, d),
        }
    }

    /// Signed distance of `pos` from the plane (positive = kept side).
    #[must_use]
    pub fn signed_distance(&self, pos: glam::DVec3) -> f64 {
        self.equation.x * pos.x
            + self.equation.y * pos.y
            + self.equation.z * pos.z
            + self.equation.w
    }
}

// ── VolumeRenderParams ────────────────────────────────────────────────────────

/// All parameters that control how a volume is rendered.
///
/// Built using a fluent builder pattern. Start from [`VolumeRenderParams::builder`]
/// or use [`VolumeRenderParams::default`].
#[derive(Debug, Clone)]
pub struct VolumeRenderParams {
    /// Colour transfer function.
    pub color_tf: ColorTransferFunction,
    /// Opacity transfer function.
    pub opacity_tf: OpacityTransferFunction,
    /// Optional gradient opacity modulation.
    pub gradient_opacity_tf: Option<OpacityTransferFunction>,
    /// Window/level mapping (applied before TF lookup).
    pub window_level: Option<WindowLevel>,
    /// Compositing algorithm.
    pub blend_mode: BlendMode,
    /// Texture interpolation quality.
    pub interpolation: Interpolation,
    /// Phong shading — only used in [`BlendMode::Composite`] and [`BlendMode::Isosurface`].
    pub shading: Option<ShadingParams>,
    /// Raycasting step size as a fraction of the smallest voxel spacing (default 0.5).
    pub step_size_factor: f32,
    /// Up to 6 world-space clip planes.
    pub clip_planes: Vec<ClipPlane>,
    /// Optional axis-aligned cropping box in world coordinates.
    pub cropping_bounds: Option<Aabb>,
    /// Background colour RGBA in `[0, 1]`.
    pub background: [f32; 4],
}

impl Default for VolumeRenderParams {
    fn default() -> Self {
        Self {
            color_tf: ColorTransferFunction::greyscale(0.0, 1.0),
            opacity_tf: OpacityTransferFunction::linear_ramp(0.0, 1.0),
            gradient_opacity_tf: None,
            window_level: None,
            blend_mode: BlendMode::default(),
            interpolation: Interpolation::default(),
            shading: Some(ShadingParams::default()),
            step_size_factor: 0.5,
            clip_planes: Vec::new(),
            cropping_bounds: None,
            background: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl VolumeRenderParams {
    /// Start building a new parameter set from the defaults.
    #[must_use]
    pub fn builder() -> VolumeRenderParamsBuilder {
        VolumeRenderParamsBuilder::default()
    }
}

/// Fluent builder for [`VolumeRenderParams`].
#[derive(Debug, Default)]
pub struct VolumeRenderParamsBuilder {
    params: VolumeRenderParams,
}

impl VolumeRenderParamsBuilder {
    /// Set the blend mode.
    #[must_use]
    pub fn blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.params.blend_mode = blend_mode;
        self
    }

    /// Set the texture interpolation mode.
    #[must_use]
    pub fn interpolation(mut self, interpolation: Interpolation) -> Self {
        self.params.interpolation = interpolation;
        self
    }

    /// Enable Phong shading with the given parameters.
    #[must_use]
    pub fn shading(mut self, params: ShadingParams) -> Self {
        self.params.shading = Some(params);
        self
    }

    /// Disable shading.
    #[must_use]
    pub fn no_shading(mut self) -> Self {
        self.params.shading = None;
        self
    }

    /// Set the raycasting step size factor.
    #[must_use]
    pub fn step_size_factor(mut self, step: f32) -> Self {
        self.params.step_size_factor = step;
        self
    }

    /// Set the colour transfer function.
    #[must_use]
    pub fn color_tf(mut self, tf: ColorTransferFunction) -> Self {
        self.params.color_tf = tf;
        self
    }

    /// Set the opacity transfer function.
    #[must_use]
    pub fn opacity_tf(mut self, tf: OpacityTransferFunction) -> Self {
        self.params.opacity_tf = tf;
        self
    }

    /// Set the gradient-based opacity modulation transfer function.
    #[must_use]
    pub fn gradient_opacity_tf(mut self, tf: OpacityTransferFunction) -> Self {
        self.params.gradient_opacity_tf = Some(tf);
        self
    }

    /// Set the window/level mapping.
    #[must_use]
    pub fn window_level(mut self, wl: WindowLevel) -> Self {
        self.params.window_level = Some(wl);
        self
    }

    /// Set an axis-aligned cropping box in world coordinates.
    #[must_use]
    pub fn cropping_bounds(mut self, bounds: Aabb) -> Self {
        self.params.cropping_bounds = Some(bounds);
        self
    }

    /// Add a clip plane.
    #[must_use]
    pub fn clip_plane(mut self, plane: ClipPlane) -> Self {
        self.params.clip_planes.push(plane);
        self
    }

    /// Set the background colour.
    #[must_use]
    pub fn background(mut self, rgba: [f32; 4]) -> Self {
        self.params.background = rgba;
        self
    }

    /// Finalise and return the [`VolumeRenderParams`].
    #[must_use]
    pub fn build(self) -> VolumeRenderParams {
        self.params
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use glam::DVec3;

    #[test]
    fn builder_overrides_defaults() {
        let params = VolumeRenderParams::builder()
            .blend_mode(BlendMode::MaximumIntensity)
            .no_shading()
            .step_size_factor(0.25)
            .build();
        assert_eq!(params.blend_mode, BlendMode::MaximumIntensity);
        assert!(params.shading.is_none());
        assert_abs_diff_eq!(params.step_size_factor as f64, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn clip_plane_from_point_normal() {
        let plane = ClipPlane::from_point_and_normal(DVec3::ZERO, DVec3::Y);
        // Points above Y=0 should be positive
        let d = plane.signed_distance(DVec3::new(0.0, 1.0, 0.0));
        assert!(d > 0.0, "expected positive distance, got {d}");
        // Points below Y=0 should be negative
        let d2 = plane.signed_distance(DVec3::new(0.0, -1.0, 0.0));
        assert!(d2 < 0.0);
    }

    #[test]
    fn clip_plane_at_point_is_zero() {
        let plane = ClipPlane::from_point_and_normal(DVec3::new(0.0, 5.0, 0.0), DVec3::Y);
        let d = plane.signed_distance(DVec3::new(3.0, 5.0, 7.0));
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }
}
