//! Multiplanar reslice (MPR) types: slice planes and thick-slab parameters.
//!
//! # VTK Equivalent
//! `vtkImageReslice`, `vtkResliceCursor` — oblique plane sampling through a 3D volume.

use glam::{DVec2, DVec3};

/// An oriented 2D plane through 3D space, used for MPR reslicing.
///
/// The plane is parameterised by:
/// - `origin`  — a point on the plane (world coordinates)
/// - `right`   — the unit vector along the plane's horizontal (X) axis
/// - `up`      — the unit vector along the plane's vertical (Y) axis
/// - `width`   — physical extent along `right` in world units (mm)
/// - `height`  — physical extent along `up` in world units (mm)
///
/// The plane normal is `right × up`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlicePlane {
    /// A point on the plane in world space (centre of the slice).
    pub origin: DVec3,
    /// Horizontal unit vector on the plane.
    pub right: DVec3,
    /// Vertical unit vector on the plane.
    pub up: DVec3,
    /// Physical extent along the `right` axis (mm).
    pub width: f64,
    /// Physical extent along the `up` axis (mm).
    pub height: f64,
}

impl SlicePlane {
    /// Create a new slice plane.
    ///
    /// `right` and `up` are normalised and orthogonalised internally.
    #[must_use]
    pub fn new(origin: DVec3, right: DVec3, up: DVec3, width: f64, height: f64) -> Self {
        let r = right.normalize_or(DVec3::X);
        // Orthogonalise up against right
        let u = (up - r * up.dot(r)).normalize_or(DVec3::Y);
        Self {
            origin,
            right: r,
            up: u,
            width,
            height,
        }
    }

    /// The standard axial (XY) plane at `z = z_coord` with the given extent.
    #[must_use]
    pub fn axial(z_coord: f64, extent: f64) -> Self {
        Self::new(
            DVec3::new(0.0, 0.0, z_coord),
            DVec3::X,
            DVec3::Y,
            extent,
            extent,
        )
    }

    /// The standard coronal (XZ) plane at `y = y_coord` with the given extent.
    #[must_use]
    pub fn coronal(y_coord: f64, extent: f64) -> Self {
        Self::new(
            DVec3::new(0.0, y_coord, 0.0),
            DVec3::X,
            DVec3::Z,
            extent,
            extent,
        )
    }

    /// The standard sagittal (YZ) plane at `x = x_coord` with the given extent.
    #[must_use]
    pub fn sagittal(x_coord: f64, extent: f64) -> Self {
        Self::new(
            DVec3::new(x_coord, 0.0, 0.0),
            DVec3::Y,
            DVec3::Z,
            extent,
            extent,
        )
    }

    /// The outward plane normal (`right × up`).
    #[must_use]
    pub fn normal(&self) -> DVec3 {
        self.right.cross(self.up)
    }

    /// Convert a 2D position on the plane `(u, v)` to a world-space point.
    ///
    /// `u` and `v` are in world units from the origin.
    #[must_use]
    pub fn to_world(&self, u: f64, v: f64) -> DVec3 {
        self.origin + self.right * u + self.up * v
    }

    /// Convert a normalised `[0,1]²` point on the slice to world coordinates.
    ///
    /// `(0.5, 0.5)` maps to the origin (centre of the plane).
    #[must_use]
    pub fn point_to_world(&self, uv: DVec2) -> DVec3 {
        let u = (uv.x - 0.5) * self.width;
        let v = (uv.y - 0.5) * self.height;
        self.origin + self.right * u + self.up * v
    }

    /// Project a world point onto the slice plane.
    ///
    /// Returns `(uv, signed_distance)` where `uv` is in normalised `[0,1]²`
    /// coordinates and `signed_distance` is the perpendicular distance from
    /// the plane (positive on the normal side).
    #[must_use]
    pub fn world_to_point(&self, world: DVec3) -> (DVec2, f64) {
        let rel = world - self.origin;
        let n = self.normal();
        let dist = rel.dot(n);
        let u = rel.dot(self.right) / self.width + 0.5;
        let v = rel.dot(self.up) / self.height + 0.5;
        (DVec2::new(u, v), dist)
    }

    /// Translate the plane along its normal by `delta` world units.
    #[must_use]
    pub fn offset_along_normal(&self, delta: f64) -> Self {
        Self {
            origin: self.origin + self.normal() * delta,
            ..*self
        }
    }

    /// Translate the plane in place along its normal by `distance` world units.
    pub fn translate_along_normal(&mut self, distance: f64) {
        self.origin += self.normal() * distance;
    }

    /// Rotate the plane around an axis through `self.origin`.
    #[must_use]
    pub fn rotated(&self, axis: DVec3, angle_rad: f64) -> Self {
        let rot = glam::DQuat::from_axis_angle(axis.normalize_or(DVec3::Z), angle_rad);
        Self::new(
            self.origin,
            rot * self.right,
            rot * self.up,
            self.width,
            self.height,
        )
    }
}

// ── ThickSlab ─────────────────────────────────────────────────────────────────

/// Mode for thick-slab projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ThickSlabMode {
    /// Maximum intensity projection along the slab.
    #[default]
    Mip,
    /// Minimum intensity projection.
    MinIp,
    /// Mean (average) of all samples.
    Mean,
}

/// Parameters for thick-slab MPR rendering.
///
/// Instead of a single infinitesimally thin slice, the renderer integrates
/// a `thickness`-deep slab centred on a [`SlicePlane`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThickSlabParams {
    /// Half-thickness on each side of the plane (world units).
    pub half_thickness: f64,
    /// Projection mode within the slab.
    pub mode: ThickSlabMode,
    /// Number of samples along the slab normal.
    pub num_samples: u32,
}

impl Default for ThickSlabParams {
    fn default() -> Self {
        Self {
            half_thickness: 1.0,
            mode: ThickSlabMode::Mip,
            num_samples: 10,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn axial_normal_is_z() {
        let p = SlicePlane::axial(0.0, 100.0);
        let n = p.normal();
        assert_abs_diff_eq!(n.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(n.y, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(n.z, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn coronal_normal_is_neg_y() {
        let p = SlicePlane::coronal(0.0, 100.0);
        let n = p.normal();
        assert_abs_diff_eq!(n.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(n.y.abs(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(n.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn to_world_origin_is_identity() {
        let p = SlicePlane::axial(5.0, 100.0);
        let w = p.to_world(0.0, 0.0);
        assert_abs_diff_eq!(w.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.y, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.z, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn to_world_displaces_correctly() {
        let p = SlicePlane::axial(0.0, 100.0);
        let w = p.to_world(3.0, 4.0);
        assert_abs_diff_eq!(w.x, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.y, 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn offset_along_normal() {
        let p = SlicePlane::axial(0.0, 100.0);
        let q = p.offset_along_normal(5.0);
        assert_abs_diff_eq!(q.origin.z, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn new_orthogonalises_up() {
        let p = SlicePlane::new(
            DVec3::ZERO,
            DVec3::X,
            DVec3::new(0.5, 0.866, 0.0),
            10.0,
            10.0,
        );
        let dot = p.right.dot(p.up);
        assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn right_and_up_are_unit_vectors() {
        let p = SlicePlane::sagittal(10.0, 100.0);
        assert_abs_diff_eq!(p.right.length(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p.up.length(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn point_to_world_center() {
        let p = SlicePlane::axial(5.0, 100.0);
        let w = p.point_to_world(DVec2::new(0.5, 0.5));
        assert_abs_diff_eq!(w.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.y, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.z, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn point_to_world_world_to_point_round_trip() {
        let p = SlicePlane::axial(5.0, 100.0);
        let uv = DVec2::new(0.3, 0.7);
        let world = p.point_to_world(uv);
        let (back_uv, dist) = p.world_to_point(world);
        assert_abs_diff_eq!(back_uv.x, uv.x, epsilon = 1e-8);
        assert_abs_diff_eq!(back_uv.y, uv.y, epsilon = 1e-8);
        assert_abs_diff_eq!(dist, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn world_to_point_distance() {
        let p = SlicePlane::axial(0.0, 100.0);
        let (_, dist) = p.world_to_point(DVec3::new(0.0, 0.0, 3.0));
        assert_abs_diff_eq!(dist, 3.0, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn world_to_point_round_trip(
            u_val in 0.01f64..0.99,
            v_val in 0.01f64..0.99,
            z_offset in -50.0f64..50.0,
        ) {
            let plane = SlicePlane::axial(z_offset, 100.0);
            let uv = DVec2::new(u_val, v_val);
            let world = plane.point_to_world(uv);
            let (back_uv, dist) = plane.world_to_point(world);
            prop_assert!((back_uv.x - uv.x).abs() < 1e-8, "u: {} vs {}", uv.x, back_uv.x);
            prop_assert!((back_uv.y - uv.y).abs() < 1e-8, "v: {} vs {}", uv.y, back_uv.y);
            prop_assert!(dist.abs() < 1e-8, "dist = {dist}");
        }
    }
}
