//! Camera model: perspective and orthographic projections, arcball manipulation.
//!
//! # VTK Equivalent
//! `vtkCamera` — position, focal point, view-up, clipping range, view angle.

use glam::{DMat4, DVec3};

// ── Projection ────────────────────────────────────────────────────────────────

/// Camera projection type.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Projection {
    /// Perspective projection (frustum).
    Perspective {
        /// Vertical field of view in degrees.
        fov_y_deg: f64,
    },
    /// Orthographic (parallel) projection.
    Orthographic {
        /// Total vertical extent of the visible region.
        parallel_scale: f64,
    },
}

impl Default for Projection {
    fn default() -> Self {
        Self::Perspective { fov_y_deg: 30.0 }
    }
}

// ── Camera ────────────────────────────────────────────────────────────────────

/// Perspective or orthographic camera for volume rendering.
///
/// The camera is defined by:
/// - **position** — the eye point in world space
/// - **focal_point** — the point the camera looks at
/// - **view_up** — the "up" direction hint (orthogonalised internally)
/// - **clip_range** — `(near, far)` clipping distances
/// - **projection** — [`Projection::Perspective`] or [`Projection::Orthographic`]
#[derive(Debug, Clone)]
pub struct Camera {
    position: DVec3,
    focal_point: DVec3,
    view_up: DVec3,
    clip_range: (f64, f64),
    projection: Projection,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: DVec3::new(0.0, 0.0, 1.0),
            focal_point: DVec3::ZERO,
            view_up: DVec3::Y,
            clip_range: (0.01, 1000.0),
            projection: Projection::default(),
        }
    }
}

impl Camera {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a perspective camera with a conventional Y-up configuration.
    #[must_use]
    pub fn new_perspective(position: DVec3, focal_point: DVec3, fov_y_deg: f64) -> Self {
        Self::new(position, focal_point, DVec3::Y)
            .with_projection(Projection::Perspective { fov_y_deg })
    }

    /// Create an orthographic camera with a conventional Y-up configuration.
    #[must_use]
    pub fn new_orthographic(position: DVec3, focal_point: DVec3, parallel_scale: f64) -> Self {
        Self::new(position, focal_point, DVec3::Y)
            .with_projection(Projection::Orthographic { parallel_scale })
    }

    /// Build a new camera pointed at `focal_point` from `position` with view-up `view_up`.
    ///
    /// The clip range defaults to `(0.01, 1000.0)` and can be adjusted with
    /// [`Camera::with_clip_range`].
    #[must_use]
    pub fn new(position: DVec3, focal_point: DVec3, view_up: DVec3) -> Self {
        Self {
            position,
            focal_point,
            view_up: view_up.normalize_or(DVec3::Y),
            clip_range: (0.01, 1000.0),
            projection: Projection::default(),
        }
    }

    /// Override the clip range.
    #[must_use]
    pub fn with_clip_range(mut self, near: f64, far: f64) -> Self {
        debug_assert!(near > 0.0 && far > near, "invalid clip range");
        self.clip_range = (near, far);
        self
    }

    /// Override the projection type.
    #[must_use]
    pub fn with_projection(mut self, projection: Projection) -> Self {
        self.projection = projection;
        self
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Camera eye position in world space.
    #[must_use]
    pub fn position(&self) -> DVec3 {
        self.position
    }

    /// The point the camera looks at.
    #[must_use]
    pub fn focal_point(&self) -> DVec3 {
        self.focal_point
    }

    /// World-space vector pointing from position to focal point (normalised).
    #[must_use]
    pub fn forward(&self) -> DVec3 {
        (self.focal_point - self.position).normalize_or(DVec3::NEG_Z)
    }

    /// Alias for [`Self::forward`], matching the plan terminology.
    #[must_use]
    pub fn direction(&self) -> DVec3 {
        self.forward()
    }

    /// Orthogonalised right vector (forward × up).
    #[must_use]
    pub fn right(&self) -> DVec3 {
        self.forward()
            .cross(self.view_up_ortho())
            .normalize_or(DVec3::X)
    }

    /// Alias for [`Self::right`], matching VTK-style naming.
    #[must_use]
    pub fn right_vector(&self) -> DVec3 {
        self.right()
    }

    /// Orthogonalised view-up vector.
    #[must_use]
    pub fn view_up_ortho(&self) -> DVec3 {
        let fwd = self.forward();
        // Gram-Schmidt: up_ortho = up - (up·fwd)fwd
        let up = self.view_up.normalize_or(DVec3::Y);
        (up - fwd * up.dot(fwd)).normalize_or(DVec3::Y)
    }

    /// The (near, far) clip distances.
    #[must_use]
    pub fn clip_range(&self) -> (f64, f64) {
        self.clip_range
    }

    /// The current projection type.
    #[must_use]
    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    /// Distance from position to focal point.
    #[must_use]
    pub fn distance(&self) -> f64 {
        (self.focal_point - self.position).length()
    }

    // ── Matrices ──────────────────────────────────────────────────────────────

    /// View (world-to-camera) matrix.
    #[must_use]
    pub fn view_matrix(&self) -> DMat4 {
        DMat4::look_at_rh(self.position, self.focal_point, self.view_up_ortho())
    }

    /// Projection matrix for the given viewport aspect ratio.
    ///
    /// `aspect` = viewport_width / viewport_height.
    #[must_use]
    pub fn projection_matrix(&self, aspect: f64) -> DMat4 {
        let (near, far) = self.clip_range;
        match self.projection {
            Projection::Perspective { fov_y_deg } => {
                let fov_rad = fov_y_deg.to_radians();
                DMat4::perspective_rh(fov_rad, aspect, near, far)
            }
            Projection::Orthographic { parallel_scale } => {
                let half_h = parallel_scale;
                let half_w = half_h * aspect;
                DMat4::orthographic_rh(-half_w, half_w, -half_h, half_h, near, far)
            }
        }
    }

    // ── Manipulation ──────────────────────────────────────────────────────────

    /// Move the camera along its forward axis (dolly).
    ///
    /// Positive `delta` moves toward the focal point; negative moves away.
    /// The focal point stays fixed.
    pub fn dolly(&mut self, delta: f64) {
        let dist = self.distance();
        let fwd = self.forward();
        let new_dist = (dist - delta).max(1e-3);
        self.position = self.focal_point - fwd * new_dist;
    }

    /// Multiply the distance to the focal point by `factor` (zoom).
    ///
    /// Values < 1.0 move closer; values > 1.0 move farther.
    pub fn zoom(&mut self, factor: f64) {
        debug_assert!(factor > 0.0, "zoom factor must be positive");
        let fwd = self.forward();
        let new_dist = self.distance() * factor;
        self.position = self.focal_point - fwd * new_dist.max(1e-3);
    }

    /// Translate both position and focal point in screen-space (pan).
    ///
    /// `delta` is in world-space units; use `right()` and `up_ortho()` to
    /// convert from screen pixels.
    pub fn pan(&mut self, delta: DVec3) {
        self.position += delta;
        self.focal_point += delta;
    }

    /// Translate the camera in its view plane.
    ///
    /// `dx` moves along the camera right vector, `dy` along the orthogonal up vector.
    pub fn pan_view(&mut self, dx: f64, dy: f64) {
        self.pan(self.right() * dx + self.view_up_ortho() * dy);
    }

    /// Rotate the camera position around the focal point about the world Y axis.
    ///
    /// This is a convenience wrapper around [`Camera::orbit`].
    pub fn azimuth(&mut self, degrees: f64) {
        self.orbit(degrees.to_radians(), 0.0);
    }

    /// Rotate the camera position around the focal point about the camera's right axis.
    ///
    /// This is a convenience wrapper around [`Camera::orbit`].
    pub fn elevation(&mut self, degrees: f64) {
        self.orbit(0.0, degrees.to_radians());
    }

    /// Rotate the view-up vector around the forward axis by `degrees`.
    pub fn roll(&mut self, degrees: f64) {
        let fwd = self.forward();
        let rot = glam::DQuat::from_axis_angle(fwd, degrees.to_radians());
        self.view_up = rot * self.view_up;
    }

    /// Orbit position around the focal point by rotating `angle_h` about the
    /// world-space `up_axis` and `angle_v` about the camera's right axis.
    ///
    /// Implements an **arcball-style** rotation matching VTK's trackball style.
    pub fn orbit(&mut self, angle_h: f64, angle_v: f64) {
        let to_eye = self.position - self.focal_point;
        let right = self.right();
        let up = DVec3::Y; // orbit around world Y to avoid roll

        // Horizontal rotation around world up
        let rot_h = glam::DQuat::from_axis_angle(up, -angle_h);
        let to_eye = rot_h * to_eye;

        // Vertical rotation around camera right
        let rot_v = glam::DQuat::from_axis_angle(right, -angle_v);
        let to_eye = rot_v * to_eye;

        self.position = self.focal_point + to_eye;
        // Re-project view_up after vertical rotation to avoid roll accumulation
        self.view_up = rot_v * self.view_up;
    }

    /// Auto-fit the camera so the given world-space bounding box is fully visible.
    ///
    /// Sets `clip_range` conservatively around the box.
    pub fn reset_to_bounds(&mut self, bounds_min: DVec3, bounds_max: DVec3, _aspect: f64) {
        let center = (bounds_min + bounds_max) * 0.5;
        let half_diag = (bounds_max - bounds_min).length() * 0.5;

        let fov_rad = match self.projection {
            Projection::Perspective { fov_y_deg } => fov_y_deg.to_radians(),
            Projection::Orthographic { .. } => std::f64::consts::FRAC_PI_4,
        };
        let dist = half_diag / (fov_rad * 0.5).tan();

        let fwd = self.forward();
        self.focal_point = center;
        self.position = center - fwd * dist;
        self.clip_range = ((dist - half_diag * 1.5).max(1e-3), dist + half_diag * 1.5);

        if let Projection::Orthographic {
            ref mut parallel_scale,
        } = self.projection
        {
            *parallel_scale = half_diag;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn view_matrix_look_at_negative_z() {
        let cam = Camera::new(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y);
        let vm = cam.view_matrix();
        // A point at the focal point should map to camera-space origin (roughly).
        let cam_space = vm.transform_point3(cam.focal_point().as_vec3().as_dvec3());
        assert_abs_diff_eq!(cam_space.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cam_space.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn orbit_preserves_distance() {
        let mut cam = Camera::new(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y);
        let d0 = cam.distance();
        cam.orbit(0.3, 0.2);
        let d1 = cam.distance();
        assert_abs_diff_eq!(d0, d1, epsilon = 1e-8);
    }

    #[test]
    fn dolly_changes_distance() {
        let mut cam = Camera::new(DVec3::new(0.0, 0.0, 10.0), DVec3::ZERO, DVec3::Y);
        cam.dolly(2.0);
        assert_abs_diff_eq!(cam.distance(), 8.0, epsilon = 1e-8);
    }

    #[test]
    fn zoom_changes_distance() {
        let mut cam = Camera::new(DVec3::new(0.0, 0.0, 10.0), DVec3::ZERO, DVec3::Y);
        cam.zoom(0.5);
        assert_abs_diff_eq!(cam.distance(), 5.0, epsilon = 1e-8);
    }

    #[test]
    fn view_up_orthogonal_to_forward() {
        let cam = Camera::new(DVec3::new(1.0, 2.0, 3.0), DVec3::ZERO, DVec3::Y);
        let dot = cam.forward().dot(cam.view_up_ortho());
        assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn right_is_normalised() {
        let cam = Camera::new(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y);
        assert_abs_diff_eq!(cam.right().length(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn pan_moves_focal_point() {
        let mut cam = Camera::new(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y);
        cam.pan(DVec3::new(1.0, 0.0, 0.0));
        assert_abs_diff_eq!(cam.focal_point().x, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cam.position().x, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn azimuth_preserves_distance() {
        let mut cam = Camera::new(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y);
        let d0 = cam.distance();
        cam.azimuth(45.0);
        assert_abs_diff_eq!(cam.distance(), d0, epsilon = 1e-8);
    }

    #[test]
    fn elevation_changes_position() {
        let mut cam = Camera::new(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y);
        let y0 = cam.position().y;
        cam.elevation(30.0);
        assert!(
            (cam.position().y - y0).abs() > 0.1,
            "elevation should move camera vertically"
        );
    }

    #[test]
    fn roll_changes_right_vector() {
        let mut cam = Camera::new(DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, DVec3::Y);
        let r0 = cam.right();
        cam.roll(45.0);
        let r1 = cam.right();
        assert!(
            (r1 - r0).length() > 0.1,
            "roll should change the right vector"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn view_matrix_is_orthonormal(
            x in -10.0f64..10.0,
            y in -10.0f64..10.0,
            z in 1.0f64..20.0,
        ) {
            let cam = Camera::new(DVec3::new(x, y, z), DVec3::ZERO, DVec3::Y);
            let vm = cam.view_matrix();
            // Extract 3×3 upper-left; columns should be orthonormal → det ≈ 1
            let m3 = glam::DMat3::from_cols(
                vm.col(0).truncate(),
                vm.col(1).truncate(),
                vm.col(2).truncate(),
            );
            let det = m3.determinant();
            prop_assert!((det - 1.0).abs() < 1e-6, "det = {det}");
        }
    }
}
