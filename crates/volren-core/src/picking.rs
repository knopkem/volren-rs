//! CPU-based ray-volume intersection for picking.

use glam::{DVec2, DVec3, DVec4, UVec2};

use crate::camera::Camera;
use crate::math::Aabb;
use crate::volume::{DynVolume, VolumeInfo};
use crate::render_params::VolumeRenderParams;

/// A ray in world space.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Ray origin.
    pub origin: DVec3,
    /// Normalised ray direction.
    pub direction: DVec3,
}

impl Ray {
    /// Create a new ray. `direction` is normalised internally.
    #[must_use]
    pub fn new(origin: DVec3, direction: DVec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Evaluate the ray at parameter `t`.
    #[must_use]
    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.direction * t
    }

    /// Intersect the ray with an axis-aligned bounding box.
    ///
    /// Returns `(t_enter, t_exit)` if the ray intersects the box, or `None` if
    /// it misses. `t_enter` may be negative if the ray starts inside the box.
    #[must_use]
    pub fn intersect_aabb(&self, aabb: &Aabb) -> Option<(f64, f64)> {
        let inv_dir = DVec3::new(
            1.0 / self.direction.x,
            1.0 / self.direction.y,
            1.0 / self.direction.z,
        );

        let t1 = (aabb.min - self.origin) * inv_dir;
        let t2 = (aabb.max - self.origin) * inv_dir;

        let t_min = t1.min(t2);
        let t_max = t1.max(t2);

        let t_enter = t_min.x.max(t_min.y).max(t_min.z);
        let t_exit = t_max.x.min(t_max.y).min(t_max.z);

        if t_enter <= t_exit && t_exit >= 0.0 {
            Some((t_enter, t_exit))
        } else {
            None
        }
    }
}

/// Unproject a screen-space pixel coordinate into a world-space ray.
///
/// `screen_pos` is in pixel coordinates (top-left origin).
/// `viewport_size` is width × height in pixels.
#[must_use]
pub fn unproject_ray(screen_pos: DVec2, camera: &Camera, viewport_size: UVec2) -> Ray {
    let aspect = f64::from(viewport_size.x) / f64::from(viewport_size.y);
    let view = camera.view_matrix();
    let proj = camera.projection_matrix(aspect);
    let inv_vp = (proj * view).inverse();

    // Normalise to NDC [-1, 1]
    let ndc_x = (screen_pos.x / f64::from(viewport_size.x)) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / f64::from(viewport_size.y)) * 2.0;

    let near_clip = inv_vp * DVec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let far_clip = inv_vp * DVec4::new(ndc_x, ndc_y, 1.0, 1.0);

    let near_world = near_clip.truncate() / near_clip.w;
    let far_world = far_clip.truncate() / far_clip.w;

    Ray::new(near_world, far_world - near_world)
}

/// Result of picking a point on a volume.
#[derive(Debug, Clone)]
pub struct PickResult {
    /// World-space position of the hit point.
    pub world_position: DVec3,
    /// Continuous voxel index (fractional) of the hit point.
    pub voxel_index: DVec3,
    /// Interpolated scalar value at the hit point.
    pub voxel_value: f64,
}

/// Cast a ray from screen coordinates through the volume.
///
/// Marches along the ray in small steps (determined by `params.step_size_factor`)
/// and returns the first hit where the opacity transfer function maps to
/// a non-trivial value (> 0.01).
///
/// Returns `None` if no hit is found.
#[must_use]
pub fn pick_volume(
    screen_pos: DVec2,
    camera: &Camera,
    viewport_size: UVec2,
    volume: &DynVolume,
    params: &VolumeRenderParams,
) -> Option<PickResult> {
    let ray = unproject_ray(screen_pos, camera, viewport_size);
    let bounds = volume.world_bounds();

    let (t_enter, t_exit) = ray.intersect_aabb(&bounds)?;
    let t_start = t_enter.max(0.0);

    // Step size in world units — fraction of the smallest voxel dimension
    let spacing = volume.spacing();
    let min_spacing = spacing.x.min(spacing.y).min(spacing.z);
    let step = min_spacing * f64::from(params.step_size_factor);

    let mut t = t_start;
    while t <= t_exit {
        let world_pos = ray.at(t);
        let voxel_idx = volume.world_to_index(world_pos);

        if let Some(value) = volume.sample_linear(voxel_idx) {
            let opacity = params.opacity_tf.evaluate(value);

            if opacity > 0.01 {
                return Some(PickResult {
                    world_position: world_pos,
                    voxel_index: voxel_idx,
                    voxel_value: value,
                });
            }
        }

        t += step;
    }

    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn ray_intersect_aabb_hit() {
        let ray = Ray::new(DVec3::new(-5.0, 0.5, 0.5), DVec3::X);
        let aabb = Aabb::new(DVec3::ZERO, DVec3::ONE);
        let (t_enter, t_exit) = ray.intersect_aabb(&aabb).expect("should hit");
        assert_abs_diff_eq!(t_enter, 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(t_exit, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn ray_intersect_aabb_miss() {
        let ray = Ray::new(DVec3::new(-5.0, 5.0, 0.5), DVec3::X);
        let aabb = Aabb::new(DVec3::ZERO, DVec3::ONE);
        assert!(ray.intersect_aabb(&aabb).is_none());
    }

    #[test]
    fn ray_intersect_aabb_inside() {
        let ray = Ray::new(DVec3::new(0.5, 0.5, 0.5), DVec3::X);
        let aabb = Aabb::new(DVec3::ZERO, DVec3::ONE);
        let (t_enter, t_exit) = ray.intersect_aabb(&aabb).expect("should hit");
        assert!(t_enter < 0.0, "origin is inside, t_enter should be negative");
        assert!(t_exit > 0.0);
    }

    #[test]
    fn ray_at() {
        let ray = Ray::new(DVec3::ZERO, DVec3::X);
        let p = ray.at(3.0);
        assert_abs_diff_eq!(p.x, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn unproject_center_looks_forward() {
        let cam = Camera::new(
            DVec3::new(0.0, 0.0, 5.0),
            DVec3::ZERO,
            DVec3::Y,
        );
        let viewport = UVec2::new(800, 600);
        let ray = unproject_ray(DVec2::new(400.0, 300.0), &cam, viewport);
        // Ray should point roughly towards -Z (toward focal point)
        assert!(ray.direction.z < 0.0, "ray should point toward -Z");
        assert_abs_diff_eq!(ray.direction.x, 0.0, epsilon = 0.01);
        assert_abs_diff_eq!(ray.direction.y, 0.0, epsilon = 0.01);
    }
}
