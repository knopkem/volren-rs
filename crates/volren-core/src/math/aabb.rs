//! Axis-aligned bounding box.

use glam::DVec3;

/// An axis-aligned bounding box in 3D world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner (smallest x, y, z).
    pub min: DVec3,
    /// Maximum corner (largest x, y, z).
    pub max: DVec3,
}

impl Aabb {
    /// Create a new AABB from min and max corners.
    ///
    /// # Panics
    /// Panics in debug mode if `min > max` on any axis.
    #[must_use]
    pub fn new(min: DVec3, max: DVec3) -> Self {
        debug_assert!(
            min.x <= max.x && min.y <= max.y && min.z <= max.z,
            "AABB min must be <= max on all axes"
        );
        Self { min, max }
    }

    /// Create from a center point and half-extents.
    #[must_use]
    pub fn from_center_half_extents(center: DVec3, half: DVec3) -> Self {
        Self { min: center - half, max: center + half }
    }

    /// The geometric center of this AABB.
    #[must_use]
    pub fn center(&self) -> DVec3 {
        (self.min + self.max) * 0.5
    }

    /// The size (extent) of each axis.
    #[must_use]
    pub fn size(&self) -> DVec3 {
        self.max - self.min
    }

    /// The diagonal length of the AABB.
    #[must_use]
    pub fn diagonal(&self) -> f64 {
        self.size().length()
    }

    /// Returns `true` if `point` is contained within or on the boundary.
    #[must_use]
    pub fn contains(&self, point: DVec3) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }

    /// Expand the AABB to include `point`.
    #[must_use]
    pub fn expanded_to_include(&self, point: DVec3) -> Self {
        Self { min: self.min.min(point), max: self.max.max(point) }
    }

    /// Intersect a ray (origin + t*direction) against this AABB using the slab method.
    ///
    /// Returns `Some((t_near, t_far))` if the ray intersects (t_far >= t_near >= 0),
    /// or `None` if the ray misses.
    #[must_use]
    pub fn intersect_ray(&self, origin: DVec3, inv_dir: DVec3) -> Option<(f64, f64)> {
        let t1 = (self.min - origin) * inv_dir;
        let t2 = (self.max - origin) * inv_dir;

        let t_near = t1.min(t2);
        let t_far = t1.max(t2);

        let t_enter = t_near.x.max(t_near.y).max(t_near.z);
        let t_exit = t_far.x.min(t_far.y).min(t_far.z);

        if t_exit >= t_enter.max(0.0) {
            Some((t_enter.max(0.0), t_exit))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn center_is_midpoint() {
        let aabb = Aabb::new(DVec3::ZERO, DVec3::splat(2.0));
        let center = aabb.center();
        assert_abs_diff_eq!(center.x, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(center.y, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(center.z, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn size_and_diagonal() {
        let aabb = Aabb::new(DVec3::ZERO, DVec3::new(3.0, 4.0, 0.0));
        assert_abs_diff_eq!(aabb.size().x, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(aabb.size().y, 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(aabb.diagonal(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn contains_boundary() {
        let aabb = Aabb::new(DVec3::ZERO, DVec3::ONE);
        assert!(aabb.contains(DVec3::ZERO));
        assert!(aabb.contains(DVec3::ONE));
        assert!(aabb.contains(DVec3::splat(0.5)));
        assert!(!aabb.contains(DVec3::new(1.1, 0.5, 0.5)));
    }

    #[test]
    fn ray_hit_axis_aligned() {
        let aabb = Aabb::new(DVec3::ZERO, DVec3::ONE);
        // Ray along +X from outside
        let origin = DVec3::new(-1.0, 0.5, 0.5);
        let dir = DVec3::new(1.0, 0.0, 0.0);
        let inv_dir = DVec3::ONE / dir;
        let hit = aabb.intersect_ray(origin, inv_dir);
        assert!(hit.is_some());
        let (t_near, t_far) = hit.unwrap();
        assert_abs_diff_eq!(t_near, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(t_far, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn ray_miss() {
        let aabb = Aabb::new(DVec3::ZERO, DVec3::ONE);
        let origin = DVec3::new(5.0, 5.0, 5.0);
        let dir = DVec3::new(1.0, 0.0, 0.0);
        let inv_dir = DVec3::ONE / dir;
        assert!(aabb.intersect_ray(origin, inv_dir).is_none());
    }
}
