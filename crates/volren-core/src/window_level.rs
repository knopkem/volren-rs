//! DICOM-compliant window/level (window centre/width) mapping.
//!
//! Implements the formula from **DICOM PS 3.3 §C.7.6.3.1.5**:
//!
//! ```text
//! if (value - window_center + 0.5) / window_width + 0.5 ≤ 0   → y_min
//! if (value - window_center + 0.5) / window_width + 0.5 ≥ 1   → y_max
//! else                                                          → linear
//! ```

/// DICOM window/level parameters.
///
/// - `center` — the value that maps to the midpoint of the output range.
/// - `width`  — the full span of input values that map to the full output range.
///
/// A width of zero is not meaningful; the caller should ensure `width > 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowLevel {
    /// Window centre (level).
    pub center: f64,
    /// Window width.
    pub width: f64,
}

impl WindowLevel {
    /// Create from center and width.
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `width <= 0`.
    #[must_use]
    pub fn new(center: f64, width: f64) -> Self {
        debug_assert!(width > 0.0, "window width must be positive");
        Self { center, width }
    }

    /// Apply the DICOM linear mapping and return a value in `[out_min, out_max]`.
    ///
    /// `out_min` and `out_max` are typically `0.0` and `1.0` for GPU normalisation,
    /// or `0.0` and `255.0` for 8-bit display.
    #[must_use]
    pub fn apply(&self, value: f64, out_min: f64, out_max: f64) -> f64 {
        // DICOM formula (PS 3.3 C.7.6.3.1.5)
        let t = (value - self.center + 0.5) / self.width + 0.5;
        let t = t.clamp(0.0, 1.0);
        out_min + t * (out_max - out_min)
    }

    /// Normalise `value` to `[0, 1]` using the window mapping.
    #[inline]
    #[must_use]
    pub fn normalise(&self, value: f64) -> f64 {
        self.apply(value, 0.0, 1.0)
    }

    /// Return the input value that maps to `t ∈ [0, 1]`.
    ///
    /// This is the inverse of [`WindowLevel::normalise`].
    #[must_use]
    pub fn denormalise(&self, t: f64) -> f64 {
        (t - 0.5) * self.width + self.center - 0.5
    }

    /// Adjust the center by `delta` (level change).
    pub fn adjust_center(&mut self, delta: f64) {
        self.center += delta;
    }

    /// Adjust the width by `factor` (window change, multiplicative).
    ///
    /// The width is clamped to at least 1.0 to avoid division by zero.
    pub fn adjust_width(&mut self, factor: f64) {
        self.width = (self.width * factor).max(1.0);
    }

    /// Derive window/level from a scalar range.
    ///
    /// Sets center to the midpoint and width to the full range (minimum 1.0).
    #[must_use]
    pub fn from_scalar_range(min: f64, max: f64) -> Self {
        Self::new((min + max) * 0.5, (max - min).max(1.0))
    }
}

/// Common CT window presets (Hounsfield units).
pub mod presets {
    use super::WindowLevel;

    /// Soft tissue window (C 40, W 400).
    pub const SOFT_TISSUE: WindowLevel = WindowLevel { center: 40.0, width: 400.0 };

    /// Lung window (C −600, W 1500).
    pub const LUNG: WindowLevel = WindowLevel { center: -600.0, width: 1500.0 };

    /// Bone window (C 400, W 1500).
    pub const BONE: WindowLevel = WindowLevel { center: 400.0, width: 1500.0 };

    /// Brain window (C 40, W 80).
    pub const BRAIN: WindowLevel = WindowLevel { center: 40.0, width: 80.0 };

    /// Liver window (C 60, W 160).
    pub const LIVER: WindowLevel = WindowLevel { center: 60.0, width: 160.0 };

    /// Abdomen window (C 60, W 400).
    pub const ABDOMEN: WindowLevel = WindowLevel { center: 60.0, width: 400.0 };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn center_maps_to_midpoint() {
        let wl = WindowLevel::new(40.0, 400.0);
        // Per DICOM PS3.3 §C.7.6.3.1.5, value=(center-0.5) maps to exactly 0.5
        assert_abs_diff_eq!(wl.normalise(39.5), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn below_window_clamps_to_zero() {
        let wl = WindowLevel::new(40.0, 400.0);
        assert_abs_diff_eq!(wl.normalise(-160.0 - 1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn above_window_clamps_to_one() {
        let wl = WindowLevel::new(40.0, 400.0);
        assert_abs_diff_eq!(wl.normalise(240.0 + 1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn normalise_denormalise_round_trip() {
        let wl = WindowLevel::new(100.0, 200.0);
        // Round-trip works for values strictly inside the window (not clamped).
        // With c=100, w=200: unclamped range is [c-0.5-w/2, c-0.5+w/2] = [-0.5, 199.5]
        for v in [0.0, 50.0, 100.0, 150.0, 199.0] {
            let t = wl.normalise(v);
            let back = wl.denormalise(t);
            assert_abs_diff_eq!(back, v, epsilon = 1e-8);
        }
    }

    #[test]
    fn apply_custom_range() {
        let wl = WindowLevel::new(0.0, 2.0);
        // value=center-0.5 → t=0.5 exactly (DICOM spec)
        let out = wl.apply(-0.5, 0.0, 255.0);
        assert_abs_diff_eq!(out, 127.5, epsilon = 0.5);
    }

    #[test]
    fn presets_are_valid() {
        // Just check no panic and sensible ranges
        assert!(presets::SOFT_TISSUE.width > 0.0);
        assert!(presets::LUNG.width > 0.0);
        assert!(presets::BONE.width > 0.0);
    }

    #[test]
    fn from_scalar_range() {
        let wl = WindowLevel::from_scalar_range(100.0, 300.0);
        assert_abs_diff_eq!(wl.center, 200.0, epsilon = 1e-10);
        assert_abs_diff_eq!(wl.width, 200.0, epsilon = 1e-10);
    }

    #[test]
    fn from_scalar_range_degenerate() {
        let wl = WindowLevel::from_scalar_range(50.0, 50.0);
        assert!(wl.width >= 1.0, "width should be at least 1.0");
    }
}
