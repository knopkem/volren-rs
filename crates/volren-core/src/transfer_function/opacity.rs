//! Opacity (scalar opacity) transfer function.

/// A piecewise-linear opacity transfer function.
///
/// Maps a scalar value to an opacity in `[0.0, 1.0]`.
/// Control points are stored sorted by scalar value.
///
/// # VTK Equivalent
/// `vtkPiecewiseFunction` used as `vtkVolumeProperty::ScalarOpacity`.
#[derive(Debug, Clone)]
pub struct OpacityTransferFunction {
    points: Vec<(f64, f64)>,
}

impl OpacityTransferFunction {
    /// Create an empty opacity function.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Create a linear ramp: fully transparent at `scalar_min`, fully opaque at `scalar_max`.
    #[must_use]
    pub fn linear_ramp(scalar_min: f64, scalar_max: f64) -> Self {
        let mut otf = Self::new();
        otf.add_point(scalar_min, 0.0);
        otf.add_point(scalar_max, 1.0);
        otf
    }

    /// Add or replace a control point.
    ///
    /// `opacity` is clamped to `[0, 1]`.
    pub fn add_point(&mut self, scalar: f64, opacity: f64) {
        let opacity = opacity.clamp(0.0, 1.0);
        match self.points.binary_search_by(|(s, _)| s.partial_cmp(&scalar).unwrap()) {
            Ok(pos) => self.points[pos] = (scalar, opacity),
            Err(pos) => self.points.insert(pos, (scalar, opacity)),
        }
    }

    /// Remove a control point within `epsilon` of `scalar`.
    pub fn remove_point(&mut self, scalar: f64, epsilon: f64) {
        if let Some(pos) = self.points.iter().position(|(s, _)| (s - scalar).abs() < epsilon) {
            self.points.remove(pos);
        }
    }

    /// Evaluate opacity at `scalar`. Returns `0.0` if no points exist.
    #[must_use]
    pub fn evaluate(&self, scalar: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        if scalar <= self.points.first().unwrap().0 {
            return self.points.first().unwrap().1;
        }
        if scalar >= self.points.last().unwrap().0 {
            return self.points.last().unwrap().1;
        }
        let pos = self.points.partition_point(|(s, _)| *s <= scalar).saturating_sub(1);
        let (s0, a0) = self.points[pos];
        let (s1, a1) = self.points[pos + 1];
        let t = (scalar - s0) / (s1 - s0);
        a0 + (a1 - a0) * t
    }

    /// Number of control points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// `true` if no control points have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl Default for OpacityTransferFunction {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn linear_ramp_midpoint() {
        let otf = OpacityTransferFunction::linear_ramp(0.0, 1.0);
        assert_abs_diff_eq!(otf.evaluate(0.5), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn clamp_opacity_to_one() {
        let mut otf = OpacityTransferFunction::new();
        otf.add_point(0.0, 1.5);
        assert_abs_diff_eq!(otf.evaluate(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn empty_returns_zero() {
        assert_abs_diff_eq!(OpacityTransferFunction::new().evaluate(0.5), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn monotone_ramp_is_monotone() {
        let otf = OpacityTransferFunction::linear_ramp(-1000.0, 1000.0);
        let mut prev = otf.evaluate(-1000.0);
        for i in -999..=1000 {
            let v = otf.evaluate(i as f64);
            assert!(v >= prev - 1e-12, "monotonicity violated at {i}");
            prev = v;
        }
    }
}
