//! Colour transfer function: maps a scalar value to an RGB colour.

/// Colour interpolation space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ColorSpace {
    /// Interpolate in linear RGB.
    #[default]
    Rgb,
    /// Interpolate in HSV (shorter arc).
    Hsv,
    /// Interpolate in CIE L\*a\*b\* (perceptually uniform).
    ///
    /// Note: Currently falls back to RGB interpolation. A proper CIE Lab
    /// implementation would require an external colour science dependency
    /// and is planned as a future enhancement.
    Lab,
}

/// A piecewise-linear colour transfer function.
///
/// Control points map a scalar value to an `[R, G, B]` triple in `[0, 1]`.
/// Between control points the colour is linearly interpolated in the
/// configured [`ColorSpace`].
#[derive(Debug, Clone)]
pub struct ColorTransferFunction {
    points: Vec<(f64, [f64; 3])>,
    color_space: ColorSpace,
}

impl ColorTransferFunction {
    /// Create an empty function. Points must be added via [`Self::add_point`].
    #[must_use]
    pub fn new(color_space: ColorSpace) -> Self {
        Self { points: Vec::new(), color_space }
    }

    /// Create a greyscale ramp: black at `scalar_min`, white at `scalar_max`.
    #[must_use]
    pub fn greyscale(scalar_min: f64, scalar_max: f64) -> Self {
        let mut ctf = Self::new(ColorSpace::Rgb);
        ctf.add_point(scalar_min, [0.0, 0.0, 0.0]);
        ctf.add_point(scalar_max, [1.0, 1.0, 1.0]);
        ctf
    }

    /// Add or replace a control point.
    ///
    /// If a point at the same scalar already exists it is replaced.
    pub fn add_point(&mut self, scalar: f64, rgb: [f64; 3]) {
        match self.points.binary_search_by(|(s, _)| s.partial_cmp(&scalar).unwrap()) {
            Ok(pos) => self.points[pos] = (scalar, rgb),
            Err(pos) => self.points.insert(pos, (scalar, rgb)),
        }
    }

    /// Remove a control point closest to `scalar` (within `epsilon`).
    pub fn remove_point(&mut self, scalar: f64, epsilon: f64) {
        if let Some(pos) =
            self.points.iter().position(|(s, _)| (s - scalar).abs() < epsilon)
        {
            self.points.remove(pos);
        }
    }

    /// Evaluate the colour at `scalar`. Returns `[0, 0, 0]` if no points exist.
    #[must_use]
    pub fn evaluate(&self, scalar: f64) -> [f64; 3] {
        if self.points.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        if scalar <= self.points.first().unwrap().0 {
            return self.points.first().unwrap().1;
        }
        if scalar >= self.points.last().unwrap().0 {
            return self.points.last().unwrap().1;
        }
        // Binary search for the enclosing segment
        let pos = self
            .points
            .partition_point(|(s, _)| *s <= scalar)
            .saturating_sub(1);
        let (s0, c0) = self.points[pos];
        let (s1, c1) = self.points[pos + 1];
        let t = (scalar - s0) / (s1 - s0);
        match self.color_space {
            ColorSpace::Rgb | ColorSpace::Lab => lerp_rgb(c0, c1, t),
            ColorSpace::Hsv => lerp_hsv(c0, c1, t),
        }
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

// ── Helpers ───────────────────────────────────────────────────────────────────

fn lerp_rgb(c0: [f64; 3], c1: [f64; 3], t: f64) -> [f64; 3] {
    [
        c0[0] + (c1[0] - c0[0]) * t,
        c0[1] + (c1[1] - c0[1]) * t,
        c0[2] + (c1[2] - c0[2]) * t,
    ]
}

fn rgb_to_hsv(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };
    let h = if delta < 1e-10 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    [h, s, v]
}

fn hsv_to_rgb(hsv: [f64; 3]) -> [f64; 3] {
    let [h, s, v] = hsv;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r1 + m, g1 + m, b1 + m]
}

fn lerp_hsv(c0: [f64; 3], c1: [f64; 3], t: f64) -> [f64; 3] {
    let h0 = rgb_to_hsv(c0);
    let h1 = rgb_to_hsv(c1);
    // Interpolate hue along the shorter arc
    let mut dh = h1[0] - h0[0];
    if dh > 180.0 {
        dh -= 360.0;
    } else if dh < -180.0 {
        dh += 360.0;
    }
    let h = h0[0] + dh * t;
    let s = h0[1] + (h1[1] - h0[1]) * t;
    let v = h0[2] + (h1[2] - h0[2]) * t;
    hsv_to_rgb([h, s, v])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn evaluate_below_first_returns_first_color() {
        let ctf = ColorTransferFunction::greyscale(0.0, 1.0);
        let c = ctf.evaluate(-1.0);
        assert_abs_diff_eq!(c[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn evaluate_above_last_returns_last_color() {
        let ctf = ColorTransferFunction::greyscale(0.0, 1.0);
        let c = ctf.evaluate(2.0);
        assert_abs_diff_eq!(c[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn evaluate_midpoint() {
        let ctf = ColorTransferFunction::greyscale(0.0, 2.0);
        let c = ctf.evaluate(1.0);
        assert_abs_diff_eq!(c[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn add_and_remove_point() {
        let mut ctf = ColorTransferFunction::new(ColorSpace::Rgb);
        ctf.add_point(0.0, [0.0, 0.0, 0.0]);
        ctf.add_point(1.0, [1.0, 1.0, 1.0]);
        assert_eq!(ctf.len(), 2);
        ctf.remove_point(1.0, 0.01);
        assert_eq!(ctf.len(), 1);
    }

    #[test]
    fn replace_existing_point() {
        let mut ctf = ColorTransferFunction::new(ColorSpace::Rgb);
        ctf.add_point(0.5, [0.1, 0.2, 0.3]);
        ctf.add_point(0.5, [0.9, 0.8, 0.7]);
        assert_eq!(ctf.len(), 1);
        let c = ctf.evaluate(0.5);
        assert_abs_diff_eq!(c[0], 0.9, epsilon = 1e-10);
    }

    #[test]
    fn empty_returns_black() {
        let ctf = ColorTransferFunction::new(ColorSpace::Rgb);
        let c = ctf.evaluate(0.5);
        assert_abs_diff_eq!(c[0], 0.0, epsilon = 1e-10);
    }
}
