//! Two-dimensional transfer function: `(scalar, gradient)` to RGBA.

/// An axis-aligned region in `(scalar, gradient)` space.
///
/// Regions are composited in insertion order using source-over alpha blending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferFunction2DRegion {
    /// Inclusive scalar range covered by the region.
    pub scalar_range: [f64; 2],
    /// Inclusive gradient-magnitude range covered by the region.
    pub gradient_range: [f64; 2],
    /// RGBA colour produced when the sample falls inside the region.
    pub rgba: [f64; 4],
}

impl TransferFunction2DRegion {
    /// Create a new region.
    #[must_use]
    pub fn new(scalar_range: [f64; 2], gradient_range: [f64; 2], rgba: [f64; 4]) -> Self {
        Self {
            scalar_range,
            gradient_range,
            rgba,
        }
    }

    /// `true` if `(scalar, gradient)` lies inside the region.
    #[must_use]
    pub fn contains(&self, scalar: f64, gradient: f64) -> bool {
        scalar >= self.scalar_range[0]
            && scalar <= self.scalar_range[1]
            && gradient >= self.gradient_range[0]
            && gradient <= self.gradient_range[1]
    }
}

/// A 2D transfer function mapping `(scalar, gradient magnitude)` to RGBA.
///
/// This is a practical building block for feature classification: for example,
/// highlight bone-like CT values only when the gradient magnitude is high.
#[derive(Debug, Clone, Default)]
pub struct TransferFunction2D {
    regions: Vec<TransferFunction2DRegion>,
    background: [f64; 4],
}

impl TransferFunction2D {
    /// Create an empty 2D transfer function with transparent background.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the fallback RGBA returned when no region matches.
    #[must_use]
    pub fn with_background(mut self, rgba: [f64; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// Add a region.
    pub fn add_region(&mut self, region: TransferFunction2DRegion) {
        self.regions.push(region);
    }

    /// Remove the region at `index`, if it exists.
    pub fn remove_region(&mut self, index: usize) -> Option<TransferFunction2DRegion> {
        if index < self.regions.len() {
            Some(self.regions.remove(index))
        } else {
            None
        }
    }

    /// Borrow all configured regions.
    #[must_use]
    pub fn regions(&self) -> &[TransferFunction2DRegion] {
        &self.regions
    }

    /// Evaluate the transfer function.
    ///
    /// Matching regions are composited in insertion order.
    #[must_use]
    pub fn evaluate(&self, scalar: f64, gradient: f64) -> [f64; 4] {
        self.regions
            .iter()
            .filter(|region| region.contains(scalar, gradient))
            .fold(self.background, |dst, region| alpha_over(dst, region.rgba))
    }
}

fn alpha_over(dst: [f64; 4], src: [f64; 4]) -> [f64; 4] {
    let src_a = src[3].clamp(0.0, 1.0);
    let dst_a = dst[3].clamp(0.0, 1.0);
    let out_a = src_a + dst_a * (1.0 - src_a);
    if out_a <= f64::EPSILON {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let blend_channel =
        |src_c: f64, dst_c: f64| (src_c * src_a + dst_c * dst_a * (1.0 - src_a)) / out_a;

    [
        blend_channel(src[0], dst[0]),
        blend_channel(src[1], dst[1]),
        blend_channel(src[2], dst[2]),
        out_a,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn empty_returns_background() {
        let tf = TransferFunction2D::new().with_background([0.1, 0.2, 0.3, 0.4]);
        assert_eq!(tf.evaluate(1.0, 2.0), [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn matching_region_returns_rgba() {
        let mut tf = TransferFunction2D::new();
        tf.add_region(TransferFunction2DRegion::new(
            [100.0, 200.0],
            [0.0, 1.0],
            [1.0, 0.9, 0.8, 0.7],
        ));
        let rgba = tf.evaluate(150.0, 0.5);
        assert_abs_diff_eq!(rgba[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rgba[1], 0.9, epsilon = 1e-12);
        assert_abs_diff_eq!(rgba[2], 0.8, epsilon = 1e-12);
        assert_abs_diff_eq!(rgba[3], 0.7, epsilon = 1e-12);
    }

    #[test]
    fn non_matching_region_ignored() {
        let mut tf = TransferFunction2D::new();
        tf.add_region(TransferFunction2DRegion::new(
            [100.0, 200.0],
            [2.0, 3.0],
            [1.0, 0.0, 0.0, 1.0],
        ));
        assert_eq!(tf.evaluate(150.0, 0.5), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn overlapping_regions_alpha_composite() {
        let mut tf = TransferFunction2D::new();
        tf.add_region(TransferFunction2DRegion::new(
            [0.0, 10.0],
            [0.0, 10.0],
            [1.0, 0.0, 0.0, 0.5],
        ));
        tf.add_region(TransferFunction2DRegion::new(
            [0.0, 10.0],
            [0.0, 10.0],
            [0.0, 0.0, 1.0, 0.5],
        ));

        let rgba = tf.evaluate(5.0, 5.0);
        assert_abs_diff_eq!(rgba[3], 0.75, epsilon = 1e-12);
    }
}
