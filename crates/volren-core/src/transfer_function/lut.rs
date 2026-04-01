//! LUT baking: combines colour and opacity functions into a single 1D RGBA texture.

use super::{ColorTransferFunction, OpacityTransferFunction};

/// A baked 1D RGBA lookup table suitable for GPU upload.
///
/// Produced by [`TransferFunctionLut::bake`] from a [`ColorTransferFunction`]
/// and an [`OpacityTransferFunction`].
///
/// The table maps a normalised scalar `t ∈ [0, 1]` (where 0 = `scalar_min`,
/// 1 = `scalar_max`) to an RGBA value in `[0, 1]`.
///
/// # GPU Upload
/// Call [`TransferFunctionLut::as_rgba_f32`] to obtain the raw data, then
/// upload as a 1D `R32G32B32A32Float` texture.
#[derive(Debug, Clone)]
pub struct TransferFunctionLut {
    /// Raw RGBA data, `lut_size * 4` entries, each in `[0, 1]`.
    rgba: Vec<f32>,
    /// Number of entries in the LUT.
    lut_size: u32,
    /// The scalar value at the start of the LUT (t = 0).
    scalar_min: f64,
    /// The scalar value at the end of the LUT (t = 1).
    scalar_max: f64,
}

impl TransferFunctionLut {
    /// Bake a LUT from a colour and opacity transfer function.
    ///
    /// `scalar_min`/`scalar_max` define the mapping range; `lut_size` is the
    /// number of texture texels (256 is typical).
    ///
    /// # Panics (debug only)
    /// Panics if `lut_size == 0` or `scalar_min >= scalar_max`.
    #[must_use]
    pub fn bake(
        ctf: &ColorTransferFunction,
        otf: &OpacityTransferFunction,
        scalar_min: f64,
        scalar_max: f64,
        lut_size: u32,
    ) -> Self {
        debug_assert!(lut_size > 0, "lut_size must be > 0");
        debug_assert!(scalar_max > scalar_min, "scalar_max must be > scalar_min");

        let mut rgba = Vec::with_capacity(lut_size as usize * 4);
        let range = scalar_max - scalar_min;

        for i in 0..lut_size {
            let t = i as f64 / (lut_size - 1).max(1) as f64;
            let scalar = scalar_min + t * range;
            let [r, g, b] = ctf.evaluate(scalar);
            let a = otf.evaluate(scalar);
            rgba.push(r as f32);
            rgba.push(g as f32);
            rgba.push(b as f32);
            rgba.push(a as f32);
        }

        Self { rgba, lut_size, scalar_min, scalar_max }
    }

    /// Raw RGBA `f32` slice, suitable for GPU texture upload.
    ///
    /// Length = `lut_size * 4`.
    #[must_use]
    pub fn as_rgba_f32(&self) -> &[f32] {
        &self.rgba
    }

    /// Raw bytes, suitable for `wgpu::Queue::write_texture`.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.rgba)
    }

    /// Number of texels in the LUT.
    #[must_use]
    pub fn lut_size(&self) -> u32 {
        self.lut_size
    }

    /// Scalar value at `t = 0`.
    #[must_use]
    pub fn scalar_min(&self) -> f64 {
        self.scalar_min
    }

    /// Scalar value at `t = 1`.
    #[must_use]
    pub fn scalar_max(&self) -> f64 {
        self.scalar_max
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer_function::{ColorTransferFunction, OpacityTransferFunction};
    use approx::assert_abs_diff_eq;

    fn grey_lut() -> TransferFunctionLut {
        let ctf = ColorTransferFunction::greyscale(0.0, 1.0);
        let otf = OpacityTransferFunction::linear_ramp(0.0, 1.0);
        TransferFunctionLut::bake(&ctf, &otf, 0.0, 1.0, 256)
    }

    #[test]
    fn lut_size_matches() {
        let lut = grey_lut();
        assert_eq!(lut.as_rgba_f32().len(), 256 * 4);
        assert_eq!(lut.lut_size(), 256);
    }

    #[test]
    fn first_entry_is_black_transparent() {
        let lut = grey_lut();
        let d = lut.as_rgba_f32();
        assert_abs_diff_eq!(d[0] as f64, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(d[3] as f64, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn last_entry_is_white_opaque() {
        let lut = grey_lut();
        let d = lut.as_rgba_f32();
        let last = (lut.lut_size() as usize - 1) * 4;
        assert_abs_diff_eq!(d[last] as f64, 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(d[last + 3] as f64, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn midpoint_is_mid_grey_half_opaque() {
        let lut = grey_lut();
        let d = lut.as_rgba_f32();
        let mid = 128 * 4;
        // t=128/255 ≈ 0.502 → close to 0.5
        assert!((d[mid] - 0.5).abs() < 0.01);
        assert!((d[mid + 3] - 0.5).abs() < 0.01);
    }

    #[test]
    fn opacity_is_monotone() {
        let lut = grey_lut();
        let d = lut.as_rgba_f32();
        let mut prev = 0.0f32;
        for i in 0..lut.lut_size() as usize {
            let a = d[i * 4 + 3];
            assert!(a >= prev - 1e-6, "LUT opacity not monotone at {i}");
            prev = a;
        }
    }
}
