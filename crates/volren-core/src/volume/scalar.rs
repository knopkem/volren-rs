//! The sealed [`Scalar`] trait and its implementations.

use bytemuck::Pod;

// ── Sealing mechanism ─────────────────────────────────────────────────────────

mod private {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for i8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

// ── Scalar trait ──────────────────────────────────────────────────────────────

/// A sealed numeric scalar type that can be stored in a [`super::Volume`].
///
/// This trait is sealed — downstream code cannot implement it for new types.
/// Supported types: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f32`, `f64`.
pub trait Scalar: private::Sealed + Copy + Send + Sync + Pod + 'static {
    /// Human-readable type name (e.g. `"u16"`, `"f32"`).
    const TYPE_NAME: &'static str;
    /// The minimum representable value, returned for out-of-bounds interpolation.
    fn min_value() -> Self;
    /// The maximum representable value.
    fn max_value() -> Self;
    /// Convert to `f64` for arithmetic.
    fn to_f64(self) -> f64;
    /// Convert from `f64`, clamping to the representable range.
    fn from_f64_clamped(v: f64) -> Self;
}

macro_rules! impl_scalar_int {
    ($T:ty, $name:expr) => {
        impl Scalar for $T {
            const TYPE_NAME: &'static str = $name;
            #[inline]
            fn min_value() -> Self {
                <$T>::MIN
            }
            #[inline]
            fn max_value() -> Self {
                <$T>::MAX
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn from_f64_clamped(v: f64) -> Self {
                v.clamp(<$T>::MIN as f64, <$T>::MAX as f64).round() as $T
            }
        }
    };
}

macro_rules! impl_scalar_float {
    ($T:ty, $name:expr) => {
        impl Scalar for $T {
            const TYPE_NAME: &'static str = $name;
            #[inline]
            fn min_value() -> Self {
                <$T>::MIN
            }
            #[inline]
            fn max_value() -> Self {
                <$T>::MAX
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn from_f64_clamped(v: f64) -> Self {
                v as $T
            }
        }
    };
}

impl_scalar_int!(u8, "u8");
impl_scalar_int!(i8, "i8");
impl_scalar_int!(u16, "u16");
impl_scalar_int!(i16, "i16");
impl_scalar_int!(u32, "u32");
impl_scalar_int!(i32, "i32");
impl_scalar_float!(f32, "f32");
impl_scalar_float!(f64, "f64");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_round_trip() {
        assert_eq!(u8::from_f64_clamped(127.6), 128u8);
        assert_eq!(u8::from_f64_clamped(-1.0), 0u8);
        assert_eq!(u8::from_f64_clamped(300.0), 255u8);
    }

    #[test]
    fn i16_round_trip() {
        assert_eq!(i16::from_f64_clamped(-1000.0), -1000i16);
        assert_eq!(i16::from_f64_clamped(i16::MIN as f64 - 1.0), i16::MIN);
        assert_eq!(i16::from_f64_clamped(i16::MAX as f64 + 1.0), i16::MAX);
    }

    #[test]
    fn f32_to_f64() {
        assert_eq!((1.5f32).to_f64(), 1.5f32 as f64);
    }
}
