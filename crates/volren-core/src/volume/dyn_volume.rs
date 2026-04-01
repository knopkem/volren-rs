//! Type-erased volume enum for storing any supported scalar type.

use super::{Volume, VolumeInfo};
use glam::{DMat3, DVec3, UVec3};

/// A volume whose scalar type is determined at runtime.
///
/// This enum lets a consumer store any supported scalar flavour behind one
/// handle without carrying `T` as a generic parameter.
///
/// # VTK Equivalent
/// `vtkImageData::GetScalarType()` — a runtime tag rather than a compile-time type.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DynVolume {
    /// Unsigned 8-bit integer voxels.
    U8(Volume<u8>),
    /// Signed 8-bit integer voxels.
    I8(Volume<i8>),
    /// Unsigned 16-bit integer voxels.
    U16(Volume<u16>),
    /// Signed 16-bit integer voxels (most common for CT Hounsfield units).
    I16(Volume<i16>),
    /// Unsigned 32-bit integer voxels.
    U32(Volume<u32>),
    /// Signed 32-bit integer voxels.
    I32(Volume<i32>),
    /// 32-bit floating-point voxels.
    F32(Volume<f32>),
    /// 64-bit floating-point voxels.
    F64(Volume<f64>),
}

impl DynVolume {
    /// Raw byte slice of the underlying voxel data, suitable for GPU upload.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::U8(v) => v.as_bytes(),
            Self::I8(v) => v.as_bytes(),
            Self::U16(v) => v.as_bytes(),
            Self::I16(v) => v.as_bytes(),
            Self::U32(v) => v.as_bytes(),
            Self::I32(v) => v.as_bytes(),
            Self::F32(v) => v.as_bytes(),
            Self::F64(v) => v.as_bytes(),
        }
    }

    /// The (min, max) scalar range, normalised to `f64`.
    #[must_use]
    pub fn scalar_range(&self) -> (f64, f64) {
        match self {
            Self::U8(v) => v.scalar_range(),
            Self::I8(v) => v.scalar_range(),
            Self::U16(v) => v.scalar_range(),
            Self::I16(v) => v.scalar_range(),
            Self::U32(v) => v.scalar_range(),
            Self::I32(v) => v.scalar_range(),
            Self::F32(v) => v.scalar_range(),
            Self::F64(v) => v.scalar_range(),
        }
    }

    /// Sample with trilinear interpolation at a continuous voxel index.
    #[must_use]
    pub fn sample_linear(&self, ijk: DVec3) -> Option<f64> {
        match self {
            Self::U8(v) => v.sample_linear(ijk),
            Self::I8(v) => v.sample_linear(ijk),
            Self::U16(v) => v.sample_linear(ijk),
            Self::I16(v) => v.sample_linear(ijk),
            Self::U32(v) => v.sample_linear(ijk),
            Self::I32(v) => v.sample_linear(ijk),
            Self::F32(v) => v.sample_linear(ijk),
            Self::F64(v) => v.sample_linear(ijk),
        }
    }

    /// Number of bytes per scalar component.
    #[must_use]
    pub fn bytes_per_component(&self) -> usize {
        match self {
            Self::U8(_) | Self::I8(_) => 1,
            Self::U16(_) | Self::I16(_) => 2,
            Self::U32(_) | Self::I32(_) | Self::F32(_) => 4,
            Self::F64(_) => 8,
        }
    }
}

impl VolumeInfo for DynVolume {
    fn dimensions(&self) -> UVec3 {
        match self {
            Self::U8(v) => v.dimensions(),
            Self::I8(v) => v.dimensions(),
            Self::U16(v) => v.dimensions(),
            Self::I16(v) => v.dimensions(),
            Self::U32(v) => v.dimensions(),
            Self::I32(v) => v.dimensions(),
            Self::F32(v) => v.dimensions(),
            Self::F64(v) => v.dimensions(),
        }
    }
    fn spacing(&self) -> DVec3 {
        match self {
            Self::U8(v) => v.spacing(),
            Self::I8(v) => v.spacing(),
            Self::U16(v) => v.spacing(),
            Self::I16(v) => v.spacing(),
            Self::U32(v) => v.spacing(),
            Self::I32(v) => v.spacing(),
            Self::F32(v) => v.spacing(),
            Self::F64(v) => v.spacing(),
        }
    }
    fn origin(&self) -> DVec3 {
        match self {
            Self::U8(v) => v.origin(),
            Self::I8(v) => v.origin(),
            Self::U16(v) => v.origin(),
            Self::I16(v) => v.origin(),
            Self::U32(v) => v.origin(),
            Self::I32(v) => v.origin(),
            Self::F32(v) => v.origin(),
            Self::F64(v) => v.origin(),
        }
    }
    fn direction(&self) -> DMat3 {
        match self {
            Self::U8(v) => v.direction(),
            Self::I8(v) => v.direction(),
            Self::U16(v) => v.direction(),
            Self::I16(v) => v.direction(),
            Self::U32(v) => v.direction(),
            Self::I32(v) => v.direction(),
            Self::F32(v) => v.direction(),
            Self::F64(v) => v.direction(),
        }
    }
    fn components(&self) -> u32 {
        match self {
            Self::U8(v) => v.components(),
            Self::I8(v) => v.components(),
            Self::U16(v) => v.components(),
            Self::I16(v) => v.components(),
            Self::U32(v) => v.components(),
            Self::I32(v) => v.components(),
            Self::F32(v) => v.components(),
            Self::F64(v) => v.components(),
        }
    }
}

/// Helper macro to convert a `Volume<T>` into the matching `DynVolume` variant.
macro_rules! impl_from_volume {
    ($T:ty, $Variant:ident) => {
        impl From<Volume<$T>> for DynVolume {
            fn from(v: Volume<$T>) -> Self {
                Self::$Variant(v)
            }
        }
    };
}

impl_from_volume!(u8, U8);
impl_from_volume!(i8, I8);
impl_from_volume!(u16, U16);
impl_from_volume!(i16, I16);
impl_from_volume!(u32, U32);
impl_from_volume!(i32, I32);
impl_from_volume!(f32, F32);
impl_from_volume!(f64, F64);

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat3, DVec3, UVec3};

    fn make_i16_volume() -> DynVolume {
        let data: Vec<i16> = (-4i16..4i16).collect();
        let vol = Volume::from_data(
            data,
            UVec3::new(2, 2, 2),
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        )
        .unwrap();
        DynVolume::I16(vol)
    }

    #[test]
    fn dyn_scalar_range() {
        let dv = make_i16_volume();
        let (lo, hi) = dv.scalar_range();
        approx::assert_abs_diff_eq!(lo, -4.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(hi, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn dyn_dimensions() {
        let dv = make_i16_volume();
        assert_eq!(dv.dimensions(), UVec3::new(2, 2, 2));
    }

    #[test]
    fn dyn_bytes_per_component() {
        assert_eq!(make_i16_volume().bytes_per_component(), 2);
    }

    #[test]
    fn from_volume_u8() {
        let v = Volume::<u8>::from_data(
            vec![0u8; 8],
            UVec3::new(2, 2, 2),
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        )
        .unwrap();
        let dv: DynVolume = v.into();
        assert!(matches!(dv, DynVolume::U8(_)));
    }
}
