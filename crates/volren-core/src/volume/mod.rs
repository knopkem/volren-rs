//! Volume module: scalar trait, typed volume, type-erased volume.

mod dyn_volume;
mod scalar;

pub use dyn_volume::DynVolume;
pub use scalar::Scalar;

use glam::{DMat3, DVec3, UVec3};
use thiserror::Error;

use crate::math::Aabb;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced during volume construction or access.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum VolumeError {
    /// The supplied data length does not match the declared dimensions.
    #[error(
        "data length {actual} does not match dimensions \
         {dims_x}×{dims_y}×{dims_z}×{components} = {expected}"
    )]
    DimensionMismatch {
        /// Expected number of scalars.
        expected: usize,
        /// Actual number of scalars.
        actual: usize,
        /// Volume dimensions used for the expectation.
        dims_x: u32,
        /// Volume Y dimension.
        dims_y: u32,
        /// Volume Z dimension.
        dims_z: u32,
        /// Number of components per voxel.
        components: u32,
    },

    /// At least one dimension was zero.
    #[error("dimensions must be non-zero, got ({x}, {y}, {z})")]
    ZeroDimension {
        /// X dimension.
        x: u32,
        /// Y dimension.
        y: u32,
        /// Z dimension.
        z: u32,
    },

    /// At least one spacing value was zero or negative.
    #[error("spacing must be positive, got ({x}, {y}, {z})")]
    InvalidSpacing {
        /// X spacing.
        x: f64,
        /// Y spacing.
        y: f64,
        /// Z spacing.
        z: f64,
    },

    /// The number of components per voxel was zero.
    #[error("components must be >= 1, got 0")]
    ZeroComponents,

    /// Slices have inconsistent lengths.
    #[error("slice {index} has length {actual}, expected {expected} (width × height)")]
    InconsistentSlice {
        /// Slice index that was wrong.
        index: usize,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// No slices were provided.
    #[error("at least one slice must be provided")]
    EmptySlices,
}

// ── VolumeInfo trait ──────────────────────────────────────────────────────────

/// Read-only metadata about a volume, without exposing the scalar type.
///
/// Implemented by both [`Volume<T>`] and [`DynVolume`]. Allows geometry and
/// coordinate operations without monomorphisation on the scalar type.
pub trait VolumeInfo {
    /// Number of voxels along each axis: `[nx, ny, nz]`.
    fn dimensions(&self) -> UVec3;

    /// Physical size of each voxel in world units (typically millimetres).
    fn spacing(&self) -> DVec3;

    /// World-space position of voxel `(0, 0, 0)`.
    fn origin(&self) -> DVec3;

    /// 3×3 orientation matrix whose columns are the axis directions.
    /// For axis-aligned volumes this is the identity matrix.
    fn direction(&self) -> DMat3;

    /// Number of scalar values stored per voxel (1 for grayscale).
    fn components(&self) -> u32;

    /// Convert a continuous voxel index `(i, j, k)` to world coordinates.
    ///
    /// Formula: `world = origin + direction * (ijk * spacing)`
    fn index_to_world(&self, ijk: DVec3) -> DVec3 {
        self.origin() + self.direction() * (ijk * self.spacing())
    }

    /// Convert world coordinates to a continuous voxel index.
    ///
    /// This is the inverse of [`VolumeInfo::index_to_world`].
    fn world_to_index(&self, xyz: DVec3) -> DVec3 {
        let rel = xyz - self.origin();
        // direction is orthonormal, so inverse = transpose
        let inv = self.direction().transpose();
        inv * rel / self.spacing()
    }

    /// Axis-aligned bounding box enclosing all voxel centres in world space.
    fn world_bounds(&self) -> Aabb {
        let dims = self.dimensions().as_dvec3();
        // corners of the index grid (0 to dims-1)
        let corners = [
            DVec3::ZERO,
            DVec3::new(dims.x - 1.0, 0.0, 0.0),
            DVec3::new(0.0, dims.y - 1.0, 0.0),
            DVec3::new(0.0, 0.0, dims.z - 1.0),
            DVec3::new(dims.x - 1.0, dims.y - 1.0, 0.0),
            DVec3::new(dims.x - 1.0, 0.0, dims.z - 1.0),
            DVec3::new(0.0, dims.y - 1.0, dims.z - 1.0),
            dims - DVec3::ONE,
        ];
        let world_corners: Vec<DVec3> =
            corners.iter().map(|&c| self.index_to_world(c)).collect();

        let min = world_corners.iter().fold(DVec3::splat(f64::INFINITY), |a, &b| a.min(b));
        let max = world_corners.iter().fold(DVec3::splat(f64::NEG_INFINITY), |a, &b| a.max(b));
        Aabb::new(min, max)
    }
}

// ── Volume<T> ─────────────────────────────────────────────────────────────────

/// A regular 3D grid of scalar values stored contiguously in memory.
///
/// The memory layout is **X-fastest** (row-major in image terms):
/// `data[x + y * nx + z * nx * ny]` for single-component data.
///
/// # VTK Equivalent
/// `vtkImageData` with scalar data in `PointData`.
///
/// # Type Parameter
/// `T` must implement [`Scalar`], which is sealed to known numeric types.
#[derive(Debug, Clone)]
pub struct Volume<T: Scalar> {
    data: Vec<T>,
    dimensions: UVec3,
    spacing: DVec3,
    origin: DVec3,
    direction: DMat3,
    components: u32,
    scalar_range_cache: std::cell::OnceCell<(f64, f64)>,
}

impl<T: Scalar> Volume<T> {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a volume from a flat voxel buffer.
    ///
    /// `data.len()` must equal `dimensions.x * dimensions.y * dimensions.z * components`.
    ///
    /// # Errors
    /// Returns [`VolumeError`] if dimensions, spacing, or data length are invalid.
    pub fn from_data(
        data: Vec<T>,
        dimensions: UVec3,
        spacing: DVec3,
        origin: DVec3,
        direction: DMat3,
        components: u32,
    ) -> Result<Self, VolumeError> {
        Self::validate(dimensions, spacing, components)?;
        let expected = (dimensions.x as usize)
            * (dimensions.y as usize)
            * (dimensions.z as usize)
            * (components as usize);
        if data.len() != expected {
            return Err(VolumeError::DimensionMismatch {
                expected,
                actual: data.len(),
                dims_x: dimensions.x,
                dims_y: dimensions.y,
                dims_z: dimensions.z,
                components,
            });
        }
        Ok(Self {
            data,
            dimensions,
            spacing,
            origin,
            direction,
            components,
            scalar_range_cache: std::cell::OnceCell::new(),
        })
    }

    /// Assemble a volume from a sequence of 2D frames stacked along Z.
    ///
    /// Each slice must have exactly `width * height` scalars in row-major order
    /// (X-fastest). Useful when building a volume from DICOM slices.
    ///
    /// # Errors
    /// Returns [`VolumeError`] if any slice has the wrong length or inputs are invalid.
    pub fn from_slices(
        slices: &[&[T]],
        width: u32,
        height: u32,
        spacing: DVec3,
        origin: DVec3,
        direction: DMat3,
    ) -> Result<Self, VolumeError> {
        if slices.is_empty() {
            return Err(VolumeError::EmptySlices);
        }
        let expected_len = (width as usize) * (height as usize);
        for (i, slice) in slices.iter().enumerate() {
            if slice.len() != expected_len {
                return Err(VolumeError::InconsistentSlice {
                    index: i,
                    expected: expected_len,
                    actual: slice.len(),
                });
            }
        }
        let depth = slices.len() as u32;
        let mut data = Vec::with_capacity(expected_len * slices.len());
        for slice in slices {
            data.extend_from_slice(slice);
        }
        Self::from_data(
            data,
            UVec3::new(width, height, depth),
            spacing,
            origin,
            direction,
            1,
        )
    }

    // ── Voxel access ──────────────────────────────────────────────────────────

    /// Direct voxel access by integer index (component 0).
    ///
    /// Returns `None` if any index is out of bounds.
    #[must_use]
    pub fn get(&self, x: u32, y: u32, z: u32) -> Option<T> {
        if x >= self.dimensions.x || y >= self.dimensions.y || z >= self.dimensions.z {
            return None;
        }
        let idx = x as usize
            + y as usize * self.dimensions.x as usize
            + z as usize * (self.dimensions.x as usize * self.dimensions.y as usize);
        self.data.get(idx).copied()
    }

    /// Sample with trilinear interpolation at a continuous voxel index.
    ///
    /// Returns `None` if the index falls outside `[0, dims-1]` on any axis.
    /// Only samples the first component for multi-component volumes.
    #[must_use]
    pub fn sample_linear(&self, ijk: DVec3) -> Option<f64> {
        let dims = self.dimensions.as_dvec3() - DVec3::ONE;
        if ijk.x < 0.0 || ijk.y < 0.0 || ijk.z < 0.0 {
            return None;
        }
        if ijk.x > dims.x || ijk.y > dims.y || ijk.z > dims.z {
            return None;
        }

        let i0 = ijk.x.floor() as u32;
        let j0 = ijk.y.floor() as u32;
        let k0 = ijk.z.floor() as u32;
        let i1 = (i0 + 1).min(self.dimensions.x - 1);
        let j1 = (j0 + 1).min(self.dimensions.y - 1);
        let k1 = (k0 + 1).min(self.dimensions.z - 1);

        let tx = ijk.x - i0 as f64;
        let ty = ijk.y - j0 as f64;
        let tz = ijk.z - k0 as f64;

        macro_rules! g {
            ($x:expr, $y:expr, $z:expr) => {
                self.get($x, $y, $z).unwrap_or(T::min_value()).to_f64()
            };
        }

        let c000 = g!(i0, j0, k0);
        let c100 = g!(i1, j0, k0);
        let c010 = g!(i0, j1, k0);
        let c110 = g!(i1, j1, k0);
        let c001 = g!(i0, j0, k1);
        let c101 = g!(i1, j0, k1);
        let c011 = g!(i0, j1, k1);
        let c111 = g!(i1, j1, k1);

        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;
        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        Some(c0 * (1.0 - tz) + c1 * tz)
    }

    /// Sample nearest-neighbour at a continuous voxel index.
    ///
    /// Returns `None` if the index falls outside the volume.
    #[must_use]
    pub fn sample_nearest(&self, ijk: DVec3) -> Option<T> {
        let x = ijk.x.round() as i64;
        let y = ijk.y.round() as i64;
        let z = ijk.z.round() as i64;
        if x < 0 || y < 0 || z < 0 {
            return None;
        }
        self.get(x as u32, y as u32, z as u32)
    }

    /// Compute the (min, max) scalar range of all voxels.
    ///
    /// The result is computed once and cached.
    #[must_use]
    pub fn scalar_range(&self) -> (f64, f64) {
        *self.scalar_range_cache.get_or_init(|| {
            self.data
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    let f = v.to_f64();
                    (lo.min(f), hi.max(f))
                })
        })
    }

    /// Raw byte slice of the voxel data, suitable for GPU upload.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn validate(dimensions: UVec3, spacing: DVec3, components: u32) -> Result<(), VolumeError> {
        if dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0 {
            return Err(VolumeError::ZeroDimension {
                x: dimensions.x,
                y: dimensions.y,
                z: dimensions.z,
            });
        }
        if spacing.x <= 0.0 || spacing.y <= 0.0 || spacing.z <= 0.0 {
            return Err(VolumeError::InvalidSpacing {
                x: spacing.x,
                y: spacing.y,
                z: spacing.z,
            });
        }
        if components == 0 {
            return Err(VolumeError::ZeroComponents);
        }
        Ok(())
    }
}

impl<T: Scalar> VolumeInfo for Volume<T> {
    fn dimensions(&self) -> UVec3 {
        self.dimensions
    }
    fn spacing(&self) -> DVec3 {
        self.spacing
    }
    fn origin(&self) -> DVec3 {
        self.origin
    }
    fn direction(&self) -> DMat3 {
        self.direction
    }
    fn components(&self) -> u32 {
        self.components
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use glam::{DMat3, DVec3, UVec3};

    fn unit_volume() -> Volume<u8> {
        let data = (0u8..=3).flat_map(|z| (0u8..=3).flat_map(move |y| (0u8..=3).map(move |x| x + y * 4 + z * 16))).collect();
        Volume::from_data(
            data,
            UVec3::new(4, 4, 4),
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        )
        .unwrap()
    }

    #[test]
    fn from_data_happy_path() {
        let vol = unit_volume();
        assert_eq!(vol.dimensions(), UVec3::new(4, 4, 4));
    }

    #[test]
    fn from_data_wrong_length() {
        let err = Volume::<u8>::from_data(
            vec![0u8; 10],
            UVec3::new(4, 4, 4),
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        );
        assert!(matches!(err, Err(VolumeError::DimensionMismatch { .. })));
    }

    #[test]
    fn from_data_zero_dim() {
        let err = Volume::<u8>::from_data(
            vec![],
            UVec3::new(0, 4, 4),
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        );
        assert!(matches!(err, Err(VolumeError::ZeroDimension { .. })));
    }

    #[test]
    fn from_data_invalid_spacing() {
        let err = Volume::<u8>::from_data(
            vec![0u8; 64],
            UVec3::new(4, 4, 4),
            DVec3::new(0.0, 1.0, 1.0),
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        );
        assert!(matches!(err, Err(VolumeError::InvalidSpacing { .. })));
    }

    #[test]
    fn get_voxel() {
        let vol = unit_volume();
        assert_eq!(vol.get(0, 0, 0), Some(0));
        assert_eq!(vol.get(3, 3, 3), Some(3 + 3 * 4 + 3 * 16));
        assert_eq!(vol.get(4, 0, 0), None);
    }

    #[test]
    fn scalar_range() {
        let vol = unit_volume();
        let (lo, hi) = vol.scalar_range();
        assert_abs_diff_eq!(lo, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(hi, 3.0 + 3.0 * 4.0 + 3.0 * 16.0, epsilon = 1e-10);
    }

    #[test]
    fn sample_linear_at_integer_matches_get() {
        let vol = unit_volume();
        let v = vol.sample_linear(DVec3::new(1.0, 2.0, 3.0)).unwrap();
        assert_abs_diff_eq!(v, vol.get(1, 2, 3).unwrap().to_f64(), epsilon = 1e-10);
    }

    #[test]
    fn sample_linear_out_of_bounds_returns_none() {
        let vol = unit_volume();
        assert!(vol.sample_linear(DVec3::new(-0.1, 0.0, 0.0)).is_none());
        assert!(vol.sample_linear(DVec3::new(3.1, 0.0, 0.0)).is_none());
    }

    #[test]
    fn index_to_world_origin_aligned() {
        let vol = unit_volume();
        // Identity direction, unit spacing, zero origin: index = world
        let world = vol.index_to_world(DVec3::new(1.0, 2.0, 3.0));
        assert_abs_diff_eq!(world.x, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(world.y, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(world.z, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn index_world_round_trip() {
        let vol = Volume::<f32>::from_data(
            vec![0.0f32; 8 * 8 * 8],
            UVec3::new(8, 8, 8),
            DVec3::new(0.5, 0.75, 1.25),
            DVec3::new(10.0, 20.0, 30.0),
            DMat3::IDENTITY,
            1,
        )
        .unwrap();
        let ijk = DVec3::new(3.5, 2.0, 6.0);
        let world = vol.index_to_world(ijk);
        let back = vol.world_to_index(world);
        assert_abs_diff_eq!(back.x, ijk.x, epsilon = 1e-10);
        assert_abs_diff_eq!(back.y, ijk.y, epsilon = 1e-10);
        assert_abs_diff_eq!(back.z, ijk.z, epsilon = 1e-10);
    }

    #[test]
    fn from_slices_assembles_correctly() {
        let slice0: Vec<u8> = (0..4).collect();
        let slice1: Vec<u8> = (4..8).collect();
        let vol = Volume::from_slices(
            &[slice0.as_slice(), slice1.as_slice()],
            2,
            2,
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
        )
        .unwrap();
        assert_eq!(vol.dimensions(), UVec3::new(2, 2, 2));
        assert_eq!(vol.get(0, 0, 0), Some(0));
        assert_eq!(vol.get(0, 0, 1), Some(4));
    }

    #[test]
    fn from_slices_inconsistent_length() {
        let s0 = vec![0u8; 4];
        let s1 = vec![0u8; 3]; // wrong
        let err = Volume::<u8>::from_slices(
            &[s0.as_slice(), s1.as_slice()],
            2,
            2,
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
        );
        assert!(matches!(err, Err(VolumeError::InconsistentSlice { index: 1, .. })));
    }

    #[test]
    fn world_bounds_axis_aligned() {
        let vol = Volume::<u8>::from_data(
            vec![0u8; 4 * 4 * 4],
            UVec3::new(4, 4, 4),
            DVec3::new(2.0, 3.0, 4.0),
            DVec3::new(1.0, 1.0, 1.0),
            DMat3::IDENTITY,
            1,
        )
        .unwrap();
        let bounds = vol.world_bounds();
        // origin=1, spacing=2,3,4, dims=4 → max corner = 1 + 3*2/3/4
        assert_abs_diff_eq!(bounds.min.x, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bounds.max.x, 1.0 + 3.0 * 2.0, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use glam::{DMat3, DVec3};
    use proptest::prelude::*;

    fn make_vol() -> Volume<u8> {
        Volume::from_data(
            vec![0u8; 4 * 4 * 4],
            UVec3::new(4, 4, 4),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        )
        .unwrap()
    }

    proptest! {
        #[test]
        fn index_to_world_round_trip(
            ix in 0.0f64..3.0,
            iy in 0.0f64..3.0,
            iz in 0.0f64..3.0,
        ) {
            let vol = make_vol();
            let idx = DVec3::new(ix, iy, iz);
            let world = vol.index_to_world(idx);
            let back = vol.world_to_index(world);
            prop_assert!((back.x - idx.x).abs() < 1e-8, "x: {} vs {}", idx.x, back.x);
            prop_assert!((back.y - idx.y).abs() < 1e-8, "y: {} vs {}", idx.y, back.y);
            prop_assert!((back.z - idx.z).abs() < 1e-8, "z: {} vs {}", idx.z, back.z);
        }

        #[test]
        fn world_to_index_round_trip(
            wx in 0.0f64..3.0,
            wy in 0.0f64..3.0,
            wz in 0.0f64..3.0,
        ) {
            let vol = make_vol();
            let world = DVec3::new(wx, wy, wz);
            let idx = vol.world_to_index(world);
            let back = vol.index_to_world(idx);
            prop_assert!((back.x - world.x).abs() < 1e-8, "x: {} vs {}", world.x, back.x);
            prop_assert!((back.y - world.y).abs() < 1e-8, "y: {} vs {}", world.y, back.y);
            prop_assert!((back.z - world.z).abs() < 1e-8, "z: {} vs {}", world.z, back.z);
        }
    }
}
