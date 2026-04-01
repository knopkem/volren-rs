//! Transfer functions: colour and opacity mappings, LUT baking.
//!
//! # VTK Equivalent
//! `vtkColorTransferFunction`, `vtkPiecewiseFunction`,
//! baked via `vtkVolumeProperty::GetRGBTable / GetScalarOpacityTable`.

mod color;
mod lut;
mod opacity;
mod two_d;

pub use color::{ColorSpace, ColorTransferFunction};
pub use lut::TransferFunctionLut;
pub use opacity::OpacityTransferFunction;
pub use two_d::{TransferFunction2D, TransferFunction2DRegion};
