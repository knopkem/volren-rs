//! Core data types, math, camera, transfer functions, and interaction for volren-rs.
//!
//! This crate has **no GPU dependency** and can be used in headless pipelines.
//! See `volren-gpu` for the wgpu-based renderer that consumes these types.

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod camera;
pub mod interaction;
pub mod math;
pub mod picking;
pub mod render_params;
pub mod reslice;
pub mod transfer_function;
pub mod volume;
pub mod window_level;

// Top-level re-exports for ergonomic use.
pub use camera::{Camera, Projection};
pub use interaction::{
    InteractionContext, InteractionResult, InteractionStyle, Key, KeyEvent, Modifiers,
    MouseEvent, MouseEventKind,
};
pub use math::aabb::Aabb;
pub use picking::{PickResult, Ray};
pub use render_params::{BlendMode, ClipPlane, Interpolation, ShadingParams, VolumeRenderParams};
pub use reslice::{SlicePlane, ThickSlabMode, ThickSlabParams};
pub use transfer_function::{
    ColorSpace, ColorTransferFunction, OpacityTransferFunction, TransferFunctionLut,
};
pub use volume::{DynVolume, Scalar, Volume, VolumeError, VolumeInfo};
pub use window_level::WindowLevel;
