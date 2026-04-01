//! GPU-accelerated volume renderer using [wgpu](https://wgpu.rs).
//!
//! This crate depends on [`volren_core`] for all data types and provides the
//! wgpu rendering pipeline. It is deliberately separated so that headless
//! pipelines (testing, server-side processing) can depend only on `volren-core`.

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]

pub mod renderer;
pub(crate) mod texture;
pub(crate) mod uniforms;

pub use renderer::{
    CrosshairParams, OrientationLabels, RenderError, Viewport, VolumeRenderer,
};
