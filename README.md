# volren-rs

A pure-Rust volume rendering library for medical imaging, inspired by VTK's domain knowledge.

Built around a two-crate architecture — a CPU-side core with no GPU dependency, and a wgpu-based renderer — so headless pipelines, tests, and server-side code never need a graphics context.

## Crates

| Crate | Purpose |
|---|---|
| **volren-core** | Volume data model, camera, transfer functions, window/level, reslice planes, interaction styles, picking — zero GPU deps |
| **volren-gpu** | wgpu render pipelines: raycasting, MPR slicing, overlays |

## Features

- **Volume data model** — generic `Volume<T>` over sealed `Scalar` types (u8, i8, u16, i16, f32, f64), type-erased `DynVolume`, trilinear sampling, index↔world coordinate transforms
- **Camera** — perspective & orthographic projection, orbit/pan/dolly/zoom, azimuth/elevation/roll
- **Transfer functions** — piecewise-linear colour & opacity TFs, region-based 2D TFs, RGB/HSV/Lab interpolation, baked LUTs for GPU upload
- **Window/level** — DICOM-standard mapping with clinical presets (CT Bone, Lung, Brain, Abdomen)
- **Reslice (MPR)** — `SlicePlane` with axial/coronal/sagittal constructors, `point_to_world` / `world_to_point`, thick-slab support
- **Interaction** — windowing-agnostic `MouseEvent`/`KeyEvent` types, `TrackballStyle` and `ImageSliceStyle` interactors
- **Picking** — CPU ray-AABB intersection, `pick_volume()` returning world position + voxel value
- **GPU raycasting** — wgpu pipeline with composite/MIP/MinIP/average/additive/isosurface blend modes, Blinn-Phong shading, gradient opacity, adaptive step sizing, clip planes, cropping, opacity correction
- **Viewer building blocks** — slice rendering, thick-slab MPR, crosshair overlays, orientation marker, off-screen render targets, reduced-resolution interactive rendering
- **Large-volume friendly upload** — chunked 3D texture uploads avoid single giant staging writes for bigger datasets
- **No I/O, no windowing** — designed as a library; bring your own data loader and event loop

## Quick start

```rust
use volren_core::{Camera, Volume, VolumeRenderParams, WindowLevel};
use glam::{UVec3, DVec3, DMat3};

// Create a volume from raw voxel data
let volume = Volume::<u16>::from_data(
    voxel_data,
    UVec3::new(512, 512, 128),
    DVec3::new(0.5, 0.5, 2.0),  // spacing in mm
    DVec3::ZERO,                  // origin
    DMat3::IDENTITY,              // direction cosines
    1,                            // samples per voxel
).unwrap();

let camera = Camera::new_perspective(
    DVec3::new(0.0, 0.0, 500.0),
    DVec3::ZERO,
    30.0,
);

let params = VolumeRenderParams::builder()
    .window_level(WindowLevel::ct_bone())
    .step_size_factor(0.5)
    .build();
```

For GPU rendering, create a `VolumeRenderer` with your wgpu device:

```rust
use volren_core::DynVolume;
use volren_gpu::{Viewport, VolumeRenderer};

let dyn_volume: DynVolume = volume.into();
let mut renderer = VolumeRenderer::new(&device, &queue, surface_format);
renderer.set_volume(&dyn_volume, true);
renderer.set_render_params(&params)?;

renderer.render_volume(
    &mut encoder,
    &target_view,
    &camera,
    &params,
    Viewport::full(1920, 1080),
)?;
```

## Design principles

- **No `unsafe`** — both crates use `#![deny(unsafe_code)]`
- **No allocations on the render path** — uniforms are stack-allocated, LUTs pre-baked
- **Separation of concerns** — core crate is pure math/data, GPU crate is pure rendering
- **Idiomatic Rust** — builder patterns, `thiserror` errors, sealed traits, `#[must_use]`, comprehensive docs
- **Testable** — 94 tests today, including `proptest` property-based checks for coordinate transforms

## Building

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace
```

Requires Rust 1.80+. The GPU crate needs a wgpu-compatible graphics driver for runtime use, but compiles without one.

## Integration with medical imaging

volren-rs is designed to be consumed by a medical viewer application. It pairs naturally with [dicom-toolkit-rs](https://github.com/your-org/dicom-toolkit-rs) for DICOM parsing — feed pixel data into `Volume::from_slices()` and render.

## License

MIT OR Apache-2.0
