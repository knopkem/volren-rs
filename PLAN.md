# volren-rs — Pure Rust Volume Rendering Library

## 1. Vision

A pure Rust **library** for GPU-accelerated volume rendering, inspired by VTK's
domain knowledge but designed from scratch using idiomatic Rust. The library provides
composable building blocks for applications that need to render volumetric data —
it does **not** own a window, event loop, or file I/O.

The primary downstream consumer is a medical imaging viewer built with
`dicom-toolkit-rs` (in the sibling directory), but `volren-rs` itself is
domain-agnostic and knows nothing about DICOM.

### Design Principles

1. **Library, not framework** — the consumer owns the window, event loop, and
   data loading. volren-rs takes data in, produces pixels out.
2. **Idiomatic Rust** — traits over inheritance, enums over type codes,
   `Result<T,E>` over exceptions, zero `unsafe` in public API, strong type safety.
3. **Testable** — every module has unit tests; GPU rendering is tested via
   snapshot/image-comparison tests; math and data model are tested with property-based
   tests where appropriate.
4. **`#![deny(missing_docs)]`** — every public item is documented.
5. **No I/O** — the library accepts `&[T]` slices and metadata structs. The
   consumer (e.g. `dicom-toolkit-image`) is responsible for decoding files.
6. **No windowing dependency** — interaction logic accepts abstract input events
   (`MouseEvent`, `KeyEvent`), not `winit` types. A thin adapter layer in the
   consumer maps platform events to volren events.

### Explicit Non-Goals

- File I/O (DICOM, NIfTI, raw — handled by consumer)
- Windowing / event loop (handled by consumer via winit, egui, etc.)
- Surface mesh rendering (polygon actors, poly-data mappers)
- General-purpose visualization (charts, graphs, info-vis)
- Filter/algorithm framework (VTK's 1,770 filter classes)
- Python/Java bindings
- CPU-based volume raycasting (GPU-only via wgpu)
- Unstructured grid volume rendering
- VR/XR support

---

## 2. Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                   volren-rs (workspace)                    │
├─────────────────────────┬─────────────────────────────────┤
│      volren-core        │          volren-gpu              │
│                         │                                  │
│  Volume<T>, DynVolume   │   VolumeRenderer                │
│  Camera                 │   ResliceRenderer                │
│  TransferFunction       │   WGSL shaders                   │
│  WindowLevel            │   3D texture management          │
│  InteractionStyle trait │   Uniform buffers                │
│  TrackballStyle         │   Render pipelines               │
│  ImageSliceStyle        │   OrientationMarker (GPU)        │
│  SlicePlane, Reslice    │   CrosshairOverlay (GPU)         │
│  VolumePicker (CPU ray) │   AnnotationOverlay (GPU)        │
│  BlendMode, Shading     │                                  │
│  Abstract input events  │   depends on: volren-core, wgpu  │
│                         │                                  │
│  depends on: glam,      │                                  │
│  bytemuck               │                                  │
└─────────────────────────┴─────────────────────────────────┘
           ▲                            ▲
           │                            │
    ┌──────┴────────────────────────────┴──────┐
    │        Consumer application               │
    │  (e.g. medical viewer using               │
    │   dicom-toolkit-rs + winit + volren-rs)   │
    └───────────────────────────────────────────┘
```

### Crate Separation Rationale

| Crate | Purpose | Dependencies |
|-------|---------|--------------|
| **volren-core** | Pure data types + math + interaction logic. **No GPU.** Can be used headless for testing, coordinate math, data preparation. | `glam`, `bytemuck`, `thiserror` |
| **volren-gpu** | wgpu-based rendering. Takes core types as input, produces rendered frames. | `wgpu`, `bytemuck`, `volren-core` |

A consumer that only needs coordinate math / volume data types (e.g. a headless
processing pipeline) can depend on `volren-core` alone without pulling in wgpu.

---

## 3. Data Model (`volren-core::volume`)

### 3.1 Volume Representation

Inspired by VTK's `vtkImageData`. Uses Rust generics for compile-time type safety
instead of VTK's runtime `GetDataType()` dispatch.

```rust
/// A regular 3D grid of scalar values, stored contiguously in memory.
///
/// The memory layout is X-fastest (column-major in VTK terms):
/// `data[x + y * dim.x + z * dim.x * dim.y]` for single-component data.
///
/// # VTK Equivalent
/// `vtkImageData` with a single scalar component in `PointData`.
///
/// # Type Parameter
/// `T` is the voxel scalar type (u8, i16, u16, f32, etc.).
#[derive(Debug, Clone)]
pub struct Volume<T: Scalar> {
    data: Vec<T>,
    dimensions: UVec3,          // [nx, ny, nz]
    spacing: DVec3,             // Physical voxel size (mm)
    origin: DVec3,              // World position of voxel (0,0,0) center
    direction: DMat3,           // Orientation matrix (columns = axis directions)
    components: u32,            // Scalars per voxel (1 for grayscale)
}
```

**`Scalar` trait** — sealed, implemented for the 8 types medical imaging uses:
```rust
/// Trait for voxel scalar types. Sealed — only implemented for known numeric types.
///
/// Replaces VTK's `VTK_UNSIGNED_SHORT` / `VTK_FLOAT` runtime type codes with
/// compile-time generics.
pub trait Scalar:
    Copy + Clone + Send + Sync + PartialOrd + bytemuck::Pod + 'static
{
    /// Human-readable type name (e.g. "u16", "f32").
    const TYPE_NAME: &'static str;

    /// Convert to f64 for transfer-function evaluation.
    fn to_f64(self) -> f64;

    /// Minimum representable value.
    fn min_value() -> Self;

    /// Maximum representable value.
    fn max_value() -> Self;
}
// Sealed impl for: u8, i8, u16, i16, u32, i32, f32, f64
```

**`DynVolume`** — for runtime type dispatch when the scalar type is not known at
compile time (e.g. the consumer decodes a DICOM file and doesn't know if it's
u8 or i16 until runtime):
```rust
/// Type-erased volume. Use when the scalar type is determined at runtime.
///
/// Each variant wraps a strongly-typed `Volume<T>`.
/// Use `match` or the `with_volume!` macro to dispatch back to generic code.
#[derive(Debug, Clone)]
pub enum DynVolume {
    U8(Volume<u8>),
    I16(Volume<i16>),
    U16(Volume<u16>),
    I32(Volume<i32>),
    F32(Volume<f32>),
    F64(Volume<f64>),
}
```

### 3.2 Volume Construction (from consumer data)

The consumer (e.g. code using `dicom-toolkit-image`) is responsible for
decoding pixel data. volren-rs accepts raw slices:

```rust
impl<T: Scalar> Volume<T> {
    /// Create a volume from pre-allocated voxel data.
    ///
    /// # Errors
    /// Returns `VolumeError::DimensionMismatch` if
    /// `data.len() != dim.x * dim.y * dim.z * components`.
    pub fn from_data(
        data: Vec<T>,
        dimensions: UVec3,
        spacing: DVec3,
        origin: DVec3,
        direction: DMat3,
        components: u32,
    ) -> Result<Self, VolumeError>;

    /// Create from a slice of 2D frames stacked along Z.
    /// Each frame is `width × height` scalars, in row-major order.
    ///
    /// Useful when assembling a volume from DICOM slices.
    pub fn from_slices(
        slices: &[&[T]],
        width: u32,
        height: u32,
        spacing: DVec3,
        origin: DVec3,
        direction: DMat3,
    ) -> Result<Self, VolumeError>;
}
```

### 3.3 Coordinate Transforms

```rust
impl<T: Scalar> Volume<T> {
    /// Convert voxel index (i,j,k) to world coordinates (x,y,z).
    /// Applies: world = origin + direction * (index * spacing)
    pub fn index_to_world(&self, ijk: DVec3) -> DVec3;

    /// Convert world coordinates to continuous voxel index.
    /// Inverse of `index_to_world`.
    pub fn world_to_index(&self, xyz: DVec3) -> DVec3;

    /// Axis-aligned bounding box in world coordinates.
    pub fn world_bounds(&self) -> Aabb;

    /// Sample with trilinear interpolation at continuous voxel index.
    /// Returns `None` if the index is outside the volume.
    pub fn sample_linear(&self, ijk: DVec3) -> Option<f64>;

    /// Sample nearest-neighbor at continuous voxel index.
    pub fn sample_nearest(&self, ijk: DVec3) -> Option<T>;

    /// Direct voxel access by integer index.
    pub fn get(&self, x: u32, y: u32, z: u32) -> Option<T>;

    /// Scalar range (min, max) of the data. Computed lazily and cached.
    pub fn scalar_range(&self) -> (f64, f64);

    /// Raw data slice for GPU upload.
    pub fn as_bytes(&self) -> &[u8];
}
```

### 3.4 Transfer Functions (`volren-core::transfer_function`)

```rust
/// Piecewise-linear opacity mapping: scalar → opacity [0,1].
/// VTK equivalent: vtkPiecewiseFunction
#[derive(Debug, Clone)]
pub struct OpacityTransferFunction {
    nodes: Vec<OpacityNode>,  // Sorted by `x`
}

/// Piecewise-linear color mapping: scalar → RGB.
/// VTK equivalent: vtkColorTransferFunction
#[derive(Debug, Clone)]
pub struct ColorTransferFunction {
    nodes: Vec<ColorNode>,    // Sorted by `x`
    color_space: ColorSpace,  // Interpolation space: Rgb | Hsv | Lab
}

/// Pre-baked RGBA lookup table for GPU upload.
/// Resolution is configurable (default: 4096 entries).
pub struct TransferFunctionLut {
    rgba: Vec<[f32; 4]>,
    scalar_range: [f64; 2],
    resolution: u32,
}

impl TransferFunctionLut {
    /// Bake color + opacity TFs into a single RGBA LUT.
    pub fn bake(
        color_tf: &ColorTransferFunction,
        opacity_tf: &OpacityTransferFunction,
        scalar_range: [f64; 2],
        resolution: u32,
    ) -> Self;
}

/// 2D transfer function: (scalar, gradient_magnitude) → RGBA.
pub struct TransferFunction2D { ... }
```

### 3.5 Window/Level (`volren-core::window_level`)

Compatible with `dicom-toolkit-image::WindowLevel` — same semantics, different
type to avoid coupling. The consumer can trivially convert between them.

```rust
/// Contrast/brightness control.
/// Follows DICOM PS 3.3 §C.7.6.3.1.5 windowing formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowLevel {
    pub center: f64,
    pub width: f64,
}

impl WindowLevel {
    pub fn new(center: f64, width: f64) -> Self;

    /// Normalize a scalar value to [0.0, 1.0].
    pub fn normalize(&self, value: f64) -> f64;

    /// Derive from a volume's scalar range.
    pub fn from_scalar_range(min: f64, max: f64) -> Self;

    /// Common CT presets.
    pub fn ct_bone() -> Self;
    pub fn ct_lung() -> Self;
    pub fn ct_brain() -> Self;
    pub fn ct_abdomen() -> Self;
}
```

### 3.6 Camera (`volren-core::camera`)

```rust
/// Configurable camera with perspective and orthographic modes.
/// VTK equivalent: vtkCamera
#[derive(Debug, Clone)]
pub struct Camera {
    position: DVec3,
    focal_point: DVec3,
    view_up: DVec3,
    projection: Projection,
    clipping_range: (f64, f64),
}

#[derive(Debug, Clone, Copy)]
pub enum Projection {
    Perspective { fov_y_degrees: f64 },
    Orthographic { parallel_scale: f64 },
}

impl Camera {
    pub fn new_perspective(position: DVec3, focal_point: DVec3, fov_y: f64) -> Self;
    pub fn new_orthographic(position: DVec3, focal_point: DVec3, scale: f64) -> Self;

    // Matrices
    pub fn view_matrix(&self) -> DMat4;
    pub fn projection_matrix(&self, aspect_ratio: f64) -> DMat4;

    // VTK-style manipulation
    pub fn azimuth(&mut self, degrees: f64);
    pub fn elevation(&mut self, degrees: f64);
    pub fn roll(&mut self, degrees: f64);
    pub fn dolly(&mut self, factor: f64);
    pub fn zoom(&mut self, factor: f64);
    pub fn pan(&mut self, dx: f64, dy: f64);
    pub fn reset_to_bounds(&mut self, bounds: &Aabb, margin: f64);

    // Queries
    pub fn direction(&self) -> DVec3;
    pub fn distance(&self) -> f64;
    pub fn right_vector(&self) -> DVec3;
}
```

### 3.7 Volume Rendering Parameters (`volren-core::render_params`)

```rust
/// All parameters needed to render a volume.
/// Passed to the GPU renderer each frame.
#[derive(Debug, Clone)]
pub struct VolumeRenderParams {
    pub color_tf: ColorTransferFunction,
    pub opacity_tf: OpacityTransferFunction,
    pub gradient_opacity_tf: Option<OpacityTransferFunction>,
    pub blend_mode: BlendMode,
    pub interpolation: Interpolation,
    pub shading: ShadingParams,
    pub window_level: Option<WindowLevel>,
    pub clip_planes: Vec<ClipPlane>,  // Up to 6
    pub cropping_bounds: Option<Aabb>,
    pub step_size_factor: f32,        // 1.0 = one sample per voxel
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    Composite,
    MaximumIntensity,
    MinimumIntensity,
    AverageIntensity,
    Additive,
    Isosurface { iso_value: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct ShadingParams {
    pub enabled: bool,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub specular_power: f32,
}

impl Default for ShadingParams {
    fn default() -> Self {
        Self { enabled: true, ambient: 0.1, diffuse: 0.7, specular: 0.2, specular_power: 10.0 }
    }
}
```

### 3.8 Slice Plane / Reslice (`volren-core::reslice`)

Building block for MPR — the library defines the geometry, the consumer
implements the layout and crosshair linking.

```rust
/// Defines an arbitrary 2D slice through a 3D volume.
/// Used for MPR (multiplanar reformatting).
///
/// VTK equivalent: The ResliceAxes matrix passed to vtkImageReslice.
#[derive(Debug, Clone)]
pub struct SlicePlane {
    pub origin: DVec3,     // Center of the slice (world coordinates)
    pub right: DVec3,      // Horizontal axis direction (unit vector)
    pub up: DVec3,         // Vertical axis direction (unit vector)
    pub width: f64,        // Physical extent along `right` (mm)
    pub height: f64,       // Physical extent along `up` (mm)
}

impl SlicePlane {
    /// Normal vector (right × up), normalized.
    pub fn normal(&self) -> DVec3;

    /// Create the three standard orthogonal planes from a volume.
    pub fn axial(volume: &impl VolumeInfo, center: DVec3) -> Self;
    pub fn sagittal(volume: &impl VolumeInfo, center: DVec3) -> Self;
    pub fn coronal(volume: &impl VolumeInfo, center: DVec3) -> Self;

    /// Translate the plane along its normal by `distance` mm.
    pub fn translate_along_normal(&mut self, distance: f64);

    /// Rotate the plane around an axis passing through `origin`.
    pub fn rotate(&mut self, axis: DVec3, angle_degrees: f64);

    /// Convert a point on the slice (in 2D normalized [0,1]² coords)
    /// to world 3D coordinates.
    pub fn point_to_world(&self, uv: DVec2) -> DVec3;

    /// Project a world point onto the slice, returns (u, v, signed_distance).
    pub fn world_to_point(&self, world: DVec3) -> (DVec2, f64);
}

/// Parameters for thick-slab reslicing.
#[derive(Debug, Clone, Copy)]
pub struct ThickSlabParams {
    pub thickness: f64,       // Slab thickness in mm
    pub mode: ThickSlabMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThickSlabMode {
    Mean,   // Average intensity projection
    Max,    // MIP through slab
    Min,    // MinIP through slab
}
```

### 3.9 Abstract Input Events & Interaction (`volren-core::interaction`)

The library defines **abstract input events** and **interaction style traits**.
The consumer maps platform-specific events (winit, egui, etc.) to these.

```rust
// ── Abstract events (no winit dependency) ─────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct MouseEvent {
    pub position: Vec2,       // Pixel coordinates in viewport
    pub kind: MouseEventKind,
}

#[derive(Debug, Clone, Copy)]
pub enum MouseEventKind {
    Press(MouseButton),
    Release(MouseButton),
    Move,
    Scroll(f32),              // Positive = scroll up/zoom in
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton { Left, Middle, Right }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KeyEvent {
    pub key: Key,
    pub pressed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    R, X, Y, Z,
    Up, Down, Left, Right,
    Plus, Minus,
    Escape,
    Char(char),
}

// ── Interaction result ────────────────────────────────────────────────────

/// What changed as a result of processing an input event.
/// Tells the consumer whether to re-render.
#[derive(Debug, Clone, Default)]
pub struct InteractionResult {
    pub camera_changed: bool,
    pub window_level_changed: bool,
    pub slice_changed: bool,
    pub needs_redraw: bool,
}

// ── Interaction style trait ───────────────────────────────────────────────

/// Strategy for mapping input events to camera/viewport changes.
/// VTK equivalent: vtkInteractorStyle
pub trait InteractionStyle: Send {
    fn handle_mouse(
        &mut self,
        event: &MouseEvent,
        modifiers: &Modifiers,
        ctx: &mut InteractionContext,
    ) -> InteractionResult;

    fn handle_key(
        &mut self,
        event: &KeyEvent,
        modifiers: &Modifiers,
        ctx: &mut InteractionContext,
    ) -> InteractionResult;
}

/// Mutable context passed to interaction styles.
pub struct InteractionContext<'a> {
    pub camera: &'a mut Camera,
    pub window_level: &'a mut WindowLevel,
    pub viewport_size: UVec2,
    pub volume_bounds: &'a Aabb,
    pub slice_plane: Option<&'a mut SlicePlane>,
}
```

**Concrete styles (shipped with the library):**

```rust
/// 3D trackball camera rotation.
/// VTK equivalent: vtkInteractorStyleTrackballCamera
///
/// Left drag:    Rotate camera around focal point (arcball)
/// Right drag:   Zoom (dolly)
/// Middle drag:  Pan
/// Scroll:       Zoom
pub struct TrackballStyle { ... }

/// 2D image/slice viewing with window/level.
/// VTK equivalent: vtkInteractorStyleImage
///
/// Left drag:          Window/level (horiz=window, vert=level)
/// Right drag:         Zoom
/// Middle drag:        Pan
/// Scroll:             Scroll through slices
/// Ctrl+Left drag:     Pan (alternative)
/// R:                  Reset window/level to auto-range
pub struct ImageSliceStyle { ... }
```

### 3.10 Picking (`volren-core::picking`)

CPU-based ray-volume intersection.

```rust
/// Result of picking a point on a volume.
#[derive(Debug, Clone)]
pub struct PickResult {
    pub world_position: DVec3,
    pub voxel_index: DVec3,       // Continuous index
    pub voxel_value: f64,
}

/// Cast a ray from screen coordinates through the volume.
/// Returns the first non-transparent hit.
pub fn pick_volume(
    screen_pos: Vec2,
    camera: &Camera,
    viewport_size: UVec2,
    volume: &DynVolume,
    params: &VolumeRenderParams,
) -> Option<PickResult>;
```

---

## 4. GPU Renderer (`volren-gpu`)

### 4.1 Renderer API

```rust
/// The main GPU volume renderer.
///
/// Manages wgpu pipelines, textures, and buffers.
/// The consumer creates this once and calls `render_*` methods each frame.
pub struct VolumeRenderer { ... }

impl VolumeRenderer {
    /// Create a new renderer for the given device and output format.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
    ) -> Self;

    /// Upload (or replace) volume data as a 3D GPU texture.
    pub fn set_volume(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &DynVolume,
    );

    /// Update transfer function LUTs on the GPU.
    pub fn set_render_params(
        &mut self,
        queue: &wgpu::Queue,
        params: &VolumeRenderParams,
    );

    /// Render the volume into the given color attachment.
    pub fn render_volume(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        camera: &Camera,
        viewport: Viewport,
    );

    /// Render a 2D reslice (MPR slice) into the given color attachment.
    pub fn render_slice(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        slice_plane: &SlicePlane,
        window_level: &WindowLevel,
        viewport: Viewport,
        thick_slab: Option<&ThickSlabParams>,
    );

    /// Render an orientation marker (annotated cube) into a corner.
    pub fn render_orientation_marker(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        viewport: Viewport,
        labels: &OrientationLabels,
    );

    /// Render crosshair overlay lines on a slice viewport.
    pub fn render_crosshair(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        viewport: Viewport,
        crosshair: &CrosshairParams,
    );

    /// Handle viewport resize.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    );
}

/// Rectangular sub-region of the render target, in pixels.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
```

### 4.2 GPU Raycasting Pipeline

Same algorithm as VTK's `vtkOpenGLGPUVolumeRayCastMapper`, in WGSL:

1. **Proxy geometry**: A unit cube, transformed to volume bounds
2. **Vertex shader**: Computes 3D texture coordinates for the front face
3. **Fragment shader**: Marches a ray from front face through the volume:
   - Sample 3D texture at each step
   - Look up transfer function (1D texture)
   - Optionally compute gradient (central differences) for shading
   - Accumulate color via selected blend mode
   - Early ray termination at opacity ≥ 0.99

### 4.3 WGSL Shader Modules

```
volren-gpu/src/shaders/
├── volume_raycast.wgsl        # Main raycasting entry point + compositing loop
├── reslice.wgsl               # 2D MPR slice sampling (+ thick slab)
├── gradient.wgsl              # Central-difference gradient estimation
├── shading.wgsl               # Blinn-Phong lighting model
├── orientation_cube.wgsl      # Orientation marker rendering
├── crosshair.wgsl             # 2D line overlay for MPR crosshairs
├── common.wgsl                # Shared uniform structs, samplers, helpers
└── fullscreen_quad.wgsl       # Utility for screen-space passes
```

### 4.4 GPU Uniform Buffer Layout

```rust
/// Packed uniform data for the volume raycaster.
/// Must match the WGSL struct layout exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VolumeUniforms {
    model_view_proj: [f32; 16],
    world_to_volume: [f32; 16],
    volume_to_world: [f32; 16],

    dimensions: [f32; 4],       // [nx, ny, nz, _pad]
    spacing: [f32; 4],          // [dx, dy, dz, _pad]
    scalar_range: [f32; 4],     // [min, max, _pad, _pad]

    step_size: f32,
    opacity_correction: f32,
    blend_mode: u32,
    shading_enabled: u32,

    ambient: f32,
    diffuse: f32,
    specular: f32,
    specular_power: f32,

    light_dir: [f32; 4],
    camera_pos: [f32; 4],

    window_center: f32,
    window_width: f32,
    num_clip_planes: u32,
    _pad: u32,

    clip_planes: [[f32; 4]; 6],
}
```

### 4.5 Texture Management

```rust
/// Manages the 3D volume texture on the GPU.
/// Handles data type conversion (i16 → r16sint, f32 → r32float, etc.)
/// and multi-component volumes.
pub(crate) struct GpuVolumeTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    dimensions: UVec3,
}

impl GpuVolumeTexture {
    /// Upload a DynVolume to the GPU.
    /// Selects the appropriate wgpu texture format based on scalar type.
    pub fn from_volume(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &DynVolume,
    ) -> Self;
}
```

### 4.6 Overlay Rendering

Thin geometry passes for UI elements rendered into the viewport:

- **Crosshair overlay**: Two colored lines showing intersection of orthogonal
  planes on a 2D slice. Configurable line colors (one per intersecting plane).
- **Orientation marker**: Small annotated cube in viewport corner that rotates
  with the camera to show patient orientation (L/R, A/P, S/I).
- **Annotation overlay**: Optional text/line rendering for measurements,
  coordinates, scalar value at cursor. (Phase 7 stretch goal)

---

## 5. Rust Crate Dependencies

### volren-core

| Crate | Purpose |
|-------|---------|
| `glam` 0.29+ | Vec3, Mat4, Quat — SIMD-optimized, supports f64 (`DVec3`, `DMat4`) |
| `bytemuck` 1.x | Safe `&[T]` → `&[u8]` for GPU upload |
| `thiserror` 2.x | Ergonomic error types |

### volren-gpu

| Crate | Purpose |
|-------|---------|
| `wgpu` 24+ | GPU abstraction (Vulkan/Metal/DX12/WebGPU) |
| `bytemuck` 1.x | Pod trait for uniform structs |
| `volren-core` | Data types, camera, params |

### Dev dependencies (both crates)

| Crate | Purpose |
|-------|---------|
| `proptest` 1.x | Property-based testing for math/coordinate transforms |
| `approx` 0.5+ | Float comparison in assertions |
| `image` 0.25+ | Snapshot test image comparison |
| `insta` 1.x | Snapshot testing framework |
| `wgpu` (with `"strict_asserts"` feature) | GPU validation in tests |
| `pollster` 0.4+ | Block on async wgpu init in tests |

### Why these choices

- **wgpu over raw OpenGL**: Cross-platform (Vulkan/Metal/DX12/WebGPU), Rust-native,
  web-ready, built-in validation. No `unsafe` GL bindings.
- **glam over nalgebra**: Simpler API, SIMD-optimized, lighter dependency graph.
  nalgebra is overkill for the linear algebra needed here.
- **No winit**: Library-only — the consumer owns the window.

---

## 6. Testing Strategy

### 6.1 Test Pyramid

```
                  ┌─────────────┐
                  │  Snapshot   │  GPU rendering compared to reference images
                  │  Tests      │  (volren-gpu, ~10 tests per blend mode/feature)
                  ├─────────────┤
                  │ Integration │  Core + GPU end-to-end: load synthetic volume,
                  │ Tests       │  render, verify output properties
                  ├─────────────┤
                  │             │  Every public function: coordinate transforms,
                  │ Unit Tests  │  TF evaluation, camera math, interaction state
                  │             │  machines, window/level formula, scalar range
                  └─────────────┘
```

### 6.2 Unit Tests (`volren-core`)

Inline `#[cfg(test)] mod tests` in every module. Target: **>90% line coverage**
of `volren-core`.

**Key test categories:**

| Module | Test focus |
|--------|-----------|
| `volume` | Construction validation, bounds checks, `scalar_range`, `from_slices` |
| `volume` (coords) | `index_to_world` / `world_to_index` round-trips, oblique orientation matrices, edge cases at volume boundaries |
| `transfer_function` | Piecewise-linear interpolation, boundary clamping, empty TF, single-node TF, LUT baking resolution |
| `window_level` | DICOM formula compliance (PS 3.3 §C.7.6.3.1.5), edge cases (width=1 step function), presets |
| `camera` | View/projection matrix correctness, `azimuth`/`elevation`/`dolly` geometry, `reset_to_bounds` fitting |
| `reslice` | `SlicePlane` axial/sagittal/coronal correctness, `point_to_world`/`world_to_point` round-trips, rotation |
| `interaction` | Trackball: drag → expected camera delta. ImageSlice: drag → expected W/L delta. Scroll → slice advance. |
| `picking` | Ray-box intersection math, voxel lookup accuracy |

**Property-based tests** (with `proptest`):
- `index_to_world(world_to_index(p)) ≈ p` for any point inside the volume
- `world_to_index(index_to_world(i)) ≈ i` for any valid index
- `SlicePlane::world_to_point(point_to_world(uv)) ≈ uv`
- TF evaluation is monotonically non-decreasing between nodes (for opacity)
- Camera `view_matrix()` produces orthonormal basis (det ≈ 1)

### 6.3 Snapshot Tests (`volren-gpu`)

Render synthetic volumes (procedural — no test fixtures needed) and compare
against reference images. Uses `insta` for snapshot management.

**Test scenes:**
1. Uniform sphere in a cube — composite rendering, verify smooth shape
2. CT head phantom — MIP rendering, verify no artifacts
3. Gradient cube — verify transfer function color mapping
4. Orthographic vs perspective — verify projection difference
5. Each blend mode — one snapshot per mode
6. Window/level — same data, different W/L, verify contrast change
7. Clip plane — half the volume clipped away
8. Reslice — axial/sagittal/coronal slices through a sphere
9. Thick slab MIP — verify slab accumulation
10. Orientation marker — verify label placement and rotation

**GPU test infrastructure:**
```rust
/// Create a headless wgpu device for testing.
/// Uses the `wgpu::Backends::all()` adapter with no surface.
fn test_device() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.unwrap();
        adapter.request_device(&Default::default(), None).await.unwrap()
    })
}
```

### 6.4 Integration Tests

In `tests/` directories at the workspace root:

- Load a synthetic volume → set TF → render → verify non-black output
- Full interaction cycle: create camera → simulate mouse drag → verify camera moved
- MPR workflow: create 3 slice planes → verify they're orthogonal and centered

### 6.5 CI Requirements

- `cargo clippy --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo doc --no-deps --document-private-items` (verifies doc completeness)
- `cargo fmt --check`
- Snapshot tests may require a GPU; use `wgpu`'s software rasterizer fallback
  or mark GPU tests with `#[ignore]` for CI without GPU.

---

## 7. Rust Code Quality Standards

### 7.1 Crate-Level Configuration

```rust
// In each crate's lib.rs:
#![deny(missing_docs)]
#![deny(unsafe_code)]        // volren-core: no unsafe at all
                              // volren-gpu: #![deny(unsafe_code)] too —
                              // wgpu handles all GPU unsafety internally
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]  // e.g. VolumeError in volume module
```

### 7.2 Error Handling

Every fallible operation returns `Result<T, E>` with typed errors:

```rust
/// Errors from volume construction and access.
#[derive(Debug, thiserror::Error)]
pub enum VolumeError {
    #[error("data length {actual} does not match dimensions {expected}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("dimensions must be > 0, got ({x}, {y}, {z})")]
    ZeroDimension { x: u32, y: u32, z: u32 },

    #[error("spacing must be > 0, got ({x}, {y}, {z})")]
    InvalidSpacing { x: f64, y: f64, z: f64 },
}

/// Errors from GPU operations.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("no volume data uploaded — call set_volume() first")]
    NoVolumeData,

    #[error("wgpu error: {0}")]
    Wgpu(#[from] wgpu::Error),

    #[error("texture dimensions ({x}×{y}×{z}) exceed device limit ({limit})")]
    TextureTooLarge { x: u32, y: u32, z: u32, limit: u32 },
}
```

### 7.3 Patterns and Idioms

| Pattern | Application |
|---------|-------------|
| **Builder pattern** | `VolumeRenderParams::builder().blend_mode(MIP).shading(off).build()` |
| **Newtype wrappers** | `Viewport`, `WindowLevel` — prevent mixing up plain tuples |
| **Sealed traits** | `Scalar` trait — users can't implement it for arbitrary types |
| **`#[non_exhaustive]`** | On all public enums — allows adding variants without breaking changes |
| **`impl Default`** | On all parameter structs — sane defaults out of the box |
| **`Send + Sync`** | All core types are `Send + Sync` (no `Rc`, no `Cell`) |
| **Zero `Clone` tax** | Large data (`Vec<T>` in `Volume`) is passed by reference; only metadata is cloneable cheaply |
| **Typestate** | `VolumeRenderer` requires `set_volume()` before `render_volume()` — enforced at runtime with `Result`, not compile time (wgpu requires runtime checks anyway) |
| **`cfg(test)`** | Test helpers are `pub(crate)`, never exposed to consumers |
| **Feature flags** | `volren-gpu/Cargo.toml`: optional `"snapshot-tests"` feature gates test utilities |

### 7.4 Documentation Standards

```rust
/// Every public item has a doc comment explaining:
/// 1. What it does (first line — shows in IDE hover)
/// 2. How it relates to VTK concepts (if applicable)
/// 3. Usage example (for important APIs)
/// 4. # Errors section (for fallible functions)
/// 5. # Panics section (if any — prefer Result over panic)
///
/// # Example
/// ```rust
/// use volren_core::camera::Camera;
/// let cam = Camera::new_perspective(
///     DVec3::new(0.0, 0.0, 5.0),
///     DVec3::ZERO,
///     30.0,
/// );
/// assert!(cam.distance() > 0.0);
/// ```
```

---

## 8. Implementation Phases

### Phase 0 — Foundation

Set up workspace, core data types, validate with unit tests.

**Deliverables:**
- Cargo workspace: `volren-core` + `volren-gpu` (empty stub)
- `Volume<T>`, `DynVolume`, `Scalar` trait
- `Volume::from_data()`, `from_slices()` with validation
- Coordinate transforms: `index_to_world`, `world_to_index`, `world_bounds`
- `Camera` with perspective/ortho, view/projection matrices
- `WindowLevel` with DICOM-compliant formula and CT presets
- `Aabb` bounding box type
- Unit tests for all of the above (including proptest round-trips)
- `#![deny(missing_docs)]` enforced from day one

### Phase 1 — GPU Raycasting MVP

Render a volume on screen. Minimal but working.

**Deliverables:**
- wgpu pipeline setup (device, queue, render pipeline)
- `GpuVolumeTexture::from_volume()` — 3D texture upload
- `VolumeUniforms` buffer
- WGSL raycaster: composite blend mode, trilinear sampling
- `TransferFunctionLut::bake()` + GPU upload
- `VolumeRenderer::render_volume()` produces correct output
- First snapshot test: render a procedural sphere volume

### Phase 2 — Transfer Functions, Blend Modes & Shading

Full-featured rendering.

**Deliverables:**
- `ColorTransferFunction` + `OpacityTransferFunction` with piecewise-linear
- Gradient estimation (central differences) in WGSL shader
- Gradient opacity modulation
- Blinn-Phong shading in WGSL
- All blend modes: Composite, MIP, MinIP, Average, Additive, Isosurface
- Window/level modulation in shader
- Early ray termination
- Snapshot tests for each blend mode

### Phase 3 — Interaction Styles

Abstract input handling.

**Deliverables:**
- `InteractionStyle` trait + `MouseEvent`/`KeyEvent` types
- `TrackballStyle` (arcball rotation, pan, zoom, scroll)
- `ImageSliceStyle` (window/level drag, slice scroll)
- Unit tests: simulate input sequence → assert camera/WL state
- `InteractionResult` feedback to consumer

### Phase 4 — 2D Reslice (MPR building block)

GPU-accelerated slice extraction.

**Deliverables:**
- `SlicePlane` with axial/sagittal/coronal constructors
- WGSL reslice shader (sample 3D texture along arbitrary plane)
- `VolumeRenderer::render_slice()` API
- Thick-slab modes (MIP, Mean, Min through slab)
- Snapshot tests: axial/sagittal/coronal through a sphere

### Phase 5 — Overlays & Decorations

Visual aids for the consumer to build a viewer.

**Deliverables:**
- Crosshair overlay renderer (two colored lines for MPR intersection)
- Orientation marker (annotated cube, corner placement)
- `CrosshairParams` and `OrientationLabels` types
- Snapshot tests for each overlay

### Phase 6 — Clip Planes, Cropping & Picking

Advanced rendering features.

**Deliverables:**
- Clip planes in raycaster (up to 6 arbitrary planes)
- Axis-aligned cropping box
- CPU-based `pick_volume()` (ray cast from screen coords → voxel)
- Unit tests for pick accuracy
- Snapshot tests for clip planes

### Phase 7 — Performance & Polish

Production readiness.

**Deliverables:**
- Adaptive step size (finer near surfaces, coarser in empty space)
- LOD: reduced resolution during interaction, full on still
- Large volume support (>512³): bricked/tiled 3D texture upload
- Render-to-texture API (for off-screen rendering / thumbnails)
- `cargo bench` benchmarks for key operations
- Documentation pass: all `///` docs reviewed, README with examples
- CI pipeline: clippy, fmt, test, doc

---

## 9. Estimated Scope

| Component | Estimated Rust LOC | Notes |
|-----------|-------------------|-------|
| `volren-core` (data model) | ~1,500 | Volume, Scalar, DynVolume, coords |
| `volren-core` (camera) | ~400 | Camera, Projection, manipulations |
| `volren-core` (transfer fn) | ~600 | TF types, LUT baking, color spaces |
| `volren-core` (interaction) | ~800 | Events, trait, Trackball, ImageSlice |
| `volren-core` (reslice, pick, W/L) | ~500 | SlicePlane, picking, WindowLevel |
| `volren-gpu` (renderer) | ~1,500 | Pipeline setup, uniform mgmt, render calls |
| `volren-gpu` (textures) | ~400 | 3D texture upload, TF texture mgmt |
| `volren-gpu` (overlays) | ~500 | Crosshair, orientation marker |
| WGSL shaders | ~800 | Raycaster, reslice, overlays |
| Tests | ~3,000 | Unit + snapshot + integration |
| **Total** | **~10,000** | |

---

## 10. Integration with `dicom-toolkit-rs`

The consumer (medical viewer application) bridges `dicom-toolkit-rs` and `volren-rs`:

```rust
use dicom_toolkit_data::FileFormat;
use dicom_toolkit_image::{DicomImage, WindowLevel as DcmWindowLevel};
use volren_core::{Volume, WindowLevel, DynVolume};

/// Example: load a DICOM series into a volren Volume.
/// This code lives in the APPLICATION, not in volren-rs.
fn load_dicom_volume(dir: &Path) -> (DynVolume, WindowLevel) {
    // 1. Read DICOM files with dicom-toolkit-data
    let mut files: Vec<FileFormat> = read_dir(dir)
        .filter_map(|p| FileFormat::read_file(&p).ok())
        .collect();

    // 2. Sort by ImagePositionPatient Z
    files.sort_by(|a, b| /* sort by slice position */);

    // 3. Extract pixel data with dicom-toolkit-image
    let images: Vec<DicomImage> = files.iter()
        .map(|f| DicomImage::from_dataset(f.dataset()).unwrap())
        .collect();

    // 4. Get metadata from first slice
    let first = &images[0];
    let spacing = DVec3::new(/* pixel_spacing */, /* slice_spacing */);
    let origin = DVec3::new(/* image_position_patient */);

    // 5. Assemble raw pixel slices
    let slices: Vec<Vec<i16>> = images.iter()
        .map(|img| img.raw_pixels_i16())
        .collect();

    // 6. Build volren Volume
    let volume = Volume::<i16>::from_slices(
        &slices.iter().map(|s| s.as_slice()).collect::<Vec<_>>(),
        first.columns, first.rows, spacing, origin, DMat3::IDENTITY,
    ).unwrap();

    // 7. Convert W/L
    let wl = WindowLevel::new(
        first.window_center.unwrap_or(128.0),
        first.window_width.unwrap_or(256.0),
    );

    (DynVolume::I16(volume), wl)
}
```

**Key principle**: volren-rs never depends on dicom-toolkit-rs. The arrow
goes one way: `application → volren-rs` and `application → dicom-toolkit-rs`.

