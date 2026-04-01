struct VolumeUniforms {
    mvp: mat4x4<f32>,
    world_to_volume: mat4x4<f32>,
    volume_to_world: mat4x4<f32>,

    dimensions: vec4<f32>,
    spacing: vec4<f32>,
    scalar_range: vec4<f32>,

    step_size: f32,
    opacity_correction: f32,
    blend_mode: u32,
    shading_enabled: u32,

    ambient: f32,
    diffuse: f32,
    specular: f32,
    specular_power: f32,

    light_position: vec4<f32>,
    camera_position: vec4<f32>,

    window_center: f32,
    window_width: f32,
    num_clip_planes: u32,
    _pad0: u32,

    clip_planes: array<vec4<f32>, 6>,
    background: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: VolumeUniforms;
@group(0) @binding(1) var vol_tex: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;
@group(0) @binding(3) var lut_tex: texture_1d<f32>;
@group(0) @binding(4) var lut_samp: sampler;
@group(0) @binding(5) var grad_lut_tex: texture_1d<f32>;
@group(0) @binding(6) var grad_lut_samp: sampler;

const BLEND_COMPOSITE: u32 = 0u;
const BLEND_MAX_INTENSITY: u32 = 1u;
const BLEND_MIN_INTENSITY: u32 = 2u;
const BLEND_AVERAGE_INTENSITY: u32 = 3u;
const BLEND_ADDITIVE: u32 = 4u;
const BLEND_ISOSURFACE: u32 = 5u;

const OPACITY_EARLY_EXIT: f32 = 0.99;
const MAX_STEPS: u32 = 2000u;
