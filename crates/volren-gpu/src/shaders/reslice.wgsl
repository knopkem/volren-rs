struct SliceUniforms {
    world_to_volume: mat4x4<f32>,
    slice_origin:    vec4<f32>,
    slice_right:     vec4<f32>,
    slice_up:        vec4<f32>,
    slice_normal:    vec4<f32>,
    slice_extent:    vec4<f32>, // width, height, half_thickness, _
    window_level:    vec4<f32>, // center, width, _, _
    slab_params:     vec4<u32>, // mode, num_samples, _, _
};

@group(0) @binding(0) var<uniform> u: SliceUniforms;
@group(0) @binding(1) var vol_tex: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;

const THICK_MIP:   u32 = 0u;
const THICK_MINIP: u32 = 1u;
const THICK_MEAN:  u32 = 2u;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    let x = select(-1.0, 1.0, vi == 1u || vi == 2u || vi == 4u);
    let y = select(-1.0, 1.0, vi == 2u || vi == 3u || vi == 4u);
    var out: VertexOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, y * 0.5 + 0.5);
    return out;
}

fn apply_window_level(value: f32) -> f32 {
    let mapped = (value - u.window_level.x + 0.5) / u.window_level.y + 0.5;
    return clamp(mapped, 0.0, 1.0);
}

fn sample_world(world_pos: vec3<f32>) -> vec2<f32> {
    let tc4 = u.world_to_volume * vec4<f32>(world_pos, 1.0);
    let tc = tc4.xyz / tc4.w;
    if any(tc < vec3<f32>(0.0)) || any(tc > vec3<f32>(1.0)) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(textureSample(vol_tex, vol_samp, tc).r, 1.0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let base_world =
        u.slice_origin.xyz +
        (in.uv.x - 0.5) * u.slice_extent.x * u.slice_right.xyz +
        (in.uv.y - 0.5) * u.slice_extent.y * u.slice_up.xyz;

    let num_samples = max(u.slab_params.y, 1u);
    let half_thickness = u.slice_extent.z;
    var best = 0.0;
    var min_value = 0.0;
    var sum = 0.0;
    var valid = 0u;

    for (var i = 0u; i < num_samples; i++) {
        let offset = if num_samples == 1u {
            0.0
        } else {
            mix(-half_thickness, half_thickness, f32(i) / f32(num_samples - 1u))
        };
        let sample = sample_world(base_world + offset * u.slice_normal.xyz);
        if sample.y > 0.5 {
            if valid == 0u {
                best = sample.x;
                min_value = sample.x;
            } else {
                best = max(best, sample.x);
                min_value = min(min_value, sample.x);
            }
            sum += sample.x;
            valid += 1u;
        }
    }

    if valid == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let value = switch u.slab_params.x {
        case THICK_MIP: { best }
        case THICK_MINIP: { min_value }
        default: { sum / f32(valid) }
    };

    let grey = apply_window_level(value);
    return vec4<f32>(grey, grey, grey, 1.0);
}
