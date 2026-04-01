struct CrosshairUniforms {
    position:        vec4<f32>, // u, v, thickness_px, _
    horizontal_color: vec4<f32>,
    vertical_color:   vec4<f32>,
    viewport:         vec4<f32>, // width, height, _, _
};

@group(0) @binding(0) var<uniform> u: CrosshairUniforms;

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

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let thickness_x = u.position.z / max(u.viewport.x, 1.0);
    let thickness_y = u.position.z / max(u.viewport.y, 1.0);
    let on_vertical = abs(in.uv.x - u.position.x) <= thickness_x * 0.5;
    let on_horizontal = abs(in.uv.y - u.position.y) <= thickness_y * 0.5;

    if on_vertical && on_horizontal {
        return mix(u.horizontal_color, u.vertical_color, 0.5);
    }
    if on_vertical {
        return u.vertical_color;
    }
    if on_horizontal {
        return u.horizontal_color;
    }
    return vec4<f32>(0.0);
}
