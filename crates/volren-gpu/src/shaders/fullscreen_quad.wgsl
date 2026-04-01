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
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}
