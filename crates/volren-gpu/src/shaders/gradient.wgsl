fn sample_vol(tc: vec3<f32>) -> f32 {
    return textureSample(vol_tex, vol_samp, tc).r;
}

fn lut_sample(t: f32) -> vec4<f32> {
    return textureSample(lut_tex, lut_samp, t);
}

fn gradient_lut_sample(t: f32) -> f32 {
    return textureSample(grad_lut_tex, grad_lut_samp, clamp(t, 0.0, 1.0)).r;
}

fn gradient(tc: vec3<f32>) -> vec3<f32> {
    let dims = u.dimensions.xyz;
    let d = vec3<f32>(1.0) / dims;
    let gx = sample_vol(tc + vec3<f32>(d.x, 0.0, 0.0))
        - sample_vol(tc - vec3<f32>(d.x, 0.0, 0.0));
    let gy = sample_vol(tc + vec3<f32>(0.0, d.y, 0.0))
        - sample_vol(tc - vec3<f32>(0.0, d.y, 0.0));
    let gz = sample_vol(tc + vec3<f32>(0.0, 0.0, d.z))
        - sample_vol(tc - vec3<f32>(0.0, 0.0, d.z));
    // Central differences: divide by 2h = 2/dims to obtain the true
    // texture-space gradient ∂f/∂tc (not scaled by 1/dims).
    return vec3<f32>(gx, gy, gz) * dims * 0.5;
}
