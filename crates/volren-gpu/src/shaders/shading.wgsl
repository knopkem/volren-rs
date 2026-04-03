fn phong_shade(normal: vec3<f32>, pos_world: vec3<f32>, color: vec3<f32>) -> vec3<f32> {
    let n = normalize(normal);
    let l = normalize(u.light_position.xyz - pos_world);
    let v = normalize(u.camera_position.xyz - pos_world);
    let h = normalize(l + v);

    // Two-sided shading: illuminate surfaces regardless of normal orientation.
    let diff = abs(dot(n, l));
    let spec = pow(max(abs(dot(n, h)), 0.0), u.specular_power);

    return color * (u.ambient + u.diffuse * diff) + vec3<f32>(u.specular * spec);
}
