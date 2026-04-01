// volren-gpu: GPU volume raycaster
// ────────────────────────────────────────────────────────────────────────────
// Architecture:
//  • Vertex shader generates a full-screen quad (NDC clip-space, no VBO).
//  • Fragment shader fires one ray per pixel into the volume AABB.
//  • Ray marching samples a 3D texture (scalar values) and accumulates colour
//    via the chosen blend mode.
//  • A 1D RGBA transfer function LUT maps scalar → colour+opacity.
//
// Coordinate spaces used:
//  • NDC    : [-1,1]³  — vertex shader output
//  • World  : physical mm,  camera is positioned here
//  • Texture: [0,1]³   — volume texture UVW
// ────────────────────────────────────────────────────────────────────────────

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Apply DICOM window/level mapping.
fn window_level(v: f32) -> f32 {
    let t = (v - u.window_center + 0.5) / u.window_width + 0.5;
    return clamp(t, 0.0, 1.0);
}

/// Normalise a scalar to [0,1] using the volume scalar range.
fn normalise_scalar(v: f32) -> f32 {
    let range = u.scalar_range.y - u.scalar_range.x;
    if range < 1e-10 { return 0.0; }
    return clamp((v - u.scalar_range.x) / range, 0.0, 1.0);
}

/// Ray-AABB intersection (slab method).  Returns (t_near, t_far).
/// Returns t_near > t_far to signal a miss.
fn intersect_aabb(ro: vec3<f32>, inv_rd: vec3<f32>) -> vec2<f32> {
    let t1 = (vec3<f32>(0.0) - ro) * inv_rd;
    let t2 = (vec3<f32>(1.0) - ro) * inv_rd;
    let t_near = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let t_far  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    return vec2<f32>(max(t_near, 0.0), t_far);
}

/// Test whether a world-space position is outside any active clip plane.
fn is_clipped(pos_world: vec3<f32>) -> bool {
    for (var i = 0u; i < u.num_clip_planes; i++) {
        let plane = u.clip_planes[i];
        if dot(plane.xyz, pos_world) + plane.w < 0.0 {
            return true;
        }
    }
    return false;
}

// ── Fragment shader ───────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Reconstruct ray in world space from the NDC position.
    // Camera position is in world space (from uniform).
    let ndc      = vec4<f32>(in.clip_pos.x, in.clip_pos.y, 0.0, 1.0);
    // We use a proxy: fire the ray from camera toward the near-plane fragment.
    // For the raycaster we simply use camera_position as ray origin and
    // compute ray direction via inverse MVP.
    let inv_mvp  = u.volume_to_world * u.world_to_volume; // placeholder — simplified

    let cam_world = u.camera_position.xyz;

    // Map NDC xy → view space ray direction
    // clip pos is in pixels; ndc_xy is in [-1,1]
    let ndc_xy = vec2<f32>(in.clip_pos.x, in.clip_pos.y);

    // Direction in volume texture space: unproject from screen
    // We derive a ray direction in texture space by transforming two points.
    // Near point: ndc at z=0 depth
    let near_ndc   = vec4<f32>(ndc_xy * 2.0 - vec2<f32>(1.0), 0.0, 1.0);

    // Convert camera position to texture space
    let cam_vol4   = u.world_to_volume * vec4<f32>(cam_world, 1.0);
    let cam_vol    = cam_vol4.xyz / cam_vol4.w;

    // Build ray direction in texture space from camera toward this fragment.
    // Fragment world position is on the near plane of the AABB proxy geometry.
    // For a simple raycaster we project the screen UV to an AABB face.
    let frag_tc    = vec3<f32>(in.uv.x, 1.0 - in.uv.y, 0.0); // axial slice at Z=0 face

    var rd_vol     = normalize(frag_tc - cam_vol);
    let ro_vol     = cam_vol;
    let inv_rd_vol = vec3<f32>(1.0) / rd_vol;

    // Intersect unit cube [0,1]³ (volume texture AABB)
    let t_hit = intersect_aabb(ro_vol, inv_rd_vol);
    if t_hit.x > t_hit.y {
        return u.background;
    }

    let step = u.step_size / max(max(u.dimensions.x, u.dimensions.y), u.dimensions.z);
    let scalar_span = max(abs(u.scalar_range.y - u.scalar_range.x), 1e-6);

    // ── Ray march ─────────────────────────────────────────────────────────────
    var accum_color: vec3<f32> = vec3<f32>(0.0);
    var accum_alpha: f32       = 0.0;
    var mip_val:     f32       = u.scalar_range.x;  // for MIP
    var sum_val:     f32       = 0.0;                // for average
    var sum_steps:   u32       = 0u;
    var iso_hit:     bool      = false;
    var iso_pos_vol: vec3<f32> = vec3<f32>(0.0);
    var iso_normal:  vec3<f32> = vec3<f32>(0.0);
    var prev_scalar: f32       = 0.0;

    var t = t_hit.x + step * 0.5; // jitter to reduce banding

    for (var step_i = 0u; step_i < MAX_STEPS && t < t_hit.y; step_i++) {
        let tc       = ro_vol + rd_vol * t;
        let pos_world = (u.volume_to_world * vec4<f32>(tc, 1.0)).xyz;

        // Clip plane test
        if is_clipped(pos_world) {
            t += step;
            continue;
        }

        let raw_scalar = sample_vol(tc);
        let norm_scalar = normalise_scalar(raw_scalar);
        let wl_scalar   = window_level(raw_scalar);
        let g = gradient(tc);
        let grad_norm = clamp(length(g) / scalar_span, 0.0, 1.0);
        let step_advance = mix(step * 2.0, step * 0.5, grad_norm);

        switch u.blend_mode {
            case BLEND_COMPOSITE: {
                let rgba = lut_sample(wl_scalar);
                let c    = rgba.rgb;
                var a    = rgba.a;
                if a > 0.001 {
                    let grad_opacity = gradient_lut_sample(grad_norm);
                    a = 1.0 - pow(max(1.0 - a * grad_opacity, 1e-6), u.opacity_correction);
                    var shaded = c;
                    if u.shading_enabled > 0u {
                        let g_world = normalize((u.volume_to_world * vec4<f32>(g, 0.0)).xyz);
                        shaded      = phong_shade(g_world, pos_world, c);
                    }
                    accum_color += (1.0 - accum_alpha) * a * shaded;
                    accum_alpha += (1.0 - accum_alpha) * a;
                }
                if accum_alpha >= OPACITY_EARLY_EXIT {
                    break;
                }
            }
            case BLEND_MAX_INTENSITY: {
                mip_val = max(mip_val, wl_scalar);
            }
            case BLEND_MIN_INTENSITY: {
                mip_val = min(mip_val, wl_scalar);
            }
            case BLEND_AVERAGE_INTENSITY: {
                sum_val   += wl_scalar;
                sum_steps += 1u;
            }
            case BLEND_ADDITIVE: {
                let rgba = lut_sample(wl_scalar);
                let corrected_alpha =
                    1.0 - pow(max(1.0 - rgba.a * gradient_lut_sample(grad_norm), 1e-6), u.opacity_correction);
                accum_color += rgba.rgb * corrected_alpha * step_advance;
                accum_alpha  = min(accum_alpha + corrected_alpha * step_advance, 1.0);
            }
            case BLEND_ISOSURFACE: {
                // iso_value stored in scalar_range.z
                let norm_iso = (u.scalar_range.z - u.scalar_range.x) /
                               max(u.scalar_range.y - u.scalar_range.x, 1e-10);
                if step_i > 0u && sign(norm_scalar - norm_iso) != sign(prev_scalar - norm_iso) {
                    iso_hit     = true;
                    iso_pos_vol = tc;
                    iso_normal  = normalize(g);
                    break;
                }
            }
            default: {}
        }

        prev_scalar = norm_scalar;
        t += step_advance;
    }

    // ── Resolve result ────────────────────────────────────────────────────────
    switch u.blend_mode {
        case BLEND_COMPOSITE: {
            let bg = u.background.rgb * max(1.0 - accum_alpha, 0.0);
            return vec4<f32>(accum_color + bg, 1.0);
        }
        case BLEND_MAX_INTENSITY, BLEND_MIN_INTENSITY: {
            let c = lut_sample(mip_val).rgb;
            return vec4<f32>(c, 1.0);
        }
        case BLEND_AVERAGE_INTENSITY: {
            if sum_steps > 0u {
                let avg = sum_val / f32(sum_steps);
                let c   = lut_sample(avg).rgb;
                return vec4<f32>(c, 1.0);
            }
            return vec4<f32>(0.0);
        }
        case BLEND_ADDITIVE: {
            return vec4<f32>(accum_color, min(accum_alpha, 1.0));
        }
        case BLEND_ISOSURFACE: {
            if iso_hit {
                let pos_world = (u.volume_to_world * vec4<f32>(iso_pos_vol, 1.0)).xyz;
                let g_world   = normalize((u.volume_to_world * vec4<f32>(iso_normal, 0.0)).xyz);
                let base_c    = vec3<f32>(0.85, 0.85, 0.85);
                let shaded    = phong_shade(g_world, pos_world, base_c);
                return vec4<f32>(shaded, 1.0);
            }
            return u.background;
        }
        default: {
            return u.background;
        }
    }
}
