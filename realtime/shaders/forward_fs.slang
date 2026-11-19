// Forward+ transparency: the pass that draws glass.
//
// Transmissive surfaces cannot go through the G-buffer, which stores one
// opaque sample per pixel. They are drawn after deferred lighting instead,
// shaded with the same BRDF, lights, shadows and froxel grid (shading.slang),
// and composited against a COPY of the lit opaque image -- `refraction_src`,
// blitted before this pass, because a pass cannot read the target it writes.
//
// What is behind the surface is sampled along the refracted direction, offset
// by the material's thickness: cheap screen-space refraction, not a traced ray.
// Its limits are worth stating plainly:
//
//   - Glass is laid down farthest first, each object's back faces then its
//     front faces, over a fresh copy each time (draw_forward), so glass behind
//     glass is seen through it. Objects that interpenetrate, or more than
//     GLASS_LAYERED_MAX of them, can still come out in the wrong order.
//   - A refracted sample that lands off screen falls back to the unrefracted
//     one, so glass at the frame edge stops bending.
//   - Glass casts tinted, partial shadows through glass maps beside the depth
//     maps (glass_lookup in shading.slang); the fog's shafts and the probe
//     bake do not see them.

#include "shading.slang"

// Material maps, as the G-buffer pass binds them.
[[vk::binding(0, 2)]] Sampler2D albedo_map;
[[vk::binding(1, 2)]] Sampler2D mr_map;
[[vk::binding(2, 2)]] Sampler2D normal_map;
[[vk::binding(3, 2)]] Sampler2D emissive_map;

[[vk::binding(4, 2)]] Sampler2DArrayShadow shadow_map;

// No `env_source` here, unlike the deferred pass: this one never draws the
// background, and Slang strips a binding nothing reads -- which leaves a hole
// in the set that SDL_GPU's contiguous-binding rule rejects.
[[vk::binding(5, 2)]] Sampler2D env_irradiance;
[[vk::binding(6, 2)]] Sampler2D env_specular;
[[vk::binding(7, 2)]] Sampler2D env_brdf_lut;

// The lit opaque image, copied before this pass.
[[vk::binding(8, 2)]] Sampler2D refraction_src;

// Uniforms before storage buffers, as in lighting_fs.slang.
[[vk::binding(0, 3)]] ConstantBuffer<MaterialUniforms> material;
[[vk::binding(1, 3)]] ConstantBuffer<LightingUniforms> frame;

[[vk::binding(9, 2)]] Sampler2DArrayShadow spot_shadow_map;
[[vk::binding(10, 2)]] Sampler2DArrayShadow omni_shadow_map;
[[vk::binding(11, 2)]] Sampler2D ltc_transform;
[[vk::binding(12, 2)]] Sampler2D ltc_magnitude;
// The omni map again, read as plain depths for the soft-shadow blocker search.
[[vk::binding(13, 2)]] Sampler2DArray omni_depth_map;
// What glass lets through to a point, per shadow type; see glass_lookup in
// shading.slang. A 1x1 stand-in when the scene has no glass.
[[vk::binding(14, 2)]] Texture2DArray<float4> sun_glass_map;
[[vk::binding(15, 2)]] Texture2DArray<float4> spot_glass_map;
[[vk::binding(16, 2)]] Texture2DArray<float4> omni_glass_map;

[[vk::binding(17, 2)]] StructuredBuffer<Light> lights;
[[vk::binding(18, 2)]] StructuredBuffer<ClusterRange> cluster_ranges;
[[vk::binding(19, 2)]] StructuredBuffer<uint> cluster_indices;
// Baked irradiance probes; read only when frame.probe_min.w is set.
[[vk::binding(20, 2)]] StructuredBuffer<float4> probe_irradiance_map;
[[vk::binding(21, 2)]] StructuredBuffer<float2> probe_depth;
[[vk::binding(22, 2)]] StructuredBuffer<float4> probe_radiance_map;
[[vk::binding(23, 2)]] StructuredBuffer<float4> probe_info;

// Screen uv of a world point, or a negative w when it is behind the camera.
float3 project(float3 world_pos, float4x4 view_proj) {
    float4 clip = mul(view_proj, float4(world_pos, 1.0));
    if (clip.w <= 1e-6) return float3(0.0, 0.0, -1.0);
    float2 ndc = clip.xy / clip.w;
    return float3(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, 1.0);
}

[shader("fragment")]
float4 fragmentMain(Varyings i) : SV_Target {
    float3 base = material.base_color.rgb;
    if (material.base_color.w > 0.5) {
        base *= albedo_map.Sample(i.uv).rgb;
    }

    float roughness = material.params.x;
    float metallic = material.params.y;
    float specular = material.params.z;
    if (material.flags.x > 0.5) {
        float4 mr = mr_map.Sample(i.uv);
        roughness *= mr.g;
        metallic *= mr.b;
    }
    roughness = saturate(roughness);
    metallic = saturate(metallic);

    float spec_trans = saturate(material.flags.z);
    float ior = max(material.flags.w, 1.0);

    float3 n = normalize(i.normal);
    if (material.flags.y > 0.5) {
        float3 t = normalize(i.tangent.xyz - n * dot(n, i.tangent.xyz));
        float3 b = cross(n, t) * i.tangent.w;
        float3 sampled = normal_map.Sample(i.uv).xyz * 2.0 - 1.0;
        sampled.xy *= material.params.w;
        n = normalize(t * sampled.x + b * sampled.y + n * sampled.z);
    }

    float3 wo = normalize(frame.eye.xyz - i.world_pos);
    // Two-sided, as every other pass here is: Lumbre's content has no
    // dependable winding, and the back of a glass shell is as real as the front.
    bool backface = dot(n, wo) < 0.0;
    if (backface) {
        n = -n;
    }
    // Glass is laid down one face at a time, back faces first, so the front
    // refracts a copy with the back already in it (draw_forward). Decided by
    // the geometric normal, which a normal map cannot flip.
    bool geometric_back = dot(i.normal, wo) < 0.0;
    float faces = frame.forward_params.w;
    if ((faces == 1.0 && !geometric_back) || (faces == 2.0 && geometric_back)) discard;

    // ── what is behind the surface ───────────────────────────────────────────
    // Refract through the surface and look up where that ray leaves it, a
    // thickness later. eta is the ratio entering the denser medium; a back face
    // is leaving it.
    float eta = backface ? ior : 1.0 / ior;
    float3 refracted = refract(-wo, n, eta);
    float3 screen = project(i.world_pos + refracted * frame.forward_params.x, frame.view_proj);
    float2 behind_uv = screen.z > 0.0 ? screen.xy : project(i.world_pos, frame.view_proj).xy;
    // Total internal reflection gives a zero vector; fall back to straight
    // through rather than sampling a garbage direction.
    if (dot(refracted, refracted) < 1e-6 || any(behind_uv < 0.0) || any(behind_uv > 1.0)) {
        behind_uv = project(i.world_pos, frame.view_proj).xy;
    }
    float3 behind = refraction_src.SampleLevel(saturate(behind_uv), 0.0).rgb;

    // A back face is the inside of the glass, seen through its front. What it
    // reflects goes back into the glass, which this pass cannot follow, so it
    // only passes light through. Shaded like an outside surface instead, every
    // back face added a second reflection of the sky and a diffuse glow, and
    // the total internal reflection at a ball's rim turned it milky white.
    if (geometric_back && faces != 0.0) {
        return float4(behind * lerp(float3(1.0), base, spec_trans), 1.0);
    }

    // ── what the surface itself reflects ─────────────────────────────────────
    float3 color = float3(0.0);
    if (material.emission.w > 0.5) {
        color += material.emission.rgb * emissive_map.Sample(i.uv).rgb;
    } else {
        color += material.emission.rgb;
    }

    // A transmissive surface has no diffuse lobe: what would have scattered
    // diffusely is what passes through instead. The path tracer weights its
    // diffuse by the same (1 - spec_trans) (core/principled.odin). Scaling the
    // albedo does it here without touching the dielectric F0, which is
    // 0.08 * specular and independent of base colour.
    float3 shading_albedo = base * (1.0 - spec_trans);

    // A clear pane has almost no diffuse lobe, so probes change it little; a
    // frosted or tinted one, or a partly transmissive solid, takes them like
    // any opaque surface.
    float4 probe = float4(0.0);
    float4 probe_spec = float4(0.0);
    if (frame.probe_min.w > 0.5) {
        probe_lighting(i.world_pos, n, wo, roughness, probe_grid(frame),
            probe_irradiance_map, probe_depth, probe_radiance_map, probe_info, probe, probe_spec);
    }
    color += ibl_with_probes(
        n, wo, shading_albedo, roughness, metallic, specular,
        frame.env, env_irradiance, env_specular, env_brdf_lut, probe, probe_spec);

    float view_z = dot(i.world_pos - frame.eye.xyz, frame.forward.xyz);
    float2 screen_uv = project(i.world_pos, frame.view_proj).xy;
    ClusterRange range = cluster_ranges[cluster_of(screen_uv, view_z, frame.cluster_dims, frame.cluster_depth)];
    for (uint ci = 0; ci < range.count; ++ci) {
        int li = int(cluster_indices[range.offset + ci]);

        if (int(lights[li].params.x) >= LIGHT_QUAD) {
            // Shade first: the soft shadow is the expensive part, and a point
            // the light leaves dark -- behind it, or out of its reach -- needs
            // none.
            float3 area = area_light(
                lights[li], i.world_pos, n, wo, shading_albedo, roughness, metallic, specular,
                ltc_transform, ltc_magnitude);
            if (max(area.r, max(area.g, area.b)) <= 1e-6) continue;
            int slot = omni_shadow_slot(lights[li]);
            if (slot >= 0) {
                float3 la, lb;
                light_shadow_axes(lights[li], i.world_pos, la, lb);
                float3 to_light = normalize(lights[li].position.xyz - i.world_pos);
                area *= omni_visibility_soft(
                    i.world_pos, n, to_light, lights[li].position.xyz, slot, frame.omni_range[slot],
                    pcss_spin(i.clip_pos.xy, frame.forward_params.z), la, lb,
                    frame.omni_view_proj, omni_shadow_map, omni_depth_map);
                area *= omni_glass(omni_glass_map, i.world_pos, lights[li].position.xyz, slot, frame.omni_view_proj);
            }
            color += area;
            continue;
        }

        float3 wi;
        float3 radiance;
        if (!light_sample(lights[li], i.world_pos, wi, radiance)) continue;

        float cos_surf = dot(n, wi);
        if (cos_surf <= 0.0) continue;

        float visibility = 1.0;
        // What glass between lets through, tinted; 1 with none.
        float3 through = float3(1.0);
        int spot = spot_shadow_layer(lights[li]);
        if (spot >= 0) {
            visibility = spot_visibility(
                i.world_pos, n, wi, lights[li].position.xyz, spot,
                frame.spot_view_proj[spot], frame.spot_texel[spot / 4][spot % 4],
                spot_shadow_map);
            if (visibility <= 0.0) continue;
            through = spot_glass(spot_glass_map, i.world_pos, lights[li].position.xyz, spot, frame.spot_view_proj[spot]);
        }
        int omni = omni_shadow_slot(lights[li]);
        if (omni >= 0) {
            float3 la, lb;
            light_shadow_axes(lights[li], i.world_pos, la, lb);
            visibility = omni_visibility_soft(
                i.world_pos, n, wi, lights[li].position.xyz, omni, frame.omni_range[omni],
                pcss_spin(i.clip_pos.xy, frame.forward_params.z), la, lb,
                frame.omni_view_proj, omni_shadow_map, omni_depth_map);
            if (visibility <= 0.0) continue;
            through = omni_glass(omni_glass_map, i.world_pos, lights[li].position.xyz, omni, frame.omni_view_proj);
        }
        if (frame.params.y > 0.5 && li == int(frame.eye.w)) {
            visibility = sun_visibility(
                i.world_pos, n, wi, view_z,
                frame.cascade_view_proj, frame.cascade_splits, frame.cascade_texel,
                frame.params.z, frame.params.w,
                shadow_map);
            if (visibility <= 0.0) continue;
            through = sun_glass(sun_glass_map, i.world_pos, view_z, frame.cascade_view_proj, frame.cascade_splits);
        }

        color += eval_brdf(shading_albedo, roughness, metallic, specular, n, wo, wi)
               * radiance * cos_surf * visibility * through;
    }

    // What passes through, tinted by the base colour, added to what the surface
    // reflected. The two are already weighted against each other: the diffuse
    // lobe above was scaled down by exactly the fraction transmitted here.
    float3 transmitted = behind * lerp(float3(1.0), base, spec_trans) * spec_trans;
    return float4(color + transmitted, 1.0);
}
