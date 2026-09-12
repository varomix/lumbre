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
//   - Only what the opaque pass drew is visible through glass. A second glass
//     surface behind the first shows the opaque scene, not the first surface;
//     the copy is taken once, before any transparency is drawn.
//   - A refracted sample that lands off screen falls back to the unrefracted
//     one, so glass at the frame edge stops bending.
//   - Transparent surfaces still cast opaque shadows: the shadow pass draws
//     every batch, and a depth map cannot express partial occlusion.

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

[[vk::binding(9, 2)]] StructuredBuffer<Light> lights;
[[vk::binding(10, 2)]] StructuredBuffer<ClusterRange> cluster_ranges;
[[vk::binding(11, 2)]] StructuredBuffer<uint> cluster_indices;

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

    color += ibl(
        n, wo, shading_albedo, roughness, metallic, specular,
        frame.env, env_irradiance, env_specular, env_brdf_lut);

    float view_z = dot(i.world_pos - frame.eye.xyz, frame.forward.xyz);
    float2 screen_uv = project(i.world_pos, frame.view_proj).xy;
    ClusterRange range = cluster_ranges[cluster_of(screen_uv, view_z, frame.cluster_dims, frame.cluster_depth)];
    for (uint ci = 0; ci < range.count; ++ci) {
        int li = int(cluster_indices[range.offset + ci]);
        float3 wi;
        float3 radiance;
        if (!light_sample(lights[li], i.world_pos, wi, radiance)) continue;

        float cos_surf = dot(n, wi);
        if (cos_surf <= 0.0) continue;

        float visibility = 1.0;
        if (frame.params.y > 0.5 && li == int(frame.eye.w)) {
            visibility = sun_visibility(
                i.world_pos, n, wi, view_z,
                frame.cascade_view_proj, frame.cascade_splits, frame.cascade_texel,
                frame.params.z, frame.params.w,
                shadow_map);
            if (visibility <= 0.0) continue;
        }

        color += eval_brdf(shading_albedo, roughness, metallic, specular, n, wo, wi)
               * radiance * cos_surf * visibility;
    }

    // What passes through, tinted by the base colour, added to what the surface
    // reflected. The two are already weighted against each other: the diffuse
    // lobe above was scaled down by exactly the fraction transmitted here.
    float3 transmitted = behind * lerp(float3(1.0), base, spec_trans) * spec_trans;
    return float4(color + transmitted, 1.0);
}
