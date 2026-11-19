// Screen-space reflections, the resolve: shares the trace pass's rays among
// neighbours and composites the result over the lit image. See ssr_fs.slang.
//
// A glossy pixel averages its own ray with its neighbours' -- 3x3 with TAA,
// which also averages over frames, and for a still 5x5 with the outer ring
// thinned to a checkerboard -- spaced wider as roughness grows, each weighted by how alike the two surfaces are -- depth,
// normal and roughness -- so a reflection is not smeared across an edge. The
// rays' hits replace their share of the environment's reflection: a pixel
// whose rays all missed keeps exactly what the lighting pass gave it.

#include "ssr_common.slang"

[[vk::binding(0, 2)]] Sampler2D scene_color; // lit opaque image, trilinear over its blur mips
[[vk::binding(1, 2)]] Sampler2D ssr_trace;   // half resolution; rgb = colour times trust, a = trust
[[vk::binding(2, 2)]] Sampler2D albedo_target;
[[vk::binding(3, 2)]] Sampler2D normal_target;
[[vk::binding(4, 2)]] Sampler2D surface_target;
[[vk::binding(5, 2)]] Sampler2D hiz;
[[vk::binding(6, 2)]] Sampler2D env_specular;
[[vk::binding(7, 2)]] Sampler2D env_brdf_lut;
[[vk::binding(8, 2)]] Sampler2D ao_target;
// What the lighting pass reflected where the rays find nothing on screen:
// with probes, their radiance rather than the sky's.
[[vk::binding(9, 2)]] StructuredBuffer<float4> probe_irradiance_map;
[[vk::binding(10, 2)]] StructuredBuffer<float2> probe_depth;
[[vk::binding(11, 2)]] StructuredBuffer<float4> probe_radiance_map;
[[vk::binding(12, 2)]] StructuredBuffer<float4> probe_info;

#include "ssr_march.slang"

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float3 scene = scene_color.SampleLevel(i.uv, 0.0).rgb;
    bool reflection_only = ssr.params.y > 0.5;
    float3 untouched = reflection_only ? float3(0.0) : scene;

    float depth = hiz.SampleLevel(i.uv, 0.0).r;
    if (depth <= 0.0) return float4(untouched, 1.0); // background

    float4 surface = surface_target.SampleLevel(i.uv, 0.0);
    float roughness = surface.r;
    float strength = 1.0 - smoothstep(ROUGHNESS_FULL, ROUGHNESS_NONE, roughness);
    if (strength <= 0.0) return float4(untouched, 1.0);

    float3 world_pos = world_at(i.uv, depth);
    float3 n = decode_normal(normal_target.SampleLevel(i.uv, 0.0).rg);
    float3 wo = normalize(ssr.eye.xyz - world_pos);
    if (dot(n, wo) < 0.0) n = -n;
    float n_dot_v = saturate(dot(n, wo));

    float4 traced;
    if (roughness < ROUGHNESS_MIRROR) {
        // A mirror's one exact ray, at full resolution.
        float view_z = dot(world_pos - ssr.eye.xyz, ssr.forward.xyz);
        float3 origin = world_pos + n * (view_z * ssr.forward.w * 2.0);
        float origin_z = dot(origin - ssr.eye.xyz, ssr.forward.xyz);
        traced = float4(0.0);
        if (origin_z > ssr.depth.x * 2.0) {
            float3 colour;
            float trust = trace_ray(world_pos, origin, origin_z, reflect(-wo, n), roughness * CONE_TAN_PER_ALPHA, colour);
            traced = float4(colour * trust, trust);
        }
    } else {
        // The half-resolution rays: this pixel's and its neighbours', a tap
        // apart in that target, which is two pixels here.
        traced = ssr_trace.SampleLevel(i.uv, 0.0);
        float view_z = dot(world_pos - ssr.eye.xyz, ssr.forward.xyz);
        float stride = lerp(1.0, SSR_GATHER_MAX_STRIDE, saturate(roughness / ROUGHNESS_FULL));
        float4 sum = traced;
        float weight_sum = 1.0;
        int radius = int(ssr.rays.y);
        [unroll]
        for (int y = -SSR_GATHER_MAX_RADIUS; y <= SSR_GATHER_MAX_RADIUS; ++y) {
            [unroll]
            for (int x = -SSR_GATHER_MAX_RADIUS; x <= SSR_GATHER_MAX_RADIUS; ++x) {
                // Beyond 3x3, every other tap: a checkerboard keeps the
                // footprint for about half the reads.
                if ((x == 0 && y == 0) || abs(x) > radius || abs(y) > radius) continue;
                if (max(abs(x), abs(y)) > 1 && ((x + y) & 1) != 0) continue;
                float2 uv = i.uv + float2(x, y) * stride * 2.0 * ssr.size.zw;
                float d = hiz.SampleLevel(uv, 0.0).r;
                if (d <= 0.0) continue;
                float3 p = world_at(uv, d);
                float3 nk = decode_normal(normal_target.SampleLevel(uv, 0.0).rg);
                if (dot(nk, wo) < 0.0) nk = -nk;
                float rk = surface_target.SampleLevel(uv, 0.0).r;
                // Off the plane by more than a few pixels' worth, facing
                // another way, or a different finish: not the same surface.
                float off_plane = abs(dot(p - world_pos, n)) / max(view_z * ssr.forward.w * stride * 4.0, 1e-6);
                float w = saturate(1.0 - off_plane)
                        * pow(saturate(dot(n, nk)), 16.0)
                        * saturate(1.0 - abs(rk - roughness) * 8.0);
                sum += ssr_trace.SampleLevel(uv, 0.0) * w;
                weight_sum += w;
            }
        }
        traced = sum / weight_sum;
    }
    float3 found = traced.rgb;
    float share = traced.a;
    if (share <= 0.0) return float4(untouched, 1.0);
    if (reflection_only) return float4(found * strength, 1.0);

    // The same weight `ibl` gives the prefiltered environment, so what is
    // taken out here is exactly what the lighting pass put in.
    float3 albedo = albedo_target.SampleLevel(i.uv, 0.0).rgb;
    float metallic = surface.g;
    float specular = surface.b;
    float3 f0 = lerp(float3(0.08 * specular), albedo, metallic);
    float2 ab = env_brdf_lut.SampleLevel(float2(n_dot_v, roughness), 0.0).rg;
    float3 weight = f0 * ab.x + ab.y;

    // What the lighting pass put in for the ambient reflection, recomputed the
    // same way so it comes out exactly: the probes' when they are in use and
    // see this point, else the environment's.
    float3 r = reflect(-wo, n);
    float4 probe_spec = float4(0.0);
    if (ssr.probe_min.w > 0.5) {
        ProbeGrid grid;
        grid.lo = ssr.probe_min;
        grid.cell = ssr.probe_cell;
        grid.dims = ssr.probe_dims;
        float4 probe_diffuse;
        probe_lighting(world_pos, n, wo, roughness, grid,
            probe_irradiance_map, probe_depth, probe_radiance_map, probe_info, probe_diffuse, probe_spec);
    }
    float visibility = ssr.params.x > 0.5 ? ao_target.SampleLevel(i.uv, 0.0).r : 1.0;
    float3 from_env = float3(0.0);
    if (probe_spec.w > 0.5) {
        from_env = probe_spec.rgb * visibility;
    } else if (ssr.env.x > 0.5) {
        float2 spec_uv = direction_to_equirect(rotate_env(r, ssr.env.y));
        float mip = roughness * (ssr.env.w - 1.0);
        from_env = env_specular.SampleLevel(spec_uv, mip).rgb * ssr.env.z * visibility;
    }

    // The hits replace their share of the environment's reflection.
    return float4(max(scene + strength * weight * (found - share * from_env), 0.0), 1.0);
}
