// Deferred PBR lighting: one fullscreen pass over the G-buffer.
//
// The shading itself -- BRDF, lights, shadows, IBL, froxel lookup -- lives in
// shading.slang, which the forward transparent pass includes too. What stays
// here is this pass's own shape: the G-buffer bindings it reads, and the
// reconstruction of a world position from a depth sample.
//
// What this pass does NOT do: transparency (forward_fs.slang draws that after
// this one) and emissive-mesh lights. The latter matters for reading the
// result -- a scene lit only by emissive geometry, like the Cornell box,
// renders its light source and nothing else here.

#include "shading.slang"

[[vk::binding(0, 2)]] Sampler2D albedo_target;
[[vk::binding(1, 2)]] Sampler2D normal_target;
[[vk::binding(2, 2)]] Sampler2D surface_target;
[[vk::binding(3, 2)]] Sampler2D emission_target;
[[vk::binding(4, 2)]] Sampler2D depth_target;
// One layer per cascade. A comparison sampler does the depth test in hardware,
// so a PCF tap is one instruction and gets bilinear filtering of the COMPARISON
// rather than of the depths -- filtering raw depths and then comparing gives
// wrong occlusion at every edge.
[[vk::binding(5, 2)]] Sampler2DArrayShadow shadow_map;

// Image-based lighting. `env_source` is the raw environment, used for the
// background; the other two are its prefiltered forms.
[[vk::binding(6, 2)]] Sampler2D env_source;
[[vk::binding(7, 2)]] Sampler2D env_irradiance;
[[vk::binding(8, 2)]] Sampler2D env_specular;
[[vk::binding(9, 2)]] Sampler2D env_brdf_lut;

// Declared BEFORE the storage buffers on purpose. Slang assigns Metal buffer
// indices in declaration order, and SDL_GPU requires a stage's uniform buffers
// to come before its storage buffers -- declaring `lights` first put the
// uniforms at [[buffer(1)]], where SDL never looks.
[[vk::binding(0, 3)]] ConstantBuffer<LightingUniforms> frame;

[[vk::binding(10, 2)]] StructuredBuffer<Light> lights;
[[vk::binding(11, 2)]] StructuredBuffer<ClusterRange> cluster_ranges;
[[vk::binding(12, 2)]] StructuredBuffer<uint> cluster_indices;

// ── entry ────────────────────────────────────────────────────────────────────

struct FullscreenOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float depth = depth_target.Sample(i.uv).r;

    float3 emission = emission_target.Sample(i.uv).rgb;

    // Depth 1 is the cleared far plane: nothing was drawn here, so this is
    // background — the environment seen directly along the view ray.
    if (depth >= 1.0) {
        float2 ndc_bg = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
        float4 far_h = mul(frame.inv_view_proj, float4(ndc_bg, 1.0, 1.0));
        float3 view_dir = normalize(far_h.xyz / far_h.w - frame.eye.xyz);
        return float4(emission + env_background(view_dir, frame.env, env_source), 1.0);
    }

    // Reconstruct the world position from depth. The fullscreen UV has its
    // origin at the top-left, so NDC y is flipped against it.
    float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    float4 world_h = mul(frame.inv_view_proj, float4(ndc, depth, 1.0));
    float3 world_pos = world_h.xyz / world_h.w;

    float3 albedo = albedo_target.Sample(i.uv).rgb;
    float3 n = decode_normal(normal_target.Sample(i.uv).rg);
    float4 surface = surface_target.Sample(i.uv);
    float roughness = surface.r;
    float metallic = surface.g;
    float specular = surface.b;

    float3 wo = normalize(frame.eye.xyz - world_pos);
    // Shade the side actually facing the viewer. The path tracer flips its
    // shading normal the same way, and Lumbre's content has no reliable
    // winding, so a two-sided surface must light from whichever face is seen.
    //
    // Known artifact: on CLOSED geometry the front and back surfaces meet at
    // the silhouette with near-identical depth, so the depth test picks either
    // one, and a back face flipped here points away from the light and shades
    // dark. The result is a thin speckled rim exactly at the silhouette. Back
    // face culling would remove it, but Lumbre's content has no dependable
    // winding, so it would also drop real single-sided surfaces -- the wrong
    // trade for a one-pixel edge.
    if (dot(n, wo) < 0.0) {
        n = -n;
    }

    float3 color = emission;
    color += ibl(
        n, wo, albedo, roughness, metallic, specular,
        frame.env, env_irradiance, env_specular, env_brdf_lut);

    float view_z = dot(world_pos - frame.eye.xyz, frame.forward.xyz);
    ClusterRange range = cluster_ranges[cluster_of(i.uv, view_z, frame.cluster_dims, frame.cluster_depth)];
    for (uint ci = 0; ci < range.count; ++ci) {
        int li = int(cluster_indices[range.offset + ci]);
        float3 wi;
        float3 radiance;
        if (!light_sample(lights[li], world_pos, wi, radiance)) continue;

        float cos_surf = dot(n, wi);
        if (cos_surf <= 0.0) continue;

        // Only the sun is shadowed; see shadows.odin for why the other lights
        // are not.
        float visibility = 1.0;
        if (frame.params.y > 0.5 && li == int(frame.eye.w)) {
            float view_depth = dot(world_pos - frame.eye.xyz, frame.forward.xyz);
            visibility = sun_visibility(
                world_pos, n, wi, view_depth,
                frame.cascade_view_proj, frame.cascade_splits, frame.cascade_texel,
                frame.params.z, frame.params.w,
                shadow_map);
            if (visibility <= 0.0) continue;
        }

        color += eval_brdf(albedo, roughness, metallic, specular, n, wo, wi)
               * radiance * cos_surf * visibility;
    }

    return float4(color, 1.0);
}
