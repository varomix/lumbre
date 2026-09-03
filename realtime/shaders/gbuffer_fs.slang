// G-buffer fragment stage: surface properties out, no lighting.
//
// Four colour targets plus depth. Deliberately a little wider than deferred PBR
// strictly needs, because these same targets are what the ground-truth pipeline
// reads later -- depth and normals are already two of the channels that phase
// has to produce.
//
// Note what is NOT here: the instance and semantic ID targets. SDL_GPU allows
// four colour targets and these use all four, so labels get their own pass --
// which is where they belong anyway, since they must be rendered jitter-free
// and un-antialiased while the beauty path is neither.

#include "common.slang"

[[vk::binding(0, 2)]] Sampler2D albedo_map;
[[vk::binding(1, 2)]] Sampler2D mr_map;
[[vk::binding(2, 2)]] Sampler2D normal_map;
[[vk::binding(3, 2)]] Sampler2D emissive_map;

[[vk::binding(0, 3)]] ConstantBuffer<MaterialUniforms> material;

struct GBuffer {
    // rgb = linear albedo, a = ambient occlusion (1 until GTAO exists).
    float4 albedo : SV_Target0;
    // Octahedral world normal. RG16F, so no visible banding on smooth shading.
    float2 normal : SV_Target1;
    // roughness, metallic, specular, unused.
    float4 surface : SV_Target2;
    // Emission can exceed 1, so this target is float.
    float4 emission : SV_Target3;
};

[shader("fragment")]
GBuffer fragmentMain(Varyings i) {
    GBuffer o;

    // The albedo map is an sRGB-format texture, so the sampler has already
    // linearized it -- and did so before filtering, which is the correct order.
    float3 base = material.base_color.rgb;
    if (material.base_color.w > 0.5) {
        base *= albedo_map.Sample(i.uv).rgb;
    }
    o.albedo = float4(base, 1.0);

    float roughness = material.params.x;
    float metallic = material.params.y;
    if (material.flags.x > 0.5) {
        // glTF metallicRoughness packing: G = roughness, B = metallic.
        float4 mr = mr_map.Sample(i.uv);
        roughness *= mr.g;
        metallic *= mr.b;
    }
    o.surface = float4(saturate(roughness), saturate(metallic), material.params.z, 0.0);

    float3 n = normalize(i.normal);
    if (material.flags.y > 0.5) {
        // Tangent-space normal map, OpenGL convention (green = +Y).
        float3 t = normalize(i.tangent.xyz - n * dot(n, i.tangent.xyz));
        float3 b = cross(n, t) * i.tangent.w;
        float3 sampled = normal_map.Sample(i.uv).xyz * 2.0 - 1.0;
        sampled.xy *= material.params.w; // normal_scale
        n = normalize(t * sampled.x + b * sampled.y + n * sampled.z);
    }
    o.normal = encode_normal(n);

    float3 emission = material.emission.rgb;
    if (material.emission.w > 0.5) {
        emission *= emissive_map.Sample(i.uv).rgb;
    }
    o.emission = float4(emission, 1.0);

    return o;
}
