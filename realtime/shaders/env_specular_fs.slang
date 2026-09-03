// Pre-convolves the environment with a GGX lobe, one mip per roughness.
//
// This is the first half of the split-sum approximation: the incoming radiance
// integrated against the GGX distribution, assuming the view direction equals
// the normal. That assumption is what makes it precomputable, and it is why
// reflections at grazing angles are less accurate than the path tracer's --
// the usual and accepted cost of the technique.

#include "common.slang"

[[vk::binding(0, 2)]] Sampler2D env_source;

struct SpecularUniforms {
    float4 params; // roughness, sample_count, source_mip, unused
};
[[vk::binding(0, 3)]] ConstantBuffer<SpecularUniforms> args;

struct FullscreenOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float3 n = equirect_to_direction(i.uv);
    float3 v = n; // the split-sum assumption

    float roughness = args.params.x;
    uint count = uint(args.params.y);
    float mip = args.params.z;

    // Mirror reflection needs no convolution, and importance sampling a
    // near-delta lobe only adds noise.
    if (roughness <= 0.001) {
        return float4(env_source.SampleLevel(i.uv, 0.0).rgb, 1.0);
    }

    float3 sum = float3(0.0);
    float weight = 0.0;

    for (uint s = 0u; s < count; ++s) {
        float2 xi = hammersley(s, count);
        float3 h = importance_sample_ggx(xi, n, roughness);
        float3 l = normalize(2.0 * dot(v, h) * h - v);

        float n_dot_l = dot(n, l);
        if (n_dot_l <= 0.0) continue;

        // Weighting by N.L rather than averaging flat is the standard
        // correction; it visibly reduces the bright halo at high roughness.
        sum += env_source.SampleLevel(direction_to_equirect(l), mip).rgb * n_dot_l;
        weight += n_dot_l;
    }

    if (weight <= 0.0) {
        return float4(env_source.SampleLevel(i.uv, mip).rgb, 1.0);
    }
    return float4(sum / weight, 1.0);
}
