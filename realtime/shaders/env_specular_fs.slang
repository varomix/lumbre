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
    // roughness, sample_count, source_mip (the coarsest is never finer than
    // this), source texel count at mip 0
    float4 params;
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
    float texels = max(args.params.w, 1.0);
    float alpha_sq = roughness * roughness * roughness * roughness;

    // Mirror reflection needs no convolution, and importance sampling a
    // near-delta lobe only adds noise.
    if (roughness <= 0.001) {
        return float4(env_source.SampleLevel(i.uv, mip).rgb, 1.0);
    }

    float3 sum = float3(0.0);
    float weight = 0.0;

    for (uint s = 0u; s < count; ++s) {
        float2 xi = hammersley(s, count);
        float3 h = importance_sample_ggx(xi, n, roughness);
        float3 l = normalize(2.0 * dot(v, h) * h - v);

        float n_dot_l = dot(n, l);
        if (n_dot_l <= 0.0) continue;

        // Filtered importance sampling (GPU Gems 3, ch. 20): each tap reads
        // the source mip whose texel covers the solid angle the tap stands
        // for, 1 / (count * pdf). With v = n, pdf(l) = D(h) / 4. A fixed
        // coarse mip blurred glossy lobes past their width and left blotches
        // and a pinch at the poles; a fixed fine one aliases on the sun.
        float n_dot_h = saturate(dot(n, h));
        float d_denom = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;
        float d = alpha_sq / max(LUMBRE_PI * d_denom * d_denom, 1e-8);
        float omega_s = 4.0 / max(float(count) * d, 1e-8);
        // An equirect texel's solid angle shrinks with sin(theta) toward the poles.
        float sin_theta = sqrt(max(1.0 - l.y * l.y, 1e-4));
        float omega_p = 2.0 * LUMBRE_PI * LUMBRE_PI * sin_theta / texels;
        float tap_mip = max(0.5 * log2(omega_s / omega_p) + 1.0, mip);

        // Weighting by N.L rather than averaging flat is the standard
        // correction; it visibly reduces the bright halo at high roughness.
        sum += env_source.SampleLevel(direction_to_equirect(l), tap_mip).rgb * n_dot_l;
        weight += n_dot_l;
    }

    if (weight <= 0.0) {
        return float4(env_source.SampleLevel(i.uv, mip).rgb, 1.0);
    }
    return float4(sum / weight, 1.0);
}
