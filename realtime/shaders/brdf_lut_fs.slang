// The second half of the split-sum approximation: the environment BRDF.
//
// Integrates the GGX BRDF over the hemisphere for a given roughness and viewing
// angle, split into a scale and a bias applied to F0. It depends on nothing but
// those two parameters -- not on the environment, not on the scene -- so it is
// built once at startup and never rebuilt.
//
// x axis is N.V, y axis is roughness.

#include "common.slang"

struct FullscreenOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

// Smith geometry term with the IBL k, which differs from the direct-lighting
// one: k = a^2/2 rather than (a+1)^2/8.
float geometry_smith_ibl(float n_dot_v, float n_dot_l, float roughness) {
    float a = roughness * roughness;
    float k = a * 0.5;
    float gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    float gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float n_dot_v = max(i.uv.x, 1e-3);
    float roughness = max(i.uv.y, 1e-3);

    float3 v = float3(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    float3 n = float3(0.0, 0.0, 1.0);

    float scale = 0.0;
    float bias = 0.0;

    const uint SAMPLES = 512u;
    for (uint s = 0u; s < SAMPLES; ++s) {
        float2 xi = hammersley(s, SAMPLES);
        float3 h = importance_sample_ggx(xi, n, roughness);
        float3 l = normalize(2.0 * dot(v, h) * h - v);

        float n_dot_l = max(l.z, 0.0);
        if (n_dot_l <= 0.0) continue;

        float n_dot_h = max(h.z, 0.0);
        float v_dot_h = max(dot(v, h), 0.0);

        float g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
        float g_vis = (g * v_dot_h) / max(n_dot_h * n_dot_v, 1e-6);
        float fc = pow(1.0 - v_dot_h, 5.0);

        scale += (1.0 - fc) * g_vis;
        bias += fc * g_vis;
    }

    return float4(scale / float(SAMPLES), bias / float(SAMPLES), 0.0, 1.0);
}
