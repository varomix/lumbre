// Convolves the environment with a cosine lobe: the diffuse half of IBL.
//
// Each output texel is a direction, and its value is the irradiance arriving at
// a surface facing that way. Doing it once here turns the diffuse environment
// term in the lighting pass into a single texture fetch.
//
// The integral is a plain sum over a 64 x 32 grid of directions, each
// weighted by its solid angle and the cosine. Exact up to that mip's
// resolution, and deterministic: a few hundred sampled directions aliased
// against a bright sun and drew stripes across every surface facing it.

#include "common.slang"

[[vk::binding(0, 2)]] Sampler2D env_source;

struct IrradianceUniforms {
    float4 params; // unused, source_mip, grid width, grid height
};
[[vk::binding(0, 3)]] ConstantBuffer<IrradianceUniforms> args;

struct FullscreenOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float3 n = equirect_to_direction(i.uv);

    float mip = args.params.y;
    uint w = uint(args.params.z);
    uint h = uint(args.params.w);

    float3 sum = float3(0.0);
    // A texel's solid angle is (2 pi / w)(pi / h) sin(theta).
    float d_omega = 2.0 * LUMBRE_PI * LUMBRE_PI / float(w * h);
    for (uint y = 0u; y < h; ++y) {
        float v = (float(y) + 0.5) / float(h);
        float sin_theta = sin(v * LUMBRE_PI);
        for (uint x = 0u; x < w; ++x) {
            float2 uv = float2((float(x) + 0.5) / float(w), v);
            float cos_n = dot(n, equirect_to_direction(uv));
            if (cos_n <= 0.0) continue;
            sum += env_source.SampleLevel(uv, mip).rgb * (cos_n * sin_theta);
        }
    }

    // Irradiance / pi, so the lighting pass multiplies by albedo alone, as the
    // cosine-weighted average it replaces did.
    return float4(sum * (d_omega / LUMBRE_PI), 1.0);
}
