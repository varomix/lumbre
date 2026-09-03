// Convolves the environment with a cosine lobe: the diffuse half of IBL.
//
// Each output texel is a direction, and its value is the irradiance arriving at
// a surface facing that way. Doing it once here turns the diffuse environment
// term in the lighting pass into a single texture fetch.
//
// Cosine-weighted hemisphere sampling, so the cosine factor is in the
// distribution rather than the estimator and the result converges with far
// fewer samples than a uniform sweep.

#include "common.slang"

[[vk::binding(0, 2)]] Sampler2D env_source;

struct IrradianceUniforms {
    float4 params; // sample_count, source_mip, unused, unused
};
[[vk::binding(0, 3)]] ConstantBuffer<IrradianceUniforms> args;

struct FullscreenOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float3 n = equirect_to_direction(i.uv);

    float3 up = abs(n.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);

    uint count = uint(args.params.x);
    float mip = args.params.y;

    float3 sum = float3(0.0);
    for (uint s = 0u; s < count; ++s) {
        float2 xi = hammersley(s, count);

        // Cosine-weighted: r = sqrt(xi.x) on the disc, lifted to the hemisphere.
        float r = sqrt(xi.x);
        float phi = 2.0 * LUMBRE_PI * xi.y;
        float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(1.0 - xi.x, 0.0)));
        float3 dir = tangent * local.x + bitangent * local.y + n * local.z;

        // Sampling a mip rather than level 0: a few hundred taps of a 4K map
        // is otherwise pure noise.
        sum += env_source.SampleLevel(direction_to_equirect(dir), mip).rgb;
    }

    // The cosine and 1/pi cancel against the cosine-weighted pdf, leaving the
    // plain average.
    return float4(sum / float(count), 1.0);
}
