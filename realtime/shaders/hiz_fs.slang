// Builds one level of the hierarchical depth buffer that ssr_fs marches.
//
// Level 0 is a copy of the depth buffer. Every level above holds, per texel,
// the NEAREST depth among the texels it covers below -- the maximum, since the
// camera is reversed-Z. A ray that is nearer the eye than that value cannot hit
// anything in the whole cell, which is what lets the march skip it in one step.
// (Occlusion culling wants the opposite pyramid, the farthest depth. This is
// not that.)
//
// A level is half the one below, rounded down, so an odd source dimension
// leaves a last row or column with no cell of its own. The reduction reads a
// third texel in that direction and the last cell absorbs it; reading it for
// every cell only makes the others more conservative, never wrong.

struct HizUniforms {
    // source level, 1 when copying the depth buffer, source width, height
    float4 params;
};

// The depth buffer for level 0, the pyramid itself for the rest. Read with a
// point sampler at texel centres rather than Load: Slang's Metal output for
// Load on a combined sampler does not compile.
[[vk::binding(0, 2)]] Sampler2D source;
[[vk::binding(0, 3)]] ConstantBuffer<HizUniforms> hiz;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    int2 px = int2(i.pos.xy);
    if (hiz.params.y > 0.5) {
        return float4(source.SampleLevel(i.uv, 0.0).r, 0.0, 0.0, 1.0);
    }

    float level = hiz.params.x;
    int2 size = int2(hiz.params.zw);
    int2 last = size - 1;
    int2 base = px * 2;
    // Two texels each way, three where the source dimension is odd.
    int2 taps = int2(2) + (size & 1);

    float nearest = 0.0;
    for (int y = 0; y < taps.y; ++y) {
        for (int x = 0; x < taps.x; ++x) {
            int2 p = min(base + int2(x, y), last);
            float2 uv = (float2(p) + 0.5) / hiz.params.zw;
            nearest = max(nearest, source.SampleLevel(uv, level).r);
        }
    }
    return float4(nearest, 0.0, 0.0, 1.0);
}
