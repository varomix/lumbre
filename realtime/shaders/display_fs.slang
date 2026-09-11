// Display encoding is separate from linear HDR lighting. Labels bypass it.
#include "common.slang"
[[vk::binding(0, 2)]] Sampler2D hdr_color;
struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    return float4(linear_to_srgb(hdr_color.Sample(i.uv).rgb), 1.0);
}
