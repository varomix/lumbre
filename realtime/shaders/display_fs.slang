// Display encoding: linear HDR in, display-referred sRGB out. Labels bypass it.
//
// Exposure scales the scene before the view transform, in stops. The transform
// is chosen per frame:
//   0  Standard -- clamp to [0, 1] then the sRGB OETF. Matches the path
//      tracer's `write_color`, so the two renderers agree by default.
//   1  Neutral  -- Khronos PBR Neutral. Identity below ~0.8, so base colours
//      survive untouched, and only highlights roll off. The tonemap for
//      lookdev: the desaturation a Reinhard curve causes is what it avoids.
//   2  AgX      -- Troy Sobotka's AgX, fitted polynomial. A filmic curve that
//      bleaches bright saturated light toward white instead of clipping hue.
//
// When FXAA follows, alpha carries the encoded colour's luma for it to read;
// otherwise this is the presented image and alpha is 1.
#include "common.slang"

struct DisplayUniforms {
    float4 params; // exposure scale (2^EV), transform, luma in alpha, unused
};

[[vk::binding(0, 2)]] Sampler2D hdr_color;
[[vk::binding(0, 3)]] ConstantBuffer<DisplayUniforms> display;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

// Khronos PBR Neutral, reference implementation.
float3 pbr_neutral(float3 color) {
    const float start_compression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < start_compression) return color;

    const float d = 1.0 - start_compression;
    float new_peak = 1.0 - d * d / (peak + d - start_compression);
    color *= new_peak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return lerp(color, float3(new_peak), g);
}

// AgX sigmoid, sixth-order fit in log2 space.
float3 agx_contrast(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return 15.5 * x4 * x2
         - 40.14 * x4 * x
         + 31.96 * x4
         - 6.868 * x2 * x
         + 0.4298 * x2
         + 0.1191 * x
         - 0.00232;
}

// Returns display-encoded values: AgX's curve already bakes in a ~2.2 gamma,
// so its result is linearised again before the shared sRGB encode below.
float3 agx(float3 color) {
    // Rec.709 in, AgX inset basis. Columns written out so the matrix reads the
    // same whichever convention the target uses.
    float3 c0 = float3(0.842479062253094, 0.0784335999999992, 0.0792237451477643);
    float3 c1 = float3(0.0423282422610123, 0.878468636469772, 0.0791661274605434);
    float3 c2 = float3(0.0423756549057051, 0.0784336, 0.879142973793104);
    color = c0 * color.r + c1 * color.g + c2 * color.b;

    const float min_ev = -12.47393;
    const float max_ev = 4.026069;
    color = clamp(log2(max(color, 1e-10)), min_ev, max_ev);
    color = (color - min_ev) / (max_ev - min_ev);
    color = agx_contrast(color);

    // Outset back to Rec.709.
    float3 o0 = float3(1.19687900512017, -0.0980208811401368, -0.0990297440797205);
    float3 o1 = float3(-0.0528968517574562, 1.15190312990417, -0.0989611768448433);
    float3 o2 = float3(-0.0529716355144438, -0.0980434501171241, 1.15107367264116);
    color = o0 * color.r + o1 * color.g + o2 * color.b;
    return pow(max(color, 0.0), float3(2.2));
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float3 c = hdr_color.Sample(i.uv).rgb * display.params.x;
    int transform = int(display.params.y + 0.5);
    if (transform == 1) {
        c = pbr_neutral(c);
    } else if (transform == 2) {
        c = agx(c);
    }
    float3 encoded = linear_to_srgb(c);
    float luma = dot(encoded, float3(0.299, 0.587, 0.114));
    return float4(encoded, display.params.z > 0.5 ? luma : 1.0);
}
