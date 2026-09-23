// FXAA 3.11 (quality), after Timothy Lottes. Post-process antialiasing over the
// display-encoded image: deferred shading gets no MSAA for free, and this is
// the cheap stand-in until TAA exists.
//
// Reads perceptual luma from the input's alpha, which the display pass writes.
// Works on the ENCODED image on purpose: edges are judged by how they look, and
// a linear-light luma would find almost no contrast in the shadows.
//
// Only the beauty passes through here. Labels are never filtered.

[[vk::binding(0, 2)]] Sampler2D ldr_color; // linear-filtered, clamped

struct FxaaUniforms {
    float4 params; // 1/width, 1/height, unused, unused
};
[[vk::binding(0, 3)]] ConstantBuffer<FxaaUniforms> fxaa;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

static const float EDGE_THRESHOLD = 0.125;    // minimum local contrast, relative
static const float EDGE_THRESHOLD_MIN = 0.0312; // skip near-black noise
static const float SUBPIX = 0.75;
static const int SEARCH_STEPS = 10;
static const float SEARCH_STEP_SIZES[10] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 4.0, 8.0 };

float luma_at(float2 uv) {
    return ldr_color.SampleLevel(uv, 0.0).a;
}

float luma_off(float2 uv, float2 px, float x, float y) {
    return ldr_color.SampleLevel(uv + float2(x, y) * px, 0.0).a;
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float2 px = fxaa.params.xy;
    float2 uv = i.uv;
    float4 centre = ldr_color.SampleLevel(uv, 0.0);

    float lm = centre.a;
    float ln = luma_off(uv, px, 0, -1);
    float ls = luma_off(uv, px, 0, 1);
    float le = luma_off(uv, px, 1, 0);
    float lw = luma_off(uv, px, -1, 0);

    float range_max = max(lm, max(max(ln, ls), max(le, lw)));
    float range_min = min(lm, min(min(ln, ls), min(le, lw)));
    float range = range_max - range_min;
    if (range < max(EDGE_THRESHOLD_MIN, range_max * EDGE_THRESHOLD)) {
        return float4(centre.rgb, 1.0);
    }

    float lnw = luma_off(uv, px, -1, -1);
    float lne = luma_off(uv, px, 1, -1);
    float lsw = luma_off(uv, px, -1, 1);
    float lse = luma_off(uv, px, 1, 1);

    // Sub-pixel blend: how far the centre stands out from its 3x3 average.
    float average = (2.0 * (ln + ls + le + lw) + (lnw + lne + lsw + lse)) / 12.0;
    float subpix = saturate(abs(average - lm) / range);
    subpix = smoothstep(0.0, 1.0, subpix);
    subpix = subpix * subpix * SUBPIX;

    // Is the edge horizontal or vertical?
    float edge_h = abs(lnw + lne - 2.0 * ln)
                 + 2.0 * abs(lw + le - 2.0 * lm)
                 + abs(lsw + lse - 2.0 * ls);
    float edge_v = abs(lnw + lsw - 2.0 * lw)
                 + 2.0 * abs(ln + ls - 2.0 * lm)
                 + abs(lne + lse - 2.0 * le);
    bool horizontal = edge_h >= edge_v;

    // Which side of the pixel the edge lies on.
    float l_neg = horizontal ? ln : lw;
    float l_pos = horizontal ? ls : le;
    float grad_neg = abs(l_neg - lm);
    float grad_pos = abs(l_pos - lm);
    float step_len = horizontal ? px.y : px.x;
    float l_edge;
    float gradient;
    if (grad_neg >= grad_pos) {
        step_len = -step_len;
        l_edge = 0.5 * (l_neg + lm);
        gradient = grad_neg;
    } else {
        l_edge = 0.5 * (l_pos + lm);
        gradient = grad_pos;
    }

    // Walk along the edge in both directions until the luma leaves it.
    float2 edge_uv = uv;
    float2 along;
    if (horizontal) {
        edge_uv.y += 0.5 * step_len;
        along = float2(px.x, 0.0);
    } else {
        edge_uv.x += 0.5 * step_len;
        along = float2(0.0, px.y);
    }
    float threshold = gradient * 0.25;

    float2 uv_n = edge_uv - along * SEARCH_STEP_SIZES[0];
    float2 uv_p = edge_uv + along * SEARCH_STEP_SIZES[0];
    float d_n = luma_at(uv_n) - l_edge;
    float d_p = luma_at(uv_p) - l_edge;
    bool done_n = abs(d_n) >= threshold;
    bool done_p = abs(d_p) >= threshold;
    for (int s = 1; s < SEARCH_STEPS && !(done_n && done_p); ++s) {
        if (!done_n) {
            uv_n -= along * SEARCH_STEP_SIZES[s];
            d_n = luma_at(uv_n) - l_edge;
            done_n = abs(d_n) >= threshold;
        }
        if (!done_p) {
            uv_p += along * SEARCH_STEP_SIZES[s];
            d_p = luma_at(uv_p) - l_edge;
            done_p = abs(d_p) >= threshold;
        }
    }

    float dist_n = horizontal ? uv.x - uv_n.x : uv.y - uv_n.y;
    float dist_p = horizontal ? uv_p.x - uv.x : uv_p.y - uv.y;
    bool nearer_n = dist_n < dist_p;
    float dist = min(dist_n, dist_p);
    float span = dist_n + dist_p;

    // Only blend when the nearer end's luma moves the way the centre's does;
    // otherwise this pixel is on the far side of the edge's end.
    bool centre_below = lm - l_edge < 0.0;
    bool end_ok = ((nearer_n ? d_n : d_p) < 0.0) != centre_below;
    float edge_blend = end_ok ? 0.5 - dist / span : 0.0;

    float blend = max(edge_blend, subpix);
    float2 final_uv = uv;
    if (horizontal) {
        final_uv.y += blend * step_len;
    } else {
        final_uv.x += blend * step_len;
    }
    return float4(ldr_color.SampleLevel(final_uv, 0.0).rgb, 1.0);
}
