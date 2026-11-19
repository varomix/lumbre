// Temporal antialiasing: blends this frame's jittered HDR image into the
// history of previous frames.
//
// Each frame the camera is shifted by a different sub-pixel offset (see
// camera_view_proj), so the history ends up averaging many sample positions
// inside each pixel. That is supersampling spread over time, and it also
// averages away the per-frame noise of the AO, reflection and fog passes.
//
// Where history comes from: for a static scene, a pixel's previous position
// follows from its depth and the two cameras alone. On a frame where
// instances moved (an edit, or playback), taa_motion_fs.slang -- this file
// with TAA_VELOCITY -- reprojects surfaces by the motion pass's velocity
// instead; the sky keeps the depth path. A separate pipeline rather than a
// branch, so static frames run exactly the code they always did: the branch
// alone reordered enough arithmetic to move TAA stills by up to 20 levels at
// edges. Either way the neighbourhood clip catches what reprojection cannot.
//
// Why history cannot be trusted as-is: it may show a surface that has just
// been uncovered, or lighting that has since changed. So it is clipped into
// the range of colours present in this frame's 3x3 neighbourhood -- variance
// clipping in YCoCg, after Salvi and Karis -- which removes ghosts while
// keeping the sub-pixel detail the history has accumulated.
//
// Works on linear HDR, before exposure and the view transform, so it is
// independent of both. Labels never pass through here.

#ifndef TAA_VELOCITY
#define TAA_VELOCITY 0
#endif

struct TaaUniforms {
    // Current pixel (NDC xy + depth) to the previous frame's clip space:
    // prev_view_proj * inverse(view_proj), the previous unjittered.
    float4x4 reproject;
    float4 size;   // width, height, 1/width, 1/height
    // Weight of the current frame when history is valid (the blend floor),
    // history valid, unused, unused.
    float4 params;
};

[[vk::binding(0, 2)]] Sampler2D current;  // point-sampled
[[vk::binding(1, 2)]] Sampler2D history;  // linear-sampled, for Catmull-Rom
[[vk::binding(2, 2)]] Sampler2D depth_tex; // point-sampled
#if TAA_VELOCITY
// Pixels moved since the last frame drawn, x right and y down (motion.odin).
// Read with Load: a storage texture, like every motion target.
[[vk::binding(3, 2)]] Texture2D<float2> velocity_tex;
#endif
[[vk::binding(0, 3)]] ConstantBuffer<TaaUniforms> taa;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float3 rgb_to_ycocg(float3 c) {
    return float3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r             - 0.5  * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

float3 ycocg_to_rgb(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// A NaN or infinity from any earlier pass would otherwise live in the history
// for good, spreading a little further every frame.
float3 sanitize(float3 c) {
    return select(isfinite(c), max(c, 0.0), 0.0);
}

// The centre of texel `px`, clamped to the image. Point-sampled at texel
// centres rather than read with Load: Slang's Metal output for Load on a
// combined texture-sampler does not compile.
float2 texel_uv(int2 px) {
    int2 clamped = clamp(px, int2(0, 0), int2(taa.size.xy) - 1);
    return (float2(clamped) + 0.5) * taa.size.zw;
}

float3 fetch(int2 px) {
    return sanitize(current.SampleLevel(texel_uv(px), 0.0).rgb);
}

// Catmull-Rom history lookup in five bilinear taps (Jimenez). Bilinear alone
// blurs the history a little every frame it is resampled, and those losses
// compound into a soft image; Catmull-Rom keeps it sharp.
float3 sample_history(float2 uv) {
    float2 pos = uv * taa.size.xy;
    float2 centre = floor(pos - 0.5) + 0.5;
    float2 f = pos - centre;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = -0.5 * f3 + f2 - 0.5 * f;
    float2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    float2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    float2 w3 = 0.5 * f3 - 0.5 * f2;

    float2 w12 = w1 + w2;
    float2 tc0 = (centre - 1.0) * taa.size.zw;
    float2 tc3 = (centre + 2.0) * taa.size.zw;
    float2 tc12 = (centre + w2 / w12) * taa.size.zw;

    float3 c = float3(0.0);
    float weight = 0.0;
    // The four corner taps are dropped: their weights are tiny and skipping
    // them is what makes this five taps rather than nine.
    float w;
    w = w12.x * w0.y;  c += history.SampleLevel(float2(tc12.x, tc0.y), 0.0).rgb * w;  weight += w;
    w = w0.x * w12.y;  c += history.SampleLevel(float2(tc0.x, tc12.y), 0.0).rgb * w;  weight += w;
    w = w12.x * w12.y; c += history.SampleLevel(float2(tc12.x, tc12.y), 0.0).rgb * w; weight += w;
    w = w3.x * w12.y;  c += history.SampleLevel(float2(tc3.x, tc12.y), 0.0).rgb * w;  weight += w;
    w = w12.x * w3.y;  c += history.SampleLevel(float2(tc12.x, tc3.y), 0.0).rgb * w;  weight += w;
    // Catmull-Rom overshoots at hard edges; the clip below bounds it anyway,
    // but a negative colour must not reach it.
    return max(c / max(weight, 1e-4), 0.0);
}

// Moves `h` toward the box centre until it lies inside the box, rather than
// clamping each axis on its own -- which shifts its hue.
float3 clip_to_box(float3 h, float3 lo, float3 hi) {
    float3 centre = 0.5 * (hi + lo);
    float3 extent = 0.5 * (hi - lo) + 1e-5;
    float3 offset = h - centre;
    float3 units = abs(offset / extent);
    float largest = max(units.x, max(units.y, units.z));
    return largest > 1.0 ? centre + offset / largest : h;
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    int2 px = int2(i.pos.xy);
    float3 c = fetch(px);

    // Neighbourhood statistics, and the nearest depth in it. Reprojecting
    // from the nearest surface makes an edge's history follow the object in
    // front rather than the background behind it. Reversed-Z: nearer is
    // greater.
    float3 m1 = float3(0.0);
    float3 m2 = float3(0.0);
    float3 lo = float3(1e30);
    float3 hi = float3(-1e30);
    float nearest = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 q = px + int2(x, y);
            float3 s = rgb_to_ycocg(fetch(q));
            m1 += s;
            m2 += s * s;
            lo = min(lo, s);
            hi = max(hi, s);
            float d = depth_tex.SampleLevel(texel_uv(q), 0.0).r;
            nearest = max(nearest, d);
        }
    }

    if (taa.params.y < 0.5) {
        return float4(c, 1.0);
    }

    // Where that surface was last frame. The pixel's own position with the
    // nearest neighbour's depth: the neighbour's position would shift the
    // whole image by its offset.
    float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    float4 prev = mul(taa.reproject, float4(ndc, nearest, 1.0));
    float2 prev_ndc = prev.xy / prev.w;
    float2 prev_uv = float2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
#if TAA_VELOCITY
    // A surface follows its own motion, read at its nearest neighbour for the
    // same reason as the depth: an edge's history follows the object in
    // front. Reversed-Z clears to 0, so `nearest > 0` means a surface.
    if (nearest > 0.0) {
        int2 q = px;
        float best = -1.0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                int2 n = clamp(px + int2(x, y), int2(0, 0), int2(taa.size.xy) - 1);
                float d = depth_tex.SampleLevel(texel_uv(n), 0.0).r;
                if (d > best) { best = d; q = n; }
            }
        }
        float2 v = velocity_tex.Load(int3(q, 0));
        prev_uv = i.uv - v * taa.size.zw;
        prev = float4(0.0, 0.0, 0.0, 1.0);
    }
#endif
    if (prev.w <= 0.0 || any(prev_uv < 0.0) || any(prev_uv > 1.0)) {
        return float4(c, 1.0); // came from off screen: nothing to blend
    }

    float3 h = rgb_to_ycocg(sample_history(prev_uv));

    // Variance clip, intersected with the plain min/max box: the variance box
    // alone can reach past every colour actually present.
    float3 mean = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    const float GAMMA = 1.0;
    float3 box_lo = max(lo, mean - GAMMA * sigma);
    float3 box_hi = min(hi, mean + GAMMA * sigma);
    h = ycocg_to_rgb(clip_to_box(h, box_lo, box_hi));

    // Luminance-weighted blend (Karis): a single very bright sample would
    // otherwise dominate the average and flicker as the jitter moves it.
    float alpha = taa.params.x;
    float wc = alpha / (1.0 + dot(c, float3(0.2126, 0.7152, 0.0722)));
    float wh = (1.0 - alpha) / (1.0 + dot(h, float3(0.2126, 0.7152, 0.0722)));
    float3 result = (c * wc + h * wh) / max(wc + wh, 1e-6);
    return float4(result, 1.0);
}
