// Screen-space ambient occlusion over the G-buffer.
//
// Hemisphere sampling in world space: each sample is a point a short way off
// the surface, inside the hemisphere around its normal. Project it, read the
// depth buffer there, and it is occluded when the visible surface at that
// pixel is nearer the eye than the sample. A range check fades out occluders
// much further away than the radius, so a foreground object does not cast a
// dark halo onto a distant wall.
//
// The kernel is rotated per pixel by a 4x4 ordered pattern, which ao_blur_fs
// averages back out: few samples per pixel, no visible noise afterwards.
//
// Only the ambient (image-based) light is darkened by this; see lighting_fs.
#include "common.slang"

struct AoUniforms {
    float4x4 inv_view_proj;
    float4x4 view_proj;
    float4 eye;     // xyz, w = radius in world units
    float4 forward; // xyz = view axis, w unused
    float4 params;  // intensity, unused, unused, unused
};

[[vk::binding(0, 2)]] Sampler2D normal_target;
[[vk::binding(1, 2)]] Sampler2D depth_target;
[[vk::binding(0, 3)]] ConstantBuffer<AoUniforms> ao;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

static const int SAMPLE_COUNT = 12;

float3 world_at(float2 uv, float depth) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 h = mul(ao.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

float view_depth(float3 p) {
    return dot(p - ao.eye.xyz, ao.forward.xyz);
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float depth = depth_target.SampleLevel(i.uv, 0.0).r;
    if (depth <= 0.0) return float4(1.0); // background (reversed-Z far plane)

    float3 p = world_at(i.uv, depth);
    float3 n = decode_normal(normal_target.SampleLevel(i.uv, 0.0).rg);
    float3 wo = normalize(ao.eye.xyz - p);
    if (dot(n, wo) < 0.0) n = -n; // the face the viewer sees, as lighting does

    // Tangent frame, rotated about the normal by the pixel's place in a 4x4
    // ordered pattern.
    uint2 px = uint2(i.pos.xy) & 3u;
    const float bayer[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float angle = (bayer[px.y * 4 + px.x] + 0.5) / 16.0 * 6.2831853;
    float3 helper = abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t0 = normalize(cross(helper, n));
    float3 b0 = cross(n, t0);
    float3 t = t0 * cos(angle) + b0 * sin(angle);
    float3 b = cross(n, t);

    float radius = ao.eye.w;
    float p_depth = view_depth(p);
    // Keeps a flat surface from occluding itself through depth quantization.
    float bias = radius * 0.03;

    float occlusion = 0.0;
    for (int s = 0; s < SAMPLE_COUNT; ++s) {
        // Golden-angle spiral over the disc, lifted onto the hemisphere
        // (cosine-weighted), with samples bunched toward the centre where
        // occlusion matters most.
        float fs = (float(s) + 0.5) / float(SAMPLE_COUNT);
        float r_disc = sqrt(fs);
        float phi = float(s) * 2.3999632;
        float2 d = float2(cos(phi), sin(phi)) * r_disc;
        float3 dir = t * d.x + b * d.y + n * sqrt(max(0.0, 1.0 - fs));
        float scale = lerp(0.2, 1.0, fs * fs);
        float3 sp = p + dir * radius * scale;

        float4 clip = mul(ao.view_proj, float4(sp, 1.0));
        if (clip.w <= 1e-6) continue;
        float2 suv = float2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
        if (any(suv < 0.0) || any(suv > 1.0)) continue;

        float scene_d = depth_target.SampleLevel(suv, 0.0).r;
        if (scene_d <= 0.0) continue;
        float q_depth = view_depth(world_at(suv, scene_d));
        float s_depth = view_depth(sp);

        float range = saturate(radius / max(abs(p_depth - q_depth), 1e-6));
        occlusion += (q_depth < s_depth - bias ? 1.0 : 0.0) * range;
    }

    float visibility = 1.0 - occlusion / float(SAMPLE_COUNT);
    return float4(pow(saturate(visibility), ao.params.x));
}
