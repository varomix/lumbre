// Ground-truth ambient occlusion (GTAO; Jimenez, Wu, Pesce and Jarabo 2016)
// over the G-buffer.
//
// For a few slices through the pixel -- planes containing the view direction,
// at angles across the screen -- march the depth buffer both ways to find the
// highest horizon within the radius, then integrate the cosine-weighted arc of
// sky between the two horizons in closed form, against the normal projected
// into the slice. The result is ambient visibility itself, not a count of
// occluded samples: a flat floor is exactly 1, the inside of a 90-degree corner
// exactly 0.5 there, and a crease's darkening has the angle-dependent shape
// the path tracer's does, not a fixed one.
//
// An occluder's weight falls off over the outer part of the radius, so a
// foreground object does not cast a dark halo onto a distant wall.
//
// The slice angle and the step positions are rotated per pixel by a 4x4
// ordered pattern, which ao_blur_fs averages back out: two slices per pixel,
// thirty-two directions per 4x4 block.
//
// Only the ambient (image-based and probe) light is darkened by this; see
// lighting_fs.
#include "common.slang"

struct AoUniforms {
    float4x4 inv_view_proj;
    float4x4 view_proj;
    float4 eye;     // xyz, w = radius in world units
    float4 forward; // xyz = view axis, w = world size of a pixel at unit depth
    float4 right;   // xyz = the camera's right, w = image height in pixels
    float4 up;      // xyz = the camera's up, w unused
    float4 params;  // intensity (an exponent), 1/width, 1/height, unused
};

[[vk::binding(0, 2)]] Sampler2D normal_target;
[[vk::binding(1, 2)]] Sampler2D depth_target;
[[vk::binding(0, 3)]] ConstantBuffer<AoUniforms> ao;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

static const int SLICES = 2;
static const int STEPS = 6; // per side
static const float HALF_PI = 1.57079633;
// An occluder counts fully out to this share of the radius, then fades to
// nothing at the radius.
static const float FALLOFF_START = 0.4;

float3 world_at(float2 uv, float depth) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 h = mul(ao.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float depth = depth_target.SampleLevel(i.uv, 0.0).r;
    if (depth <= 0.0) return float4(1.0); // background (reversed-Z far plane)

    float3 p = world_at(i.uv, depth);
    float3 n = decode_normal(normal_target.SampleLevel(i.uv, 0.0).rg);
    float3 v = normalize(ao.eye.xyz - p);
    if (dot(n, v) < 0.0) n = -n; // the face the viewer sees, as lighting does

    float radius = ao.eye.w;
    float view_z = max(dot(p - ao.eye.xyz, ao.forward.xyz), 1e-6);
    // The radius on screen, in pixels. Too small to cover a pixel: nothing to
    // find. The steps are spread out to it, at least a pixel apart.
    float radius_px = radius / (view_z * ao.forward.w);
    if (radius_px < 1.0) return float4(1.0);
    float2 texel = ao.params.yz;

    uint2 px = uint2(i.pos.xy) & 3u;
    const float bayer[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float slice_jitter = (bayer[px.y * 4 + px.x] + 0.5) / 16.0;
    float step_jitter = (bayer[(px.x * 4 + px.y) ^ 5u] + 0.5) / 16.0;

    float falloff_mul = -1.0 / (radius * (1.0 - FALLOFF_START));
    float falloff_add = 1.0 / (1.0 - FALLOFF_START);

    float visibility = 0.0;
    float open_sky = 0.0;
    [unroll]
    for (int s = 0; s < SLICES; ++s) {
        float phi = (float(s) + slice_jitter) * (3.14159265 / float(SLICES));
        float2 omega = float2(cos(phi), sin(phi));
        // The slice's direction in the world, as it runs across the screen
        // (screen y runs down, the camera's up runs up).
        float3 dir = ao.right.xyz * omega.x - ao.up.xyz * omega.y;
        float3 ortho = dir - v * dot(dir, v);
        float3 axis = normalize(cross(ortho, v));
        float3 proj_n = n - axis * dot(n, axis);
        float proj_len = length(proj_n);
        if (proj_len < 1e-4) continue;
        float cos_n = saturate(dot(proj_n, v) / proj_len);
        float gamma = (dot(ortho, proj_n) < 0.0 ? -1.0 : 1.0) * acos(cos_n);

        // The lowest a horizon can be on each side: the plane of the surface.
        float low0 = cos(gamma + HALF_PI);
        float low1 = cos(gamma - HALF_PI);
        float horizon0 = low0;
        float horizon1 = low1;
        [unroll]
        for (int k = 0; k < STEPS; ++k) {
            float t = (float(k) + step_jitter) / float(STEPS);
            float offset = max(t * t * radius_px, float(k + 1));
            float2 duv = omega * offset * texel;

            float2 uv0 = i.uv + duv;
            float d0 = depth_target.SampleLevel(uv0, 0.0).r;
            if (d0 > 0.0 && all(uv0 >= 0.0) && all(uv0 <= 1.0)) {
                float3 delta = world_at(uv0, d0) - p;
                float dist = length(delta);
                float c = dot(delta, v) / max(dist, 1e-6);
                float w = saturate(dist * falloff_mul + falloff_add);
                horizon0 = max(horizon0, lerp(low0, c, w));
            }
            float2 uv1 = i.uv - duv;
            float d1 = depth_target.SampleLevel(uv1, 0.0).r;
            if (d1 > 0.0 && all(uv1 >= 0.0) && all(uv1 <= 1.0)) {
                float3 delta = world_at(uv1, d1) - p;
                float dist = length(delta);
                float c = dot(delta, v) / max(dist, 1e-6);
                float w = saturate(dist * falloff_mul + falloff_add);
                horizon1 = max(horizon1, lerp(low1, c, w));
            }
        }

        // The two horizon angles, measured from the view direction, each at
        // most a quarter turn from the normal; then the cosine-weighted
        // visible arc between them, in closed form.
        float h0 = -acos(clamp(horizon1, -1.0, 1.0));
        float h1 = acos(clamp(horizon0, -1.0, 1.0));
        h0 = gamma + clamp(h0 - gamma, -HALF_PI, HALF_PI);
        h1 = gamma + clamp(h1 - gamma, -HALF_PI, HALF_PI);
        float sin_g = sin(gamma);
        float arc0 = (cos_n + 2.0 * h0 * sin_g - cos(2.0 * h0 - gamma)) * 0.25;
        float arc1 = (cos_n + 2.0 * h1 * sin_g - cos(2.0 * h1 - gamma)) * 0.25;
        visibility += proj_len * (arc0 + arc1);
        // What the same slice gives with nothing in the way, horizons at the
        // surface's own plane.
        open_sky += proj_len * (cos_n + gamma * sin_g);
    }
    // As a share of what these slices would see unoccluded. Their average
    // over every angle is 1 on an open surface, but two angles per pixel only
    // get there once the blur has mixed all sixteen rotations in, and on a
    // floor seen low it cannot: the rows above and below are too far off in
    // depth to blend. Unnormalized, such a floor read 0.95.
    visibility = open_sky > 1e-6 ? visibility / open_sky : 1.0;
    return float4(pow(saturate(visibility), ao.params.x));
}
