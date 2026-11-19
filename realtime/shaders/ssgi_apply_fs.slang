// Screen-space indirect diffuse light, the apply: averages ssgi_fs's raw
// correction, traced at half resolution, over a window the size of the 4x4
// block of its texels that its rotation pattern repeats on, and adds it
// to the lit image (the pipeline blends ONE + ONE) weighted as
// ibl_with_probes weights the diffuse ambient, albedo * kd. The correction
// can be negative: where the near field is darker than the probes thought,
// it takes light away.
//
// The blur rejects taps across a depth discontinuity, as ao_blur_fs does,
// and taps on a surface facing another way, so the bounce a floor gets from
// a red wall does not bleed up the wall at their seam.
#include "shading.slang"

struct ApplyUniforms {
    float4x4 inv_view_proj;
    float4 eye;    // xyz, w unused
    float4 params; // the raw target's 1/width, 1/height; near, far
};

[[vk::binding(0, 2)]] Sampler2D ssgi_raw;
[[vk::binding(1, 2)]] Sampler2D depth_target;
[[vk::binding(2, 2)]] Sampler2D albedo_target;
[[vk::binding(3, 2)]] Sampler2D normal_target;
[[vk::binding(4, 2)]] Sampler2D surface_target;
[[vk::binding(0, 3)]] ConstantBuffer<ApplyUniforms> apply;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

// Reversed-Z depth to distance along the view axis.
float linear_depth(float d) {
    float near = apply.params.z;
    float far = apply.params.w;
    return near * far / max(near + d * (far - near), 1e-9);
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    // A window four half-resolution texels wide, centred on this pixel, so
    // it slides a half texel from one pixel to the next; texels it covers in
    // part count in part. A window fixed to the texel grid gave every 2x2
    // block of pixels the same answer, and a small, bright source's bounce
    // came out in blocks. Each tap reads depth and normal at its texel's
    // centre, where ssgi_fs read them.
    float2 texel = apply.params.xy;
    float2 c = i.uv / texel - 0.5;  // in texels, texel k's centre at k
    float2 base = floor(c);
    float2 f = c - base;
    // Five taps per axis cover the window; which five depends on the side.
    float2 first = base - 2.0 + step(0.5, f);
    float d0 = depth_target.SampleLevel(i.uv, 0.0).r;
    if (d0 <= 0.0) return float4(0.0);
    float z0 = linear_depth(d0);
    float3 n0 = decode_normal(normal_target.SampleLevel(i.uv, 0.0).rg);

    float3 sum = float3(0.0);
    float weight = 0.0;
    for (int y = 0; y < 5; ++y) {
        float ky = first.y + float(y);
        float wy = saturate(min(ky + 0.5, c.y + 2.0) - max(ky - 0.5, c.y - 2.0));
        for (int x = 0; x < 5; ++x) {
            float kx = first.x + float(x);
            float wx = saturate(min(kx + 0.5, c.x + 2.0) - max(kx - 0.5, c.x - 2.0));
            float cover = wx * wy;
            if (cover <= 0.0) continue;
            float2 uv = (float2(kx, ky) + 0.5) * texel;
            float d = depth_target.SampleLevel(uv, 0.0).r;
            if (d <= 0.0) continue;
            float z = linear_depth(d);
            float3 nk = decode_normal(normal_target.SampleLevel(uv, 0.0).rg);
            // Normals are compared unsigned: two-sided surfaces may store
            // either face.
            float w = abs(z - z0) < z0 * 0.02 && abs(dot(nk, n0)) > 0.9 ? cover : 0.0;
            sum += ssgi_raw.SampleLevel(uv, 0.0).rgb * w;
            weight += w;
        }
    }
    // Nothing alike nearby (a sliver the half-resolution trace missed):
    // nothing is changed.
    if (weight <= 0.0) return float4(0.0);
    float3 correction = sum / weight;

    // ibl_with_probes' diffuse weight.
    float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    float4 h = mul(apply.inv_view_proj, float4(ndc, d0, 1.0));
    float3 p = h.xyz / h.w;
    float3 wo = normalize(apply.eye.xyz - p);
    float3 n = dot(n0, wo) < 0.0 ? -n0 : n0;
    float3 albedo = albedo_target.SampleLevel(i.uv, 0.0).rgb;
    float4 surface = surface_target.SampleLevel(i.uv, 0.0);
    float roughness = surface.r;
    float metallic = surface.g;
    float specular = surface.b;
    float n_dot_v = saturate(dot(n, wo));
    float3 f0 = lerp(float3(0.08 * specular), albedo, metallic);
    float3 f_roughness = f0 + (max(float3(1.0 - roughness), f0) - f0) * pow5(1.0 - n_dot_v);
    float3 kd = (float3(1.0) - f_roughness) * (1.0 - metallic);
    return float4(albedo * kd * correction, 0.0);
}
