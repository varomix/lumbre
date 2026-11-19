// Shared by the two screen-space reflection passes: ssr_fs traces a ray per
// pixel, ssr_resolve_fs shares them among neighbours and composites. See
// ssr_fs.slang for the design.

#include "shading.slang"

struct SsrUniforms {
    float4x4 inv_view_proj;
    float4x4 view_proj;
    float4 eye;     // xyz, w = how far a ray is traced, world units
    float4 forward; // xyz = view axis, w = world size of a pixel at unit depth
    float4 env;     // has_env, rotation, intensity, specular mip count
    float4 size;    // width, height, 1/width, 1/height
    float4 depth;   // near, far, pyramid levels, thickness in world units
    float4 params;  // AO drawn this frame, show the reflection alone, blur levels, TAA frame
    // Baked probes, as in LightingUniforms; probe_min.w = 1 when in use.
    float4 probe_min;
    float4 probe_cell;
    float4 probe_dims;
    float4 rays;    // rays traced per pixel, gather radius in taps (1 = 3x3), unused x2
};

[[vk::binding(0, 3)]] ConstantBuffer<SsrUniforms> ssr;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

// Full strength up to the first roughness, gone by the second: past it the
// lobe is wider than what the screen holds can stand in for.
static const float ROUGHNESS_FULL = 0.5;
static const float ROUGHNESS_NONE = 0.75;
// Below this a surface is a mirror, and one ray is the whole lobe.
static const float ROUGHNESS_MIRROR = 0.04;
// The resolve pass's footprint: taps on a square grid, 3x3 with TAA and 5x5
// for a still (`rays.y`), whose spacing grows with roughness from 1 pixel to
// SSR_GATHER_MAX_STRIDE.
static const float SSR_GATHER_MAX_STRIDE = 3.0;
static const int SSR_GATHER_MAX_RADIUS = 2;

// A surface point as world position, from its uv and depth.
float3 world_at(float2 uv, float depth) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 h = mul(ssr.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

