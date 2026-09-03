// Shared by every realtime shader. `#include`d rather than imported so slangc
// needs no module search path.
//
// One file per SHADER STAGE, not per pass. SDL_GPU requires each stage's
// uniform buffers to start at [[buffer(0)]] in MSL, and Slang assigns buffer
// indices across every resource declared in a translation unit — so a vertex
// and a fragment entry point sharing a file push the fragment's uniforms to
// [[buffer(1)]] and the binding silently misses. Splitting the stages keeps
// each one's first uniform at index 0.

#ifndef LUMBRE_COMMON
#define LUMBRE_COMMON

struct CameraUniforms {
    float4x4 view_proj;
    float4 eye; // xyz, w unused
};

struct MaterialUniforms {
    float4 base_color; // rgb, w = 1 when an albedo map is bound
    float4 emission;   // rgb premultiplied by strength, w = 1 when a map is bound
    float4 params;     // roughness, metallic, specular, normal_scale
    float4 flags;      // has_mr_map, has_normal_map, unused, unused
};

// Interpolants between the G-buffer's two stages.
struct Varyings {
    float4 clip_pos : SV_Position;
    float3 world_pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 tangent : TEXCOORD3;
};

// ── octahedral normal encoding ───────────────────────────────────────────────
//
// A unit vector in two channels instead of three. Worth the arithmetic here:
// it lets the normal target be RG16F, which is both smaller and far more
// precise than the RGB8 a naive `n * 0.5 + 0.5` would need — and 8-bit normals
// band visibly across a smooth surface under a sharp specular.

float2 oct_wrap(float2 v) {
    return (1.0 - abs(v.yx)) * select(v.xy >= 0.0, 1.0, -1.0);
}

float2 encode_normal(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = n.z >= 0.0 ? n.xy : oct_wrap(n.xy);
    return n.xy;
}

float3 decode_normal(float2 f) {
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, -t, t);
    return normalize(n);
}

#endif
