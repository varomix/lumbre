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

// ── equirectangular mapping ──────────────────────────────────────────────────
//
// Must agree with `env_lookup` in shaders/raytrace.metal, or the two renderers
// light the same scene from different directions: v=0 is the +Y pole,
// u wraps from phi = atan2(z, x).

static const float LUMBRE_PI = 3.14159265358979;

float2 direction_to_equirect(float3 d) {
    d = normalize(d);
    float theta = acos(clamp(d.y, -1.0, 1.0));
    float phi = atan2(d.z, d.x);
    return float2((phi + LUMBRE_PI) / (2.0 * LUMBRE_PI), theta / LUMBRE_PI);
}

float3 equirect_to_direction(float2 uv) {
    float phi = uv.x * 2.0 * LUMBRE_PI - LUMBRE_PI;
    float theta = uv.y * LUMBRE_PI;
    float sin_theta = sin(theta);
    return float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
}

// The environment's Y rotation, applied to a lookup direction. The path tracer
// rotates by -rotation before looking up; this matches it.
float3 rotate_env(float3 d, float rotation) {
    float c = cos(-rotation);
    float s = sin(-rotation);
    return float3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

// ── sampling ─────────────────────────────────────────────────────────────────

// Van der Corput radical inverse, the second dimension of a Hammersley set.
float radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley(uint i, uint n) {
    return float2(float(i) / float(n), radical_inverse(i));
}

// GGX importance sample, returning a half-vector around `n`.
float3 importance_sample_ggx(float2 xi, float3 n, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * LUMBRE_PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    float3 h = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    float3 up = abs(n.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

// ── display encode ───────────────────────────────────────────────────────────
//
// The exact sRGB OETF, and NO tonemap -- the same transform `write_color` in
// core/color_output.odin applies. That file records why: a Reinhard curve
// desaturated highlights and flattened contrast, so the path tracer clamps
// linear radiance and applies the OETF directly. The rasterizer has to make the
// same choice or a side-by-side comparison is measuring the transform rather
// than the lighting.

float3 linear_to_srgb(float3 c) {
    c = saturate(c);
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(c, float3(1.0 / 2.4)) - 0.055;
    return select(c <= 0.0031308, lo, hi);
}

#endif
