// Deferred PBR lighting: one fullscreen pass over the G-buffer.
//
// The BRDF is the path tracer's Principled BSDF restricted to the four
// parameters the G-buffer carries -- albedo, roughness, metallic, specular --
// so it reduces exactly to `principled_eval_f` in shaders/raytrace.metal when
// spec_tint is white and sheen, clearcoat and spec_trans are zero. Same GGX D,
// same Smith G, same Schlick F, same Burley diffuse weighted by (1 - F), same
// `0.08 * specular` dielectric F0. Where it differs it differs because the
// G-buffer cannot carry the parameter, not because the maths was re-derived.
//
// What this pass does NOT do yet: shadows (next step), image-based lighting
// from the environment (the step after), and emissive-mesh lights. That last
// one matters for reading the result -- a scene lit only by emissive geometry,
// like the Cornell box, renders its light source and nothing else here.

#include "common.slang"

[[vk::binding(0, 2)]] Sampler2D albedo_target;
[[vk::binding(1, 2)]] Sampler2D normal_target;
[[vk::binding(2, 2)]] Sampler2D surface_target;
[[vk::binding(3, 2)]] Sampler2D emission_target;
[[vk::binding(4, 2)]] Sampler2D depth_target;

// One analytic light. Area shapes are collapsed to a representative point
// carrying their area; see `light_radiance` for what that costs.
struct Light {
    float4 position;  // xyz, w = radius
    float4 direction; // xyz aim or axis, w = height
    float4 emission;  // rgb radiance or intensity, w = area
    float4 params;    // kind, cos_inner, cos_outer, unused
};

struct LightingUniforms {
    float4x4 inv_view_proj;
    float4 eye;    // xyz
    float4 params; // light_count, unused, unused, unused
};

// Declared BEFORE the storage buffer on purpose. Slang assigns Metal buffer
// indices in declaration order, and SDL_GPU requires a stage's uniform buffers
// to come before its storage buffers -- declaring `lights` first put the
// uniforms at [[buffer(1)]], where SDL never looks.
[[vk::binding(0, 3)]] ConstantBuffer<LightingUniforms> frame;

[[vk::binding(5, 2)]] StructuredBuffer<Light> lights;

static const int LIGHT_POINT = 0;
static const int LIGHT_SPOT = 1;
static const int LIGHT_DISTANT = 2;
static const int LIGHT_QUAD = 3;
static const int LIGHT_SPHERE = 4;
static const int LIGHT_DISC = 5;
static const int LIGHT_CYLINDER = 6;

// ── BRDF, mirroring shaders/raytrace.metal ───────────────────────────────────

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

// `roughness` is perceptual; squaring it gives alpha^2, as pbr_alpha_sq does.
float alpha_sq_of(float roughness) {
    float a = saturate(roughness);
    return a * a;
}

// Trowbridge-Reitz / GGX.
float ggx_D(float cos_h, float alpha_sq) {
    if (cos_h <= 0.0 || alpha_sq <= 0.0) return 0.0;
    float denom = cos_h * cos_h * (alpha_sq - 1.0) + 1.0;
    return alpha_sq / max(3.14159265 * denom * denom, 1e-30);
}

float smith_G1(float cos_v, float alpha_sq) {
    if (cos_v <= 0.0) return 0.0;
    float cos2 = cos_v * cos_v;
    float tan2 = (1.0 - cos2) / max(cos2, 1e-30);
    return 2.0 / (1.0 + sqrt(1.0 + alpha_sq * tan2));
}

float3 schlick(float cos_theta, float3 f0) {
    return f0 + (float3(1.0) - f0) * pow5(1.0 - saturate(cos_theta));
}

// Evaluates the BSDF for one light direction. `wo` points at the viewer.
float3 eval_brdf(
    float3 albedo, float roughness, float metallic, float specular,
    float3 n, float3 wo, float3 wi
) {
    float cos_o = dot(wo, n);
    float cos_i = dot(wi, n);
    if (cos_o <= 0.0 || cos_i <= 0.0) return float3(0.0);

    float3 wh = normalize(wo + wi);
    float cos_h = dot(wh, n);
    float cos_d = max(dot(wo, wh), 0.0);
    if (cos_h <= 0.0 || cos_d <= 0.0) return float3(0.0);

    // Dielectric F0 is 0.08 * specular (0.5 -> 0.04); metals use base colour.
    // The G-buffer carries no spec_tint, so the dielectric part is untinted --
    // which is the default value anyway.
    float3 f0 = lerp(float3(0.08 * specular), albedo, metallic);
    float3 F = schlick(cos_d, f0);

    float a2 = alpha_sq_of(roughness);
    float D = ggx_D(cos_h, a2);
    float G = smith_G1(cos_o, a2) * smith_G1(cos_i, a2);
    float3 specular_term = F * (D * G) / max(4.0 * cos_o * cos_i, 1e-9);

    // Burley diffuse, weighted by (1 - F) so the two lobes split the energy at
    // grazing angles.
    float3 diffuse = float3(0.0);
    if (metallic < 1.0) {
        float fd90 = 0.5 + 2.0 * roughness * cos_d * cos_d;
        float fl = 1.0 + (fd90 - 1.0) * pow5(1.0 - cos_i);
        float fv = 1.0 + (fd90 - 1.0) * pow5(1.0 - cos_o);
        diffuse = (1.0 - metallic) * albedo * (1.0 / 3.14159265) * fl * fv * (float3(1.0) - F);
    }

    return diffuse + specular_term;
}

// ── lights ───────────────────────────────────────────────────────────────────

// Direction to the light and the radiance arriving along it.
//
// Area shapes (quad, sphere, disc, cylinder) are integrated as a single
// representative point carrying the shape's area -- the standard "most
// representative point" simplification. It is correct in the far field and
// increasingly wrong as the surface approaches the source, where the path
// tracer's real area sampling pulls ahead. That is the deal this mode makes.
bool light_sample(Light l, float3 p, out float3 wi, out float3 radiance) {
    int kind = int(l.params.x);
    wi = float3(0.0, 1.0, 0.0);
    radiance = float3(0.0);

    if (kind == LIGHT_DISTANT) {
        // `direction` is the direction the light TRAVELS, so negate it.
        wi = -normalize(l.direction.xyz);
        radiance = l.emission.rgb;
        return true;
    }

    float3 to_light = l.position.xyz - p;
    float dist = length(to_light);
    if (dist <= 1e-6) return false;
    wi = to_light / dist;

    float inv_sq = 1.0 / (dist * dist);

    if (kind == LIGHT_POINT || kind == LIGHT_SPOT) {
        radiance = l.emission.rgb * inv_sq;
        if (kind == LIGHT_SPOT) {
            float3 axis = normalize(l.direction.xyz);
            float cos_angle = dot(axis, -wi);
            float cos_inner = l.params.y;
            float cos_outer = l.params.z;
            float atten = 0.0;
            if (cos_angle >= cos_inner) {
                atten = 1.0;
            } else if (cos_angle > cos_outer) {
                float t = (cos_angle - cos_outer) / max(cos_inner - cos_outer, 1e-9);
                atten = t * t * (3.0 - 2.0 * t); // smoothstep, as the path tracer does
            }
            radiance *= atten;
        }
        return true;
    }

    // Area shapes: emission is radiance, so irradiance needs the projected
    // solid angle -- area * cos(angle at the light) / distance^2.
    float area = l.emission.w;
    float cos_at_light = 1.0;
    if (kind == LIGHT_QUAD || kind == LIGHT_DISC) {
        float3 ln = normalize(l.direction.xyz);
        cos_at_light = abs(dot(ln, -wi));
    }
    radiance = l.emission.rgb * (area * cos_at_light * inv_sq);
    return true;
}

// ── entry ────────────────────────────────────────────────────────────────────

struct FullscreenOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float depth = depth_target.Sample(i.uv).r;

    float3 emission = emission_target.Sample(i.uv).rgb;

    // Depth 1 is the cleared far plane: nothing was drawn here. Until the
    // environment is wired in, that is simply background.
    if (depth >= 1.0) {
        return float4(linear_to_srgb(emission), 1.0);
    }

    // Reconstruct the world position from depth. The fullscreen UV has its
    // origin at the top-left, so NDC y is flipped against it.
    float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    float4 world_h = mul(frame.inv_view_proj, float4(ndc, depth, 1.0));
    float3 world_pos = world_h.xyz / world_h.w;

    float3 albedo = albedo_target.Sample(i.uv).rgb;
    float3 n = decode_normal(normal_target.Sample(i.uv).rg);
    float4 surface = surface_target.Sample(i.uv);
    float roughness = surface.r;
    float metallic = surface.g;
    float specular = surface.b;

    float3 wo = normalize(frame.eye.xyz - world_pos);
    // Shade the side actually facing the viewer. The path tracer flips its
    // shading normal the same way, and Lumbre's content has no reliable
    // winding, so a two-sided surface must light from whichever face is seen.
    if (dot(n, wo) < 0.0) {
        n = -n;
    }

    float3 color = emission;
    int count = int(frame.params.x);
    for (int li = 0; li < count; ++li) {
        float3 wi;
        float3 radiance;
        if (!light_sample(lights[li], world_pos, wi, radiance)) continue;

        float cos_surf = dot(n, wi);
        if (cos_surf <= 0.0) continue;

        color += eval_brdf(albedo, roughness, metallic, specular, n, wo, wi)
               * radiance * cos_surf;
    }

    return float4(linear_to_srgb(color), 1.0);
}
