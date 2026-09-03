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
// One layer per cascade. A comparison sampler does the depth test in hardware,
// so a PCF tap is one instruction and gets bilinear filtering of the COMPARISON
// rather than of the depths -- filtering raw depths and then comparing gives
// wrong occlusion at every edge.
[[vk::binding(6, 2)]] SamplerComparisonState shadow_map_sampler;
[[vk::binding(7, 2)]] Texture2DArray shadow_map;

// Image-based lighting. `env_source` is the raw environment, used for the
// background; the other two are its prefiltered forms.
[[vk::binding(8, 2)]] Sampler2D env_source;
[[vk::binding(9, 2)]] Sampler2D env_irradiance;
[[vk::binding(10, 2)]] Sampler2D env_specular;
[[vk::binding(11, 2)]] Sampler2D env_brdf_lut;

// One analytic light. Area shapes are collapsed to a representative point
// carrying their area; see `light_radiance` for what that costs.
struct Light {
    float4 position;  // xyz, w = radius
    float4 direction; // xyz aim or axis, w = height
    float4 emission;  // rgb radiance or intensity, w = area
    float4 params;    // kind, cos_inner, cos_outer, unused
};

#define CASCADE_COUNT 4

struct LightingUniforms {
    float4x4 inv_view_proj;
    float4x4 cascade_view_proj[CASCADE_COUNT];
    float4 eye;          // xyz, w unused
    float4 forward;      // camera view axis, for picking a cascade by depth
    float4 cascade_splits; // view-space far distance of each cascade
    float4 cascade_texel;  // world-space texel size of each cascade
    float4 params;       // light_count, shadows_enabled, depth_bias, texel_size
    // has_env, rotation (radians), intensity, specular mip count.
    float4 env;
};

// Declared BEFORE the storage buffer on purpose. Slang assigns Metal buffer
// indices in declaration order, and SDL_GPU requires a stage's uniform buffers
// to come before its storage buffers -- declaring `lights` first put the
// uniforms at [[buffer(1)]], where SDL never looks.
[[vk::binding(0, 3)]] ConstantBuffer<LightingUniforms> frame;

[[vk::binding(5, 2)]] StructuredBuffer<Light> lights;

// Which cascade covers this point, by distance along the view axis.
int pick_cascade(float view_depth, float4 splits) {
    for (int i = 0; i < CASCADE_COUNT; ++i) {
        if (view_depth <= splits[i]) return i;
    }
    return CASCADE_COUNT - 1;
}

// Fraction of light reaching `world_pos`, 0 fully shadowed to 1 fully lit.
//
// The bias scales with the surface's slope: a polygon nearly edge-on to the
// light spans many depth values within one shadow texel, and a constant bias
// large enough for that case detaches shadows from every other surface.
float sun_visibility(
    float3 world_pos, float3 n, float3 wi, float view_depth,
    float4x4 cascade_view_proj[CASCADE_COUNT], float4 splits, float4 texel_world,
    float depth_bias, float texel_size,
    Texture2DArray shadow_tex, SamplerComparisonState shadow_smp
) {
    int cascade = pick_cascade(view_depth, splits);

    // Normal-offset bias: move the lookup off the surface along its normal,
    // scaled by how much world one shadow texel covers here, and by how
    // grazing the light is. A depth bias alone cannot fix a surface curving
    // away from the light -- its slope runs to infinity at the terminator, so
    // any capped bias speckles along the rim. Offsetting in space instead of in
    // depth has no such limit.
    float cos_theta = saturate(dot(n, wi));
    float offset_scale = texel_world[cascade] * 1.5 * (2.0 - cos_theta);
    world_pos += n * offset_scale;

    float4 light_clip = mul(cascade_view_proj[cascade], float4(world_pos, 1.0));
    float3 proj = light_clip.xyz / light_clip.w;

    // Clip space to texture space; the shadow map's y runs opposite to NDC's.
    float2 uv = float2(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || proj.z > 1.0) {
        return 1.0; // outside this cascade: unshadowed rather than black
    }

    float slope = sqrt(1.0 - cos_theta * cos_theta) / max(cos_theta, 1e-3);
    float bias = depth_bias * (1.0 + min(slope, 8.0));
    float compare_depth = proj.z - bias;

    // 3x3 PCF. Enough to soften the staircase on a 2048 map without turning
    // the loop into the cost of the whole pass.
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 offset = float2(x, y) * texel_size;
            sum += shadow_tex.SampleCmpLevelZero(
                shadow_smp, float3(uv + offset, float(cascade)), compare_depth);
        }
    }
    return sum / 9.0;
}

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

// ── image-based lighting ─────────────────────────────────────────────────────

// Rotation and intensity are applied here rather than baked into the
// prefiltered maps: both commute with the convolution, so turning the HDRI
// costs a lookup transform instead of rebuilding every map.
float3 env_background(float3 dir, float4 env, Sampler2D source) {
    if (env.x < 0.5) return float3(0.0);
    float2 uv = direction_to_equirect(rotate_env(dir, env.y));
    return source.SampleLevel(uv, 0.0).rgb * env.z;
}

// The split-sum approximation: prefiltered radiance times the environment BRDF.
float3 ibl(
    float3 n, float3 wo, float3 albedo, float roughness, float metallic, float specular,
    float4 env, Sampler2D irradiance_map, Sampler2D specular_map, Sampler2D brdf_lut
) {
    if (env.x < 0.5) return float3(0.0);

    float n_dot_v = saturate(dot(n, wo));
    float3 f0 = lerp(float3(0.08 * specular), albedo, metallic);

    // Diffuse: irradiance along the normal, minus what the specular lobe takes.
    // The roughness-aware Fresnel keeps rough dielectrics from losing energy at
    // grazing angles the way a plain Schlick term does here.
    float3 f_roughness = f0 + (max(float3(1.0 - roughness), f0) - f0) * pow5(1.0 - n_dot_v);
    float3 kd = (float3(1.0) - f_roughness) * (1.0 - metallic);

    float2 irr_uv = direction_to_equirect(rotate_env(n, env.y));
    float3 diffuse = irradiance_map.SampleLevel(irr_uv, 0.0).rgb * albedo * kd;

    // Specular: the prefiltered lobe along the reflection vector, at the mip
    // matching this roughness.
    float3 r = reflect(-wo, n);
    float2 spec_uv = direction_to_equirect(rotate_env(r, env.y));
    float mip = roughness * (env.w - 1.0);
    float3 prefiltered = specular_map.SampleLevel(spec_uv, mip).rgb;

    float2 ab = brdf_lut.SampleLevel(float2(n_dot_v, roughness), 0.0).rg;
    float3 spec = prefiltered * (f_roughness * ab.x + ab.y);

    return (diffuse + spec) * env.z;
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

    // Depth 1 is the cleared far plane: nothing was drawn here, so this is
    // background — the environment seen directly along the view ray.
    if (depth >= 1.0) {
        float2 ndc_bg = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
        float4 far_h = mul(frame.inv_view_proj, float4(ndc_bg, 1.0, 1.0));
        float3 view_dir = normalize(far_h.xyz / far_h.w - frame.eye.xyz);
        return float4(linear_to_srgb(emission + env_background(view_dir, frame.env, env_source)), 1.0);
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
    //
    // Known artifact: on CLOSED geometry the front and back surfaces meet at
    // the silhouette with near-identical depth, so the depth test picks either
    // one, and a back face flipped here points away from the light and shades
    // dark. The result is a thin speckled rim exactly at the silhouette. Back
    // face culling would remove it, but Lumbre's content has no dependable
    // winding, so it would also drop real single-sided surfaces -- the wrong
    // trade for a one-pixel edge.
    if (dot(n, wo) < 0.0) {
        n = -n;
    }

    float3 color = emission;
    color += ibl(
        n, wo, albedo, roughness, metallic, specular,
        frame.env, env_irradiance, env_specular, env_brdf_lut);

    int count = int(frame.params.x);
    for (int li = 0; li < count; ++li) {
        float3 wi;
        float3 radiance;
        if (!light_sample(lights[li], world_pos, wi, radiance)) continue;

        float cos_surf = dot(n, wi);
        if (cos_surf <= 0.0) continue;

        // Only the sun is shadowed; see shadows.odin for why the other lights
        // are not.
        float visibility = 1.0;
        if (frame.params.y > 0.5 && int(lights[li].params.x) == LIGHT_DISTANT) {
            float view_depth = dot(world_pos - frame.eye.xyz, frame.forward.xyz);
            visibility = sun_visibility(
                world_pos, n, wi, view_depth,
                frame.cascade_view_proj, frame.cascade_splits, frame.cascade_texel,
                frame.params.z, frame.params.w,
                shadow_map, shadow_map_sampler);
            if (visibility <= 0.0) continue;
        }

        color += eval_brdf(albedo, roughness, metallic, specular, n, wo, wi)
               * radiance * cos_surf * visibility;
    }

    return float4(linear_to_srgb(color), 1.0);
}
