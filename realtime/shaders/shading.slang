#ifndef LUMBRE_SHADING_INCLUDED
#define LUMBRE_SHADING_INCLUDED

// Shading shared by the deferred pass and the forward (transparent) pass.
//
// Everything here takes its resources as parameters rather than reading a
// binding, so the two passes can bind them at whatever slots their own layouts
// require. The BRDF mirrors the path tracer's Principled BSDF restricted to the
// parameters a G-buffer can carry -- albedo, roughness, metallic, specular --
// so it reduces exactly to `principled_eval_f` in shaders/raytrace.metal when
// spec_tint is white and sheen, clearcoat and spec_trans are zero.

#include "common.slang"

#define CASCADE_COUNT 4

// One analytic light. Area shapes are collapsed to a representative point
// carrying their area; see `light_sample` for what that costs.
struct Light {
    float4 position;  // xyz, w = radius
    float4 direction; // xyz aim or axis, w = height
    float4 emission;  // rgb radiance or intensity, w = area
    float4 params;    // kind, cos_inner, cos_outer, unused
};

// Mirrors `Lighting_Uniforms` in camera.odin field for field.
struct LightingUniforms {
    float4x4 inv_view_proj;
    float4x4 cascade_view_proj[CASCADE_COUNT];
    float4 eye;          // xyz, w = shadow-casting light index
    float4 forward;      // camera view axis, for picking a cascade by depth
    float4 cascade_splits; // view-space far distance of each cascade
    float4 cascade_texel;  // world-space texel size of each cascade
    float4 params;       // light_count, shadows_enabled, depth_bias, texel_size
    // has_env, rotation (radians), intensity, specular mip count.
    float4 env;
    // Froxel grid: tiles across, tiles down, depth slices, max lights per cell.
    float4 cluster_dims;
    // near, far, the log scale that maps a view depth to a slice, unused.
    float4 cluster_depth;
    // Forward transparency only: projects a refracted point to screen space.
    float4x4 view_proj;
    // refraction thickness in world units, unused, unused, unused.
    float4 forward_params;
};

// Clustered light assignment (see clusters.odin). `cluster_ranges` is one entry
// per froxel; `cluster_indices` is the concatenated per-froxel light lists, in
// ascending light order, so a pixel accumulates exactly as an unclustered loop
// would.
struct ClusterRange {
    uint offset;
    uint count;
};

// Which froxel a pixel falls in. Depth slices are logarithmic, so they stay
// thin near the camera where perspective packs the most geometry.
int cluster_of(float2 uv, float view_depth, float4 dims, float4 depth_params) {
    int tx = clamp(int(uv.x * dims.x), 0, int(dims.x) - 1);
    int ty = clamp(int(uv.y * dims.y), 0, int(dims.y) - 1);
    float z = max(view_depth, depth_params.x);
    int tz = clamp(int(log(z / depth_params.x) * depth_params.z), 0, int(dims.z) - 1);
    return (tz * int(dims.y) + ty) * int(dims.x) + tx;
}

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
    Sampler2DArrayShadow shadow_tex
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
                float3(uv + offset, float(cascade)), compare_depth);
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

#endif // LUMBRE_SHADING_INCLUDED
