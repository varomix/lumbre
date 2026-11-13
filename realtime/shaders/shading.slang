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
// Must match MAX_SPOT_SHADOWS and SPOT_SHADOW_RESOLUTION in shadows.odin.
#define MAX_SPOT_SHADOWS 8
#define SPOT_SHADOW_RESOLUTION 1024.0
// Must match MAX_OMNI_SHADOWS, OMNI_SHADOW_RESOLUTION and OMNI_FACE_TAN there.
#define MAX_OMNI_SHADOWS 4
#define OMNI_SHADOW_RESOLUTION 512.0
#define OMNI_FACE_TAN (1.0 + 4.0 / OMNI_SHADOW_RESOLUTION)

// One analytic light. Area shapes are collapsed to a representative point
// carrying their area; see `light_sample` for what that costs.
struct Light {
    float4 position;  // xyz, w = radius
    float4 direction; // xyz aim or axis, w = height
    float4 emission;  // rgb radiance or intensity, w = area
    float4 params;    // kind, cos_inner, cos_outer, shadow slot or -1
    float4 half_u;    // quad only: half of each edge, from the centre
    float4 half_v;
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
    // One perspective map per shadowed spot light, by its params.w.
    float4x4 spot_view_proj[MAX_SPOT_SHADOWS];
    // World size of a texel per unit of distance from the light.
    float4 spot_texel[MAX_SPOT_SHADOWS / 4];
    // Six faces per shadowed point or sphere light: slot * 6 + face.
    float4x4 omni_view_proj[MAX_OMNI_SHADOWS * 6];
    // Baked irradiance probes; see probes.odin. min.w = 1 when in use.
    float4 probe_min;    // grid minimum, in use
    float4 probe_cell;   // cell size, depth moments' distance scale
    float4 probe_dims;   // probe counts, lookup offset off the surface
};

// The spot shadow layer a light owns, or -1.
int spot_shadow_layer(Light l) {
    return int(l.params.x) == 1 ? int(l.params.w) : -1;
}

// The omni shadow slot a point (0) or sphere (4) light owns, or -1.
int omni_shadow_slot(Light l) {
    int kind = int(l.params.x);
    return (kind == 0 || kind == 4) ? int(l.params.w) : -1;
}

// Which face of a light's six sees `to_point`, in the layer order of
// OMNI_FACES in shadows.odin: +X, -X, +Y, -Y, +Z, -Z.
int omni_face(float3 to_point) {
    float3 a = abs(to_point);
    if (a.x >= a.y && a.x >= a.z) return to_point.x > 0.0 ? 0 : 1;
    if (a.y >= a.z) return to_point.y > 0.0 ? 2 : 3;
    return to_point.z > 0.0 ? 4 : 5;
}

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

// Fraction of a spot light reaching `world_pos`, from its perspective map.
//
// The same normal-offset bias as the sun's, scaled to what a texel covers at
// this distance from the light -- a perspective texel grows with distance,
// where a cascade's is the same everywhere. No depth bias: perspective depth
// is far from linear, so no single constant suits both ends of the range, and
// the offset alone keeps lit surfaces off their own depth.
float spot_visibility(
    float3 world_pos, float3 n, float3 wi, float3 light_pos, int layer,
    float4x4 view_proj, float texel_scale, Sampler2DArrayShadow spot_tex
) {
    float cos_theta = saturate(dot(n, wi));
    float texel_world = length(light_pos - world_pos) * texel_scale;
    world_pos += n * (texel_world * 1.5 * (2.0 - cos_theta));

    float4 clip = mul(view_proj, float4(world_pos, 1.0));
    if (clip.w <= 0.0) return 1.0; // behind the light: its cone does not reach
    float3 proj = clip.xyz / clip.w;
    float2 uv = float2(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || proj.z > 1.0) return 1.0;

    float texel = 1.0 / SPOT_SHADOW_RESOLUTION;
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            sum += spot_tex.SampleCmpLevelZero(
                float3(uv + float2(x, y) * texel, float(layer)), proj.z);
        }
    }
    return sum / 9.0;
}

// Fraction of a point or sphere light reaching `world_pos`: the spot lookup
// on whichever of the light's six faces sees the point.
float omni_visibility(
    float3 world_pos, float3 n, float3 wi, float3 light_pos, int slot,
    float4x4 face_view_proj[MAX_OMNI_SHADOWS * 6], Sampler2DArrayShadow omni_tex
) {
    float cos_theta = saturate(dot(n, wi));
    float texel_world = length(light_pos - world_pos) * (2.0 * OMNI_FACE_TAN / OMNI_SHADOW_RESOLUTION);
    // Offset first, then pick the face, so the lookup and its face agree.
    world_pos += n * (texel_world * 1.5 * (2.0 - cos_theta));
    int layer = slot * 6 + omni_face(world_pos - light_pos);

    float4 clip = mul(face_view_proj[layer], float4(world_pos, 1.0));
    if (clip.w <= 0.0) return 1.0;
    float3 proj = clip.xyz / clip.w;
    float2 uv = float2(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (proj.z > 1.0) return 1.0;

    float texel = 1.0 / OMNI_SHADOW_RESOLUTION;
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            sum += omni_tex.SampleCmpLevelZero(
                float3(uv + float2(x, y) * texel, float(layer)), proj.z);
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

// ── area lights: linearly transformed cosines ────────────────────────────────
//
// See realtime/ltc/ltc.odin, whose CPU twins of these functions are what the
// tests check. Must match its SIZE and MIN_ROUGHNESS.

#define LTC_SIZE 64.0
#define LTC_MIN_ROUGHNESS 0.03
// Corners in a disc or sphere's polygon, and the radius scale that gives the
// polygon the circle's area: sqrt(2 pi / (N sin(2 pi / N))).
#define LTC_DISC_SIDES 12
#define LTC_DISC_SCALE 1.0233267
#define LTC_MAX_CORNERS 12

// One edge's vector term; theta / (2 pi sin theta) is Hill's rational fit.
float3 ltc_edge(float3 a, float3 b) {
    float x = dot(a, b);
    float y = abs(x);
    float fa = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float fb = 3.4175940 + (4.1616724 + y) * y;
    float v = fa / fb;
    float theta_sin = x > 0.0 ? v : 0.5 * rsqrt(max(1.0 - x * x, 1e-7)) - v;
    return cross(a, b) * theta_sin;
}

// The clamped cosine's integral over a polygon given in its own frame. Two-
// sided, as the path tracer's area lights are. No clipping: the unclipped
// vector form factor, corrected for the horizon as a spherical cap with the
// same vector would be, from the table's third channel. See
// polygon_form_factor in realtime/ltc/ltc.odin.
float ltc_polygon(float3 corners[LTC_MAX_CORNERS], int count, Sampler2D ltc_magnitude) {
    float3 first = normalize(corners[0]);
    float3 prev = first;
    float3 f = float3(0.0);
    [unroll]
    for (int i = 1; i < LTC_MAX_CORNERS; ++i) {
        if (i < count) {
            float3 next = normalize(corners[i]);
            f += ltc_edge(prev, next);
            prev = next;
        }
    }
    f += ltc_edge(prev, first);
    float len = length(f);
    if (len <= 0.0) return 0.0;
    if (dot(f, corners[0]) < 0.0) f = -f;
    float2 uv = (float2(f.z / len * 0.5 + 0.5, saturate(len)) * (LTC_SIZE - 1.0) + 0.5) / LTC_SIZE;
    return len * ltc_magnitude.SampleLevel(uv, 0.0).z;
}

// A light as a polygon around the shading point `p`: corners relative to p.
int area_polygon(Light l, float3 p, out float3 corners[LTC_MAX_CORNERS]) {
    int kind = int(l.params.x);
    [unroll]
    for (int i = 0; i < LTC_MAX_CORNERS; ++i) corners[i] = float3(0.0);
    float3 c = l.position.xyz - p;
    if (kind == LIGHT_QUAD) {
        float3 u = l.half_u.xyz;
        float3 v = l.half_v.xyz;
        corners[0] = c - u - v;
        corners[1] = c + u - v;
        corners[2] = c + u + v;
        corners[3] = c - u + v;
        return 4;
    }
    if (kind == LIGHT_CYLINDER) {
        // The rectangle along the axis, as wide as the cylinder, turned to
        // face the point. The caps do not emit.
        float3 axis = normalize(l.direction.xyz) * (l.direction.w * 0.5);
        float3 across = cross(axis, c);
        if (dot(across, across) < 1e-12) {
            across = cross(axis, abs(axis.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0));
        }
        across = normalize(across) * l.position.w;
        corners[0] = c - axis - across;
        corners[1] = c + axis - across;
        corners[2] = c + axis + across;
        corners[3] = c - axis + across;
        return 4;
    }
    // Disc and sphere: a regular polygon of the circle's area.
    float3 normal;
    float radius = l.position.w;
    if (kind == LIGHT_SPHERE) {
        // The disc facing the point that subtends the sphere's solid angle:
        // its edge at the sphere's tangent cone, tan = sin / cos.
        float dist = max(length(c), radius * 1.001);
        normal = c / max(length(c), 1e-9);
        radius = radius * dist / sqrt(dist * dist - radius * radius);
    } else {
        normal = normalize(l.direction.xyz);
    }
    float3 t = normalize(cross(abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0), normal));
    float3 b = cross(normal, t);
    radius *= LTC_DISC_SCALE;
    [unroll]
    for (int i = 0; i < LTC_DISC_SIDES; ++i) {
        float phi = float(i) * (2.0 * LUMBRE_PI / float(LTC_DISC_SIDES));
        corners[i] = c + (t * cos(phi) + b * sin(phi)) * radius;
    }
    return LTC_DISC_SIDES;
}

// Light reflected toward `wo` from an area light: diffuse from the polygon's
// form factor, specular from the fitted lobe. Emission is radiance.
float3 area_light(
    Light l, float3 p, float3 n, float3 wo,
    float3 albedo, float roughness, float metallic, float specular,
    Sampler2D ltc_transform, Sampler2D ltc_magnitude
) {
    float3 corners[LTC_MAX_CORNERS];
    int count = area_polygon(l, p, corners);

    // The lobe's frame: the view direction in its xz plane.
    float n_dot_v = saturate(dot(n, wo));
    float3 t1 = wo - n * dot(wo, n);
    t1 = dot(t1, t1) > 1e-10 ? normalize(t1)
       : normalize(cross(abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0), n));
    float3 t2 = cross(n, t1);
    float3 local[LTC_MAX_CORNERS];
    [unroll]
    for (int i = 0; i < LTC_MAX_CORNERS; ++i) {
        local[i] = float3(dot(corners[i], t1), dot(corners[i], t2), dot(corners[i], n));
    }

    float diffuse = ltc_polygon(local, count, ltc_magnitude);

    float2 coord = float2(clamp(roughness, LTC_MIN_ROUGHNESS, 1.0), sqrt(1.0 - n_dot_v));
    float2 uv = (coord * (LTC_SIZE - 1.0) + 0.5) / LTC_SIZE;
    float4 m = ltc_transform.SampleLevel(uv, 0.0);
    float2 mag = ltc_magnitude.SampleLevel(uv, 0.0).xy;
    // Rows of M^-1; see Entry in ltc.odin.
    float3 r0 = float3(m.x, 0.0, m.y);
    float3 r2 = float3(m.z, 0.0, m.w);
    float3 transformed[LTC_MAX_CORNERS];
    [unroll]
    for (int i = 0; i < LTC_MAX_CORNERS; ++i) {
        transformed[i] = float3(dot(r0, local[i]), local[i].y, dot(r2, local[i]));
    }
    float spec_lobe = ltc_polygon(transformed, count, ltc_magnitude);

    float3 f0 = lerp(float3(0.08 * specular), albedo, metallic);
    float3 spec = (f0 * (mag.x - mag.y) + mag.y) * spec_lobe;
    // The diffuse lobe gets what the specular one does not reflect at normal
    // incidence, as eval_brdf's (1 - F) does on average.
    float3 kd = (1.0 - metallic) * (float3(1.0) - f0);
    return l.emission.rgb * (albedo * kd * diffuse + spec);
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

// ── baked irradiance probes ──────────────────────────────────────────────────
// Layout and conventions are core/probes.odin's; the two must agree.

#define PROBE_TILE_RES 8
#define PROBE_TILE 10

float probe_sign(float v) { return v >= 0.0 ? 1.0 : -1.0; }

// Unit direction to [0,1]^2: `probe_oct_encode` in core/probes.odin.
float2 probe_oct_encode(float3 d) {
    float l1 = abs(d.x) + abs(d.y) + abs(d.z);
    float2 p = d.xy / l1;
    if (d.z < 0.0) {
        p = float2((1.0 - abs(p.y)) * probe_sign(p.x), (1.0 - abs(p.x)) * probe_sign(p.y));
    }
    return p * 0.5 + 0.5;
}

// The four texels around `dir` in a probe's bordered tile, and the bilinear
// weights between them. Storage buffers do not filter, so this does; the
// border texels make it seamless across the octahedron's folds.
struct ProbeTap {
    int i00, i10, i01, i11;
    float2 f;
};

ProbeTap probe_tap(int probe, float3 dir) {
    float2 t = probe_oct_encode(dir) * float(PROBE_TILE_RES) + 0.5;
    int2 i0 = clamp(int2(floor(t)), int2(0), int2(PROBE_TILE - 2));
    ProbeTap tap;
    tap.f = saturate(t - float2(i0));
    int base = probe * PROBE_TILE * PROBE_TILE + i0.y * PROBE_TILE + i0.x;
    tap.i00 = base;
    tap.i10 = base + 1;
    tap.i01 = base + PROBE_TILE;
    tap.i11 = base + PROBE_TILE + 1;
    return tap;
}

// The probe's cosine-weighted mean radiance about `n`.
float3 probe_tile_irradiance(StructuredBuffer<float4> irradiance, int probe, float3 n) {
    ProbeTap tap = probe_tap(probe, n);
    float3 a = lerp(irradiance[tap.i00].rgb, irradiance[tap.i10].rgb, tap.f.x);
    float3 b = lerp(irradiance[tap.i01].rgb, irradiance[tap.i11].rgb, tap.f.x);
    return lerp(a, b, tap.f.y);
}

// (mean, mean square) of distance / max distance along `dir`.
float2 probe_depth_moments(StructuredBuffer<float2> depth, int probe, float3 dir) {
    ProbeTap tap = probe_tap(probe, dir);
    float2 a = lerp(depth[tap.i00], depth[tap.i10], tap.f.x);
    float2 b = lerp(depth[tap.i01], depth[tap.i11], tap.f.x);
    return lerp(a, b, tap.f.y);
}

// Probe irradiance (as a cosine-weighted mean radiance) at surface point `p`
// facing `n`, seen from `wo`. DDGI's lookup: the eight surrounding probes,
// trilinear, each weighted by whether it faces the surface and by a Chebyshev
// test of its distance against the probe's depth moments. w = 0 when no probe
// could see the point; the caller then falls back to the environment.
float4 probe_irradiance(
    float3 p, float3 n, float3 wo, LightingUniforms frame,
    StructuredBuffer<float4> irradiance, StructuredBuffer<float2> depth
) {
    float3 lo = frame.probe_min.xyz;
    float3 cell = frame.probe_cell.xyz;
    float max_distance = frame.probe_cell.w;
    int3 dims = int3(frame.probe_dims.xyz + 0.5);

    // Look up from slightly off the surface, mostly toward the viewer, so the
    // depth test does not compare a surface with itself.
    float3 q = p + (n * 0.2 + wo * 0.8) * frame.probe_dims.w;
    // Grid space: probe i sits at i (probes are at cell centres).
    float3 g = clamp((q - lo) / cell - 0.5, float3(0.0), float3(dims - 1));
    int3 base = min(int3(floor(g)), max(dims - 2, int3(0)));
    float3 alpha = saturate(g - float3(base));

    float3 sum = float3(0.0);
    float weight_sum = 0.0;
    for (int i = 0; i < 8; ++i) {
        int3 off = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 idx = min(base + off, dims - 1);
        int probe = idx.x + dims.x * (idx.y + dims.y * idx.z);
        float3 probe_pos = lo + (float3(idx) + 0.5) * cell;

        float3 tri = lerp(1.0 - alpha, alpha, float3(off));
        float trilinear = tri.x * tri.y * tri.z;

        // Probes behind the surface count for little, but not nothing: in a
        // thin room every probe on one side may be behind something.
        float3 to_probe = probe_pos - p;
        float to_probe_len = length(to_probe);
        float3 dir = to_probe_len > 1e-6 ? to_probe / to_probe_len : n;
        float facing = (dot(dir, n) + 1.0) * 0.5;
        float w = facing * facing + 0.2;

        // Chebyshev: how likely the probe sees the lookup point, given the
        // distribution of what it hit in that direction.
        float3 from_probe = q - probe_pos;
        float r = length(from_probe);
        float2 m = probe_depth_moments(depth, probe, r > 1e-6 ? from_probe / r : n) * float2(max_distance, max_distance * max_distance);
        if (r > m.x) {
            float variance = abs(m.y - m.x * m.x);
            float d = r - m.x;
            float cheb = variance / (variance + d * d);
            w *= cheb * cheb * cheb;
        }

        // Crush tiny weights, so a probe that barely sees the point does not
        // tint it.
        w = max(w, 1e-6);
        const float crush = 0.2;
        if (w < crush) w *= w * w / (crush * crush);

        w *= trilinear;
        sum += w * probe_tile_irradiance(irradiance, probe, n);
        weight_sum += w;
    }
    return weight_sum > 1e-5 ? float4(sum / weight_sum, 1.0) : float4(0.0);
}

// The mean radiance arriving at `p` from all directions, trilinear between the
// eight surrounding probes' means: what fog scatters. No visibility test, so
// it leaks through walls thinner than a cell; fog is soft enough to bear it.
float3 probe_mean_radiance(float3 p, LightingUniforms frame, StructuredBuffer<float4> mean) {
    float3 lo = frame.probe_min.xyz;
    float3 cell = frame.probe_cell.xyz;
    int3 dims = int3(frame.probe_dims.xyz + 0.5);
    float3 g = clamp((p - lo) / cell - 0.5, float3(0.0), float3(dims - 1));
    int3 base = min(int3(floor(g)), max(dims - 2, int3(0)));
    float3 alpha = saturate(g - float3(base));
    float3 sum = float3(0.0);
    for (int i = 0; i < 8; ++i) {
        int3 off = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 idx = min(base + off, dims - 1);
        float3 tri = lerp(1.0 - alpha, alpha, float3(off));
        sum += tri.x * tri.y * tri.z * mean[idx.x + dims.x * (idx.y + dims.y * idx.z)].rgb;
    }
    return sum;
}

// The split-sum approximation: prefiltered radiance times the environment BRDF,
// with the diffuse irradiance taken from `probe` instead when probe.w is set.
// Probe irradiance is absolute -- the bake already applied the environment's
// intensity -- so it is not scaled by env.z, and it stands even with no
// environment at all.
float3 ibl_with_probes(
    float3 n, float3 wo, float3 albedo, float roughness, float metallic, float specular,
    float4 env, Sampler2D irradiance_map, Sampler2D specular_map, Sampler2D brdf_lut,
    float4 probe
) {
    bool use_probe = probe.w > 0.5;
    if (env.x < 0.5 && !use_probe) return float3(0.0);

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

    // The LUT is the BRDF integrated with Schlick's Fresnel already inside it:
    // a scale on F0 and a bias that IS the rise toward grazing. So it takes F0,
    // not `f_roughness` -- that adds the same rise a second time, and made a
    // dielectric floor reflect about 1.6 times what the path tracer gives it.
    float2 ab = brdf_lut.SampleLevel(float2(n_dot_v, roughness), 0.0).rg;
    float3 spec = prefiltered * (f0 * ab.x + ab.y);

    if (use_probe) {
        float3 env_spec = env.x < 0.5 ? float3(0.0) : spec * env.z;
        return probe.rgb * albedo * kd + env_spec;
    }
    return (diffuse + spec) * env.z;
}

// `ibl_with_probes` without probes.
float3 ibl(
    float3 n, float3 wo, float3 albedo, float roughness, float metallic, float specular,
    float4 env, Sampler2D irradiance_map, Sampler2D specular_map, Sampler2D brdf_lut
) {
    return ibl_with_probes(n, wo, albedo, roughness, metallic, specular,
        env, irradiance_map, specular_map, brdf_lut, float4(0.0));
}

#endif // LUMBRE_SHADING_INCLUDED
