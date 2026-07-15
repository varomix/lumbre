#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant int DIRECT_LIGHT_SAMPLES = 4;
constant float PI = 3.14159265358979323846;
constant float INV_PI = 0.31830988618379067154;
constant int GI_CACHE_MAX_POINTS = 262144;
constant int GI_GRID_SIZE = 32768; // power of 2
constant int GI_MAX_PER_CELL = 16;
constant int GI_CACHE_MIN_SAMPLES = 4;
constant float GI_CACHE_MAX_REL_STDDEV = 0.75;

constant int PHOTON_MAX_COUNT = 1048576;
constant int PHOTON_GRID_SIZE = 16384;
constant int PHOTON_MAX_BOUNCES = 8;
constant float PHOTON_SEARCH_RADIUS = 1.0;
// Minimum photons inside the search radius before the density estimate is
// trusted enough to replace brute-force indirect. Below this, fall back.
constant int PHOTON_MIN_FOUND = 8;

// ── GPU data types ───────────────────────────────────────────────────────────

struct GPUMaterial {
	float4 albedo;       // rgb = base color
	float4 emission;     // rgb = emission
	float4 params0;      // x=kind, y=fuzz, z=ir, w=roughness
	float4 params1;      // x=metallic, y=emission_strength, z=specular, w=clearcoat
	float4 params2;      // x=clearcoat_roughness, y=sheen, z=normal_scale, w=anisotropic
	float4 spec_tint;    // rgb = specular_tint
	float4 sheen_tint;   // rgb = sheen_tint
	// Each *_info is {pixel_offset, width, height, has_tex} into `tex_pixels`.
	float4 tex_info;     // base color (sRGB)
	float4 mr_info;      // metallic-roughness: G = rough, B = metal (linear)
	float4 nrm_info;     // tangent-space normal map (linear)
	float4 emis_info;    // emissive (sRGB)
	float4 params3;      // x=spec_trans, yzw=unused
	float4 params4;      // rgb=SSS albedo, w=SSS weight
	float4 params5;      // rgb=SSS mean free path
};

struct GPUSceneData {
	float4 origin;
	float4 lower_left;
	float4 horizontal;
	float4 vertical;
	float4 u;
	float4 v;
	float  lens_radius;
	int    image_width;
	int    image_height;
	int    samples_per_pixel;
	int    max_depth;
	float  max_radiance;
	int    debug_mode;
	int    tri_light_count;
	int    primitive_count;
	uint   seed;
	int    quad_light_count;
	int    sphere_light_count;
	float  roughness_cutoff;
	float  glossy_bias;
	int    gi_cache_enabled;
	float  gi_cache_distance;
	float  gi_cache_normal_angle;
	int    gi_cache_num_points;
	int    photon_enabled;
	int    photon_count;
	float  photon_radius;
	int    photon_max_bounces;
	int    disc_light_count;
	int    cylinder_light_count;
	int    punctual_light_count;
	// HDRI environment (dome light).
	int    has_env;
	int    env_width;
	int    env_height;
	float  env_rotation;
	float  env_intensity;
	float  env_func_int;
	int    hide_default_sky;
};

struct GICachePoint {
	float4 position;   // xyz = world position, w = grid cell x
	float4 normal;     // xyz = shading normal, w = grid cell y
	float4 irradiance; // xyz = cached irradiance, w = grid cell z
};

struct Photon {
	float4 position; // xyz = world position, w = grid cell x
	float4 incident; // xyz = incident direction, w = grid cell y
	float4 power;    // xyz = photon power, w = grid cell z
};

struct TriVertex {
	float4 position;
	float4 normal;
	float4 uv;       // x, y, has_uv (0/1), unused
};

struct GPULightTriangle {
	float4 p0;
	float4 p1;
	float4 p2;
	float4 emission;
};

struct GPUQuadLight {
	float4 position;
	float4 u;
	float4 v;
	float4 emission;
};

struct GPUSphereLight {
	float4 position;
	float4 emission;
	float  radius;
	float  _pad[3];
};

struct GPUDiscLight {
	float4 position; // xyz = center, w = radius
	float4 normal;   // xyz = disc normal
	float4 emission;
};

struct GPUCylinderLight {
	float4 position; // xyz = base center, w = radius
	float4 axis;     // xyz = axis (normalized), w = height
	float4 emission;
};

// point / spot / distant. params = (kind, cos_inner, cos_outer, angular_radius).
// kind: 0 = point, 1 = spot, 2 = distant.
struct GPUPunctualLight {
	float4 position;
	float4 direction;
	float4 emission;
	float4 params;
};

// ── GPU RNG (PCG-style) ─────────────────────────────────────────────────────

static uint rng_next(thread uint& state) {
	state = state * 747796405u + 2891336453u;
	uint w = state ^ (state >> 22);
	w = w * 1664525u + 1013904223u;
	return w;
}

static float rng_float(thread uint& state) {
	return float(rng_next(state) & 0x00FFFFFFu) / float(0x01000000u);
}

static float rng_float_range(thread uint& state, float lo, float hi) {
	return lo + (hi - lo) * rng_float(state);
}

static float3 rng_in_unit_sphere(thread uint& state) {
	for (;;) {
		float3 p = float3(rng_float_range(state, -1.0, 1.0),
		                  rng_float_range(state, -1.0, 1.0),
		                  rng_float_range(state, -1.0, 1.0));
		if (length_squared(p) < 1.0) return p;
	}
}

static float3 rng_in_unit_disk(thread uint& state) {
	for (;;) {
		float3 p = float3(rng_float_range(state, -1.0, 1.0),
		                  rng_float_range(state, -1.0, 1.0), 0.0);
		if (length_squared(p) < 1.0) return p;
	}
}

static float3 rng_unit_vector(thread uint& state) {
	return normalize(rng_in_unit_sphere(state));
}

// ── Utility functions ───────────────────────────────────────────────────────

static bool near_zero(float3 v) {
	float eps = 1.0e-8;
	return fabs(v.x) < eps && fabs(v.y) < eps && fabs(v.z) < eps;
}

static float schlick_reflectance(float cosine, float ref_idx) {
	float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
	r0 *= r0;
	return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

static float luminance(float3 c) {
	return c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
}

static float3 debug_heat(float v) {
	float x = clamp(v, 0.0, 1.0);
	return clamp(float3(2.0 * x, 2.0 * (1.0 - fabs(x - 0.5) * 2.0), 2.0 * (1.0 - x)) - 0.5, 0.0, 1.0);
}

static float power_heuristic(float pdf_a, float pdf_b) {
	float a2 = pdf_a * pdf_a;
	float b2 = pdf_b * pdf_b;
	return a2 / max(a2 + b2, 1.0e-12);
}

static float3 make_tangent(float3 n) {
	float3 helper = fabs(n.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
	return normalize(cross(helper, n));
}

// ── Principled BSDF (Disney-style: GGX specular + Fresnel + diffuse) ───────

// `roughness` is the user-facing value in [0,1]. We square it to get alpha^2
// for the GGX distribution — the "perceptual roughness" mapping.
static float pbr_alpha_sq(float roughness) {
	float a = clamp(roughness, 0.0, 1.0);
	return a * a;
}

// Trowbridge-Reitz / GGX normal distribution.
static float pbr_D(float cos_h, float alpha_sq) {
	if (cos_h <= 0.0 || alpha_sq <= 0.0) return 0.0;
	float denom = cos_h * cos_h * (alpha_sq - 1.0) + 1.0;
	return alpha_sq / max(PI * denom * denom, 1.0e-30);
}

// Smith G1 term for a single direction.
static float pbr_G1(float cos_v, float alpha_sq) {
	if (cos_v <= 0.0) return 0.0;
	float cos2 = cos_v * cos_v;
	float tan2 = (1.0 - cos2) / max(cos2, 1.0e-30);
	return 2.0 / (1.0 + sqrt(1.0 + alpha_sq * tan2));
}

// Smith G2 = G1(wo) * G1(wi).
static float pbr_G(float cos_o, float cos_i, float alpha_sq) {
	return pbr_G1(cos_o, alpha_sq) * pbr_G1(cos_i, alpha_sq);
}

// ── Anisotropic GGX (Phase B) — mirror of principled.odin ──────────────────
// Tangent-local coords: x=tangent, y=bitangent, z=normal. ax==ay reduces to
// the isotropic forms above (the isotropic path is only entered when
// anisotropic==0, keeping it byte-identical).

static void pbr_aniso_alphas(float roughness, float anisotropic, thread float& ax, thread float& ay) {
	float a = clamp(roughness, 0.0, 1.0);
	float aniso = clamp(anisotropic, 0.0, 1.0);
	float aspect = sqrt(1.0 - 0.9 * aniso);
	ax = max(a / aspect, 1.0e-4);
	ay = max(a * aspect, 1.0e-4);
}

static float pbr_D_aniso(float3 h, float ax, float ay) {
	if (h.z <= 0.0) return 0.0;
	float d = (h.x * h.x) / (ax * ax) + (h.y * h.y) / (ay * ay) + h.z * h.z;
	return 1.0 / max(PI * ax * ay * d * d, 1.0e-30);
}

static float pbr_lambda_aniso(float3 w, float ax, float ay) {
	if (w.z <= 0.0) return 0.0;
	float t2 = (ax * ax * w.x * w.x + ay * ay * w.y * w.y) / (w.z * w.z);
	return 0.5 * (-1.0 + sqrt(1.0 + t2));
}

static float pbr_G1_aniso(float3 w, float ax, float ay) {
	return 1.0 / (1.0 + pbr_lambda_aniso(w, ax, ay));
}

static float pbr_G_aniso(float3 wo, float3 wi, float ax, float ay) {
	return pbr_G1_aniso(wo, ax, ay) * pbr_G1_aniso(wi, ax, ay);
}

// Heitz 2018 anisotropic VNDF sample; `wo_local` in tangent frame, returns the
// tangent-local half-vector.
static float3 sample_ggx_vndf_aniso_local(float3 wo_local, float ax, float ay, float u1, float u2) {
	float3 vh = normalize(float3(ax * wo_local.x, ay * wo_local.y, wo_local.z));
	float lensq = vh.x * vh.x + vh.y * vh.y;
	float3 t1 = (lensq > 1.0e-12) ? float3(-vh.y, vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
	float3 t2 = cross(vh, t1);
	float r = sqrt(u1);
	float phi = 2.0 * PI * u2;
	float p1 = r * cos(phi);
	float p2 = r * sin(phi);
	float s = 0.5 * (1.0 + vh.z);
	p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;
	float3 nh = p1 * t1 + p2 * t2 + sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0)) * vh;
	return normalize(float3(ax * nh.x, ay * nh.y, max(nh.z, 1.0e-6)));
}

// Orthonormalized surface tangent (fallback when `t` is unusable).
static float3 surface_tangent(float3 n, float3 t) {
	float3 proj = t - n * dot(n, t);
	if (length(proj) < 1.0e-6) return make_tangent(n);
	return normalize(proj);
}

// Main specular D, G, and VNDF pdf factor; anisotropic in the (t,b,n) frame
// when anisotropic>0, else isotropic (byte-identical to Phase A).
static void principled_specular_dg(
	GPUMaterial mat, float3 wo, float3 wi, float3 wh, float3 n, float3 t,
	float cos_o, float cos_i, float cos_h,
	thread float& D, thread float& G, thread float& pdf_s
) {
	if (mat.params2.w <= 0.0) {
		float alpha_sq = pbr_alpha_sq(mat.params0.w);
		D = pbr_D(cos_h, alpha_sq);
		G = pbr_G(cos_o, cos_i, alpha_sq);
		pdf_s = pbr_G1(cos_o, alpha_sq) * D / (4.0 * cos_o);
		return;
	}
	float3 b = cross(n, t);
	float ax, ay;
	pbr_aniso_alphas(mat.params0.w, mat.params2.w, ax, ay);
	float3 wo_l = float3(dot(wo, t), dot(wo, b), cos_o);
	float3 wi_l = float3(dot(wi, t), dot(wi, b), cos_i);
	float3 wh_l = float3(dot(wh, t), dot(wh, b), cos_h);
	D = pbr_D_aniso(wh_l, ax, ay);
	G = pbr_G_aniso(wo_l, wi_l, ax, ay);
	pdf_s = pbr_G1_aniso(wo_l, ax, ay) * D / (4.0 * cos_o);
}

// Schlick Fresnel. Returns the spectral Fresnel curve for an arbitrary F0.
static float3 schlick_fresnel_color(float cos_theta, float3 f0) {
	float ct = clamp(cos_theta, 0.0, 1.0);
	float p = 1.0 - ct;
	float p5 = p * p;
	p5 = p5 * p5 * p;
	return f0 + (float3(1.0) - f0) * p5;
}

// Dielectric F0 from a "specular" parameter in [0,1] (0.5 → 0.08, 1.0 → 0.16).
static float dielectric_f0(float specular) {
	return 0.08 * specular;
}

// Build the spectral F0 used by Schlick. Dielectrics tint F0 with
// spec_tint; metals use the base color (modulated by spec_tint).
static float3 principled_f0(GPUMaterial mat) {
	float3 dielectric_part = dielectric_f0(mat.params1.z) * mat.spec_tint.xyz;
	if (mat.params1.x >= 1.0) {
		return mat.albedo.xyz;
	}
	return mix(dielectric_part, mat.albedo.xyz, mat.params1.x);
}

// Heitz 2017 VNDF importance sampling for isotropic GGX. `wo_local` is in
// tangent space (z = surface normal). Returns the half-vector in tangent space.
static float3 sample_ggx_vndf_local(float3 wo_local, float alpha, float u1, float u2) {
	float3 v = float3(wo_local.x * alpha, wo_local.y * alpha, wo_local.z);
	if (length(v) < 1.0e-12) v = float3(0.0, 0.0, 1.0);
	v = normalize(v);

	float3 t1 = fabs(v.z) < 0.9999
		? normalize(cross(v, float3(0.0, 1.0, 0.0)))
		: float3(1.0, 0.0, 0.0);
	float3 t2 = cross(t1, v);

	float a = 1.0 / (1.0 + v.z);
	float r = sqrt(u1);
	float phi = (u2 < a) ? (u2 / a) * PI : PI + (u2 - a) / (1.0 - a) * PI;
	float s1 = r * cos(phi);
	float s2 = r * sin(phi) * sqrt(max(1.0 - s1 * s1, 0.0));

	float3 n_h = s1 * t1 + s2 * t2 + sqrt(max(1.0 - s1 * s1 - s2 * s2, 0.0)) * v;
	return normalize(float3(n_h.x * alpha, n_h.y * alpha, n_h.z));
}

// Reflect an *incident* direction (pointing toward the surface) about `wh`.
// To mirror a direction that points away from the surface, negate it first.
static float3 reflect_over(float3 incident, float3 wh) {
	return incident - 2.0 * dot(incident, wh) * wh;
}

// ── Disney lobes: Burley diffuse + sheen + GGX specular + clearcoat ─────────
// Mirror of principled.odin; keep the two in lockstep (plans/PRINCIPLED_BSDF.md).

static float pbr_pow5(float x) {
	float x2 = x * x;
	return x2 * x2 * x;
}

// Schlick Fresnel for a scalar F0 (clearcoat uses F0 = 0.04).
static float schlick_scalar(float cos_theta, float f0) {
	return f0 + (1.0 - f0) * pbr_pow5(1.0 - clamp(cos_theta, 0.0, 1.0));
}

// Clearcoat's own GGX alpha^2 (roughness floored so it stays samplable).
static float clearcoat_alpha_sq(GPUMaterial mat) {
	float r = clamp(mat.params2.x, 0.03, 1.0);
	return r * r;
}

// Relative selection weights of the three reflection lobes. With clearcoat = 0
// these reduce to the old (1-metallic) vs f0_lum split.
// A pure-diffuse Principled surface — what normalize_material produces for a
// Lambertian, and the case the GI cache/photon map are tuned for. Mirrors
// material_is_lambertian_like in principled.odin.
static bool material_is_lambertian_like(GPUMaterial mat) {
	return int(mat.params0.x) == 3 &&
		mat.params1.x < 0.01 &&  // metallic
		mat.params3.x < 0.01 &&  // spec_trans
		mat.params1.z < 0.01 &&  // specular
		mat.params1.w < 0.01 &&  // clearcoat
		mat.params2.y < 0.01 &&  // sheen
		mat.params0.w > 0.99;    // roughness
}

static void principled_lobe_weights(GPUMaterial mat, thread float& w_diff, thread float& w_spec, thread float& w_cc) {
	float3 f0 = principled_f0(mat);
	float metallic = clamp(mat.params1.x, 0.0, 1.0);
	float spec_trans = clamp(mat.params3.x, 0.0, 1.0);
	w_diff = (1.0 - metallic) * (1.0 - spec_trans);
	w_spec = luminance(f0);
	w_cc = 0.25 * max(mat.params1.w, 0.0);
}

// Full BSDF value for a reflection pair (both vectors above the surface).
static float3 principled_eval_f(GPUMaterial mat, float3 wo, float3 wi, float3 n, float3 t) {
	float cos_o = dot(wo, n);
	float cos_i = dot(wi, n);
	if (cos_o <= 0.0 || cos_i <= 0.0) return float3(0.0);
	float3 wh = normalize(wo + wi);
	if (length(wh) < 1.0e-12) return float3(0.0);
	float cos_h = dot(wh, n);
	float cos_d = max(dot(wo, wh), 0.0);
	if (cos_h <= 0.0 || cos_d <= 0.0) return float3(0.0);
	float metallic = clamp(mat.params1.x, 0.0, 1.0);
	float roughness = mat.params0.w;

	// Specular (GGX + Schlick metallic Fresnel), anisotropy-aware.
	float3 tan = surface_tangent(n, t);
	float D, G, pdf_s_unused;
	principled_specular_dg(mat, wo, wi, wh, n, tan, cos_o, cos_i, cos_h, D, G, pdf_s_unused);
	float3 f0 = principled_f0(mat);
	float3 F = schlick_fresnel_color(cos_d, f0);
	float3 specular = F * (D * G) / (4.0 * cos_o * cos_i);

	// Burley diffuse, weighted by (1 - F) so diffuse+specular split the
	// energy at grazing (Frostbite-style; see principled.odin for the note).
	float3 diffuse = float3(0.0);
	float3 sheen = float3(0.0);
	if (metallic < 1.0) {
		float fd90 = 0.5 + 2.0 * roughness * cos_d * cos_d;
		float fl = 1.0 + (fd90 - 1.0) * pbr_pow5(1.0 - cos_i);
		float fv = 1.0 + (fd90 - 1.0) * pbr_pow5(1.0 - cos_o);
		float trans_w = 1.0 - clamp(mat.params3.x, 0.0, 1.0); // (1 - spec_trans): glass has no diffuse
		diffuse = (1.0 - metallic) * trans_w * mat.albedo.xyz * INV_PI * fl * fv * (float3(1.0) - F);

		float sheen_amt = mat.params2.y;
		if (sheen_amt > 0.0) {
			sheen = (1.0 - metallic) * sheen_amt * mat.sheen_tint.xyz * pbr_pow5(1.0 - cos_d);
		}
	}

	float3 base = diffuse + sheen + specular;

	// Clearcoat coat over the base; attenuate the base by the coat's
	// reflectance rather than adding on top.
	if (mat.params1.w > 0.0) {
		float acc = clearcoat_alpha_sq(mat);
		float dcc = pbr_D(cos_h, acc);
		float gcc = pbr_G(cos_o, cos_i, acc);
		float fcc = schlick_scalar(cos_d, 0.04);
		float cc = 0.25 * mat.params1.w * dcc * fcc * gcc / (4.0 * cos_o * cos_i);
		float atten = 1.0 - mat.params1.w * fcc;
		return base * atten + float3(cc);
	}

	return base;
}

// Combined solid-angle sampling pdf at `wi`. `t` is the surface tangent.
static float principled_pdf(GPUMaterial mat, float3 wo, float3 wi, float3 n, float3 t) {
	float cos_o = dot(wo, n);
	float cos_i = dot(wi, n);
	if (cos_o <= 0.0 || cos_i <= 0.0) return 0.0;
	float3 wh = normalize(wo + wi);
	if (length(wh) < 1.0e-12) return 0.0;
	float cos_h = dot(wh, n);
	float cos_d = max(dot(wo, wh), 0.0);
	if (cos_h <= 0.0 || cos_d <= 0.0) return 0.0;

	float w_diff, w_spec, w_cc;
	principled_lobe_weights(mat, w_diff, w_spec, w_cc);
	float total = w_diff + w_spec + w_cc;
	if (total <= 0.0) return 0.0;

	float3 tan = surface_tangent(n, t);
	float D_unused, G_unused, pdf_s;
	principled_specular_dg(mat, wo, wi, wh, n, tan, cos_o, cos_i, cos_h, D_unused, G_unused, pdf_s);
	float pdf_d = cos_i * INV_PI;
	float pdf_cc = 0.0;
	if (w_cc > 0.0) {
		float acc = clearcoat_alpha_sq(mat);
		pdf_cc = pbr_G1(cos_o, acc) * pbr_D(cos_h, acc) / (4.0 * cos_o);
	}
	return (w_diff * pdf_d + w_spec * pdf_s + w_cc * pdf_cc) / total;
}

// ── Specular transmission (glass) — Phase C ─────────────────────────────────
// Microfacet dielectric (Walter 2007), mirror of principled.odin. Returns the
// outgoing direction (below the surface for a transmission event) and the
// path throughput (tint * G2/G1). ok=false on a masked/degenerate sample. At
// roughness 0 this reduces to the perfect glass of the legacy Dielectric kind.
//
// Treat an almost-zero perceptual roughness as the delta limit.  Re-sampling a
// GGX normal at every interface of a thick/concave glass object otherwise
// compounds sub-pixel roughness into visible stochastic "frost".
constant float GLASS_DELTA_ROUGHNESS = 0.02;
static bool principled_sample_glass(
	GPUMaterial mat, float3 wo, float3 n, bool front_face,
	thread uint& seed,
	thread float3& wi, thread float3& throughput
) {
	float cos_o = dot(wo, n);
	if (cos_o <= 0.0) return false;
	float eta = front_face ? (1.0 / mat.params0.z) : mat.params0.z;
	float alpha_sq = pbr_alpha_sq(max(mat.params0.w, 0.001));
	float alpha = sqrt(alpha_sq);

	// Sample an isotropic GGX microfacet normal about n.
	float3 wh = n;
	if (mat.params0.w > GLASS_DELTA_ROUGHNESS) {
		float3 tangent = make_tangent(n);
		float3 bitangent = cross(n, tangent);
		float u1 = rng_float(seed);
		float u2 = rng_float(seed);
		float3 wo_local = float3(dot(wo, tangent), dot(wo, bitangent), cos_o);
		float3 wh_local = sample_ggx_vndf_local(wo_local, alpha, u1, u2);
		wh = normalize(wh_local.x * tangent + wh_local.y * bitangent + wh_local.z * n);
	}

	float cos_ow = dot(wo, wh);
	if (cos_ow <= 0.0) return false;

	float3 incident = -wo;
	float sin_ow = sqrt(max(1.0 - cos_ow * cos_ow, 0.0));
	bool cannot_refract = eta * sin_ow > 1.0;
	float3 tint;
	if (cannot_refract || schlick_reflectance(cos_ow, eta) > rng_float(seed)) {
		wi = reflect(incident, wh);
		tint = float3(1.0);
		if (dot(wi, n) <= 0.0) return false;
	} else {
		wi = refract(incident, wh, eta);
		tint = mat.albedo.xyz;
		if (dot(wi, n) >= 0.0) return false; // failed to cross the surface
	}

	float cos_i_abs = fabs(dot(wi, n));
	float g2 = pbr_G(cos_o, cos_i_abs, alpha_sq);
	float g1 = pbr_G1(cos_o, alpha_sq);
	float gratio = (g1 > 0.0) ? g2 / g1 : 0.0;
	throughput = tint * gratio;
	return true;
}


// NEE direct-light contribution for the current material kind.
// `light_dir` points from the surface toward the light. `cos_surf` is
// `max(n . light_dir, 0)`. `light_pdf` is the solid-angle light sampling
// PDF. `emission` is the light radiance.
static float3 nee_contribution(
	GPUMaterial mat,
	int mat_kind_eff,
	float3 wo, float3 n, float3 t,
	float3 light_dir,
	float cos_surf,
	float light_pdf,
	float3 emission
) {
	if (cos_surf <= 0.0 || light_pdf <= 0.0) return float3(0.0);
	float bsdf_pdf;
	float3 brdf;
	if (mat_kind_eff == 3) {
		brdf = principled_eval_f(mat, wo, light_dir, n, t);
		bsdf_pdf = principled_pdf(mat, wo, light_dir, n, t);
	} else {
		// Lambertian
		bsdf_pdf = cos_surf * INV_PI;
		brdf = mat.albedo.xyz * INV_PI;
	}
	if (bsdf_pdf <= 0.0) return float3(0.0);
	float mis_weight = power_heuristic(light_pdf, bsdf_pdf);
	return emission * brdf * cos_surf * mis_weight / light_pdf;
}

// Delta-light (point/spot/distant) contribution: the direction is sampled with
// certainty, so there is no MIS weight and no division by a pdf. `radiance` is
// the incident radiance at the surface (already including any falloff).
static float3 nee_contribution_delta(
	GPUMaterial mat,
	int mat_kind_eff,
	float3 wo, float3 n, float3 t,
	float3 light_dir,
	float cos_surf,
	float3 radiance
) {
	if (cos_surf <= 0.0) return float3(0.0);
	float3 brdf;
	if (mat_kind_eff == 3) {
		brdf = principled_eval_f(mat, wo, light_dir, n, t);
	} else {
		brdf = mat.albedo.xyz * INV_PI;
	}
	return radiance * brdf * cos_surf;
}

// Sample a Principled BSDF direction. Returns (wi, f, pdf). `t` is the
// surface tangent (only used when the material is anisotropic).
static void principled_sample(
	GPUMaterial mat,
	float3 wo, float3 n, float3 t,
	thread uint& seed,
	thread float3& wi,
	thread float3& f,
	thread float& pdf
) {
	float cos_o = dot(wo, n);
	if (cos_o <= 0.0) {
		wi = float3(0.0, 0.0, 1.0);
		f = float3(0.0);
		pdf = 0.0;
		return;
	}

	float w_diff, w_spec, w_cc;
	principled_lobe_weights(mat, w_diff, w_spec, w_cc);
	float total = w_diff + w_spec + w_cc;
	if (total <= 0.0) {
		wi = float3(0.0, 0.0, 1.0);
		f = float3(0.0);
		pdf = 0.0;
		return;
	}

	float3 tangent = make_tangent(n);
	float3 bitangent = cross(n, tangent);

	float pick = rng_float(seed) * total;
	if (pick < w_diff) {
		// Cosine-weighted diffuse hemisphere sample.
		float u1 = rng_float(seed);
		float u2 = rng_float(seed);
		float r = sqrt(u1);
		float phi = 2.0 * PI * u2;
		float x = r * cos(phi);
		float y = r * sin(phi);
		float z = sqrt(max(1.0 - u1, 0.0));
		wi = normalize(x * tangent + y * bitangent + z * n);
	} else if (pick < w_diff + w_spec && mat.params2.w > 0.0) {
		// Anisotropic main specular lobe, sampled in the surface tangent frame.
		float3 st = surface_tangent(n, t);
		float3 sb = cross(n, st);
		float ax, ay;
		pbr_aniso_alphas(mat.params0.w, mat.params2.w, ax, ay);
		float u1 = rng_float(seed);
		float u2 = rng_float(seed);
		float3 wo_local = float3(dot(wo, st), dot(wo, sb), cos_o);
		float3 wh_local = sample_ggx_vndf_aniso_local(wo_local, ax, ay, u1, u2);
		float3 wh_world = normalize(wh_local.x * st + wh_local.y * sb + wh_local.z * n);
		wi = reflect_over(-wo, wh_world);
	} else {
		// Isotropic GGX reflection: main specular (isotropic) or clearcoat.
		float alpha = (pick < w_diff + w_spec)
			? sqrt(max(pbr_alpha_sq(mat.params0.w), 0.0))
			: sqrt(max(clearcoat_alpha_sq(mat), 0.0));
		float u1 = rng_float(seed);
		float u2 = rng_float(seed);
		float3 wo_local = float3(dot(wo, tangent), dot(wo, bitangent), cos_o);
		float3 wh_local = sample_ggx_vndf_local(wo_local, alpha, u1, u2);
		float3 wh_world = normalize(wh_local.x * tangent + wh_local.y * bitangent + wh_local.z * n);
		wi = reflect_over(-wo, wh_world);
	}

	if (dot(wi, n) <= 0.0) {
		wi = float3(0.0, 0.0, 1.0);
		f = float3(0.0);
		pdf = 0.0;
		return;
	}
	f = principled_eval_f(mat, wo, wi, n, t);
	pdf = principled_pdf(mat, wo, wi, n, t);
}

static float3 cosine_sample_hemisphere(float3 n, thread uint& state, thread float& pdf) {
	float r1 = rng_float(state);
	float r2 = rng_float(state);
	float phi = 2.0 * PI * r1;
	float radius = sqrt(r2);
	float x = radius * cos(phi);
	float y = radius * sin(phi);
	float z = sqrt(max(1.0 - r2, 0.0));

	float3 tangent = make_tangent(n);
	float3 bitangent = cross(n, tangent);
	float3 dir = normalize(x * tangent + y * bitangent + z * n);
	pdf = max(dot(n, dir), 0.0) * INV_PI;
	return dir;
}

// Trace the volumetric part of a MaterialX standard_surface subsurface lobe.
// The camera path enters through the current front-facing surface, samples
// exponential free flights, and resumes outside at the first boundary it
// reaches.  This is a stochastic random walk rather than the old local
// diffuse substitute, so the radius changes where light exits the object.
static bool subsurface_random_walk(
	GPUMaterial mat,
	float3 entry_point,
	float3 entry_normal,
	primitive_acceleration_structure accel,
	thread uint& seed,
	thread ray& exit_ray,
	thread float3& throughput
) {
	float3 radius = max(mat.params5.xyz, float3(1.0e-5));
	float mean_free_path = max(luminance(radius), 1.0e-5);
	float3 albedo = clamp(mat.params4.xyz, 0.0, 0.999);

	float pdf;
	exit_ray.origin = entry_point - entry_normal * 0.001;
	exit_ray.direction = cosine_sample_hemisphere(-entry_normal, seed, pdf);
	exit_ray.min_distance = 0.001;
	exit_ray.max_distance = INFINITY;
	throughput = 1.0;

	// A compact cap keeps pathological thin meshes from trapping paths while
	// still allowing several real volume events for high-albedo wax/skin.
	for (int bounce = 0; bounce < 8; bounce++) {
		intersector<> isect;
		isect.assume_geometry_type(geometry_type::triangle);
		auto hit = isect.intersect(exit_ray, accel);
		if (hit.type == intersection_type::none || hit.distance <= 0.0) return false;

		float travel = -log(max(1.0 - rng_float(seed), 1.0e-6)) * mean_free_path;
		float segment = min(travel, hit.distance);
		// Beer-Lambert attenuation retains the authored per-channel radius even
		// when the sampled free-flight distribution uses a scalar majorant.
		throughput *= exp(-float3(segment) / radius);
		if (max(throughput.x, max(throughput.y, throughput.z)) <= 1.0e-5) return false;

		if (travel >= hit.distance) {
			// The direction points from the medium through this boundary toward
			// the exterior. The outer path loop will continue toward lights.
			exit_ray.origin += exit_ray.direction * (hit.distance + 0.001);
			exit_ray.min_distance = 0.001;
			exit_ray.max_distance = INFINITY;
			return true;
		}

		exit_ray.origin += exit_ray.direction * travel;
		throughput *= albedo;
		// Russian roulette preserves energy while capping long internal walks.
		float survive = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
		if (rng_float(seed) > survive) return false;
		throughput /= survive;
		exit_ray.direction = rng_unit_vector(seed);
		exit_ray.min_distance = 0.001;
		exit_ray.max_distance = INFINITY;
	}
	return false;
}

// Christensen-Burley normalized diffusion profile.  Sampling its radial PDF
// selects the BSSRDF's exit point, rather than evaluating a Lambertian BRDF at
// the entry point. The profile is a 1/4 exponential with scale d plus a 3/4
// exponential with scale 3d, where d is the MaterialX mean free path.
static bool subsurface_sample_exit(
	GPUMaterial mat, float3 entry_point, float3 entry_normal,
	primitive_acceleration_structure accel,
	device const TriVertex* vertices, device const uint* indices,
	thread uint& seed, thread float3& exit_point, thread float3& exit_normal
) {
	float d = max(luminance(max(mat.params5.xyz, float3(1.0e-5))), 1.0e-5);
	float scale = rng_float(seed) < 0.25 ? d : 3.0 * d;
	float radius = -scale * log(max(1.0 - rng_float(seed), 1.0e-6));
	float phi = 2.0 * PI * rng_float(seed);
	float3 tangent = make_tangent(entry_normal);
	float3 bitangent = cross(entry_normal, tangent);
	float3 offset = radius * (cos(phi) * tangent + sin(phi) * bitangent);

	ray probe;
	probe.origin = entry_point + offset + entry_normal * max(6.0 * d, 0.002);
	probe.direction = -entry_normal;
	probe.min_distance = 0.001;
	probe.max_distance = max(12.0 * d, 0.01);
	intersector<> isect;
	isect.assume_geometry_type(geometry_type::triangle);
	auto hit = isect.intersect(probe, accel);
	if (hit.type == intersection_type::none || hit.distance <= 0.0) return false;
	exit_point = probe.origin + probe.direction * hit.distance;
	uint b = hit.primitive_id * 3;
	float3 p0 = vertices[indices[b]].position.xyz;
	float3 p1 = vertices[indices[b + 1]].position.xyz;
	float3 p2 = vertices[indices[b + 2]].position.xyz;
	exit_normal = normalize(cross(p1 - p0, p2 - p0));
	if (dot(exit_normal, entry_normal) < 0.0) exit_normal = -exit_normal;
	return true;
}

// IEC 61966-2-1 sRGB electro-optical transfer function.
static float srgb_to_linear(float c) {
	return c <= 0.04045 ? c / 12.92 : powr((c + 0.055) / 1.055, 2.4);
}

// Fetch one texel from a packed RGBA8 texture. `tex_offset` is the starting
// pixel index in the buffer (not bytes); `width` is the texture width.
//
// `srgb` selects the decode: color maps (base color, emissive) are sRGB-encoded
// and are converted to linear here, before bilinear filtering, so that shading
// works in linear space. Data maps (metallic-roughness, normal) are already
// linear and must be read raw — decoding them would skew roughness and bend
// normals. Alpha is always linear.
static float4 fetch_tex_pixel(
	device const uchar* tex_pixels,
	int tex_offset, int width,
	int x, int y, bool srgb
) {
	int idx = tex_offset + y * width + x;
	int off = idx * 4;
	float3 rgb = float3(
		float(tex_pixels[off + 0]) / 255.0,
		float(tex_pixels[off + 1]) / 255.0,
		float(tex_pixels[off + 2]) / 255.0
	);
	if (srgb) {
		rgb = float3(srgb_to_linear(rgb.x), srgb_to_linear(rgb.y), srgb_to_linear(rgb.z));
	}
	return float4(rgb, float(tex_pixels[off + 3]) / 255.0);
}

static int wrap_coord(int idx, int w) {
	int out = idx % w;
	if (out < 0) out += w;
	return out;
}

// Bilinear sample of a packed RGBA8 texture. `info` is the material's
// {pixel_offset, width, height, has_tex} descriptor.
static float3 sample_tex_rgba8(
	device const uchar* tex_pixels,
	float4 info,
	float2 uv,
	bool srgb
) {
	int tex_offset = int(info.x);
	int width = int(info.y);
	int height = int(info.z);

	// Wrap to [0, 1)
	float uu = uv.x - floor(uv.x);
	float vv = uv.y - floor(uv.y);
	// Texel coordinates centered on pixel centers
	float px = uu * float(width) - 0.5;
	float py = (1.0 - vv) * float(height) - 0.5; // flip V (loader stores bottom-up)
	int x0 = int(floor(px));
	int y0 = int(floor(py));
	int x1 = x0 + 1;
	int y1 = y0 + 1;
	float fx = px - float(x0);
	float fy = py - float(y0);

	int xi0 = wrap_coord(x0, width);
	int xi1 = wrap_coord(x1, width);
	int yi0 = wrap_coord(y0, height);
	int yi1 = wrap_coord(y1, height);

	float4 p00 = fetch_tex_pixel(tex_pixels, tex_offset, width, xi0, yi0, srgb);
	float4 p10 = fetch_tex_pixel(tex_pixels, tex_offset, width, xi1, yi0, srgb);
	float4 p01 = fetch_tex_pixel(tex_pixels, tex_offset, width, xi0, yi1, srgb);
	float4 p11 = fetch_tex_pixel(tex_pixels, tex_offset, width, xi1, yi1, srgb);

	float a = (1.0 - fx) * (1.0 - fy);
	float b = fx * (1.0 - fy);
	float c = (1.0 - fx) * fy;
	float d = fx * fy;
	return (p00 * a + p10 * b + p01 * c + p11 * d).rgb;
}

// Perturb the interpolated shading normal by a tangent-space normal-map
// sample. The tangent frame is derived from the triangle's position and UV
// derivatives, so no per-vertex TANGENT attribute is required.
//
// `ts` is the decoded map value in [-1, 1]. glTF normal maps use the OpenGL
// convention (green points up), which in UV space means +Y runs against
// dP/dv, since v increases downward.
static float3 perturb_normal(
	float3 n,
	float3 p0, float3 p1, float3 p2,
	float2 uv0, float2 uv1, float2 uv2,
	float3 ts
) {
	float3 e1 = p1 - p0;
	float3 e2 = p2 - p0;
	float2 d1 = uv1 - uv0;
	float2 d2 = uv2 - uv0;
	float det = d1.x * d2.y - d2.x * d1.y;
	if (fabs(det) < 1.0e-12) return n;
	float inv = 1.0 / det;

	float3 dpdu = ( e1 * d2.y - e2 * d1.y) * inv;
	float3 dpdv = (-e1 * d2.x + e2 * d1.x) * inv;

	// Gram-Schmidt: project the tangent into the plane of the shading normal.
	float3 t = dpdu - n * dot(n, dpdu);
	if (length(t) < 1.0e-12) return n;
	t = normalize(t);

	float3 bt = cross(n, t);
	if (dot(bt, -dpdv) < 0.0) bt = -bt;

	float3 mapped = ts.x * t + ts.y * bt + ts.z * n;
	if (length(mapped) < 1.0e-12) return n;
	return normalize(mapped);
}

// UV-aligned surface tangent (dP/du) for anisotropy, from the triangle's
// position and UV derivatives — the same basis perturb_normal uses. Falls
// back to an arbitrary (but stable) tangent when the triangle has no usable
// UVs, in which case anisotropy orientation is undefined anyway.
static float3 derive_tangent(
	float3 n,
	float3 p0, float3 p1, float3 p2,
	float2 uv0, float2 uv1, float2 uv2,
	bool has_uv
) {
	if (!has_uv) return make_tangent(n);
	float3 e1 = p1 - p0;
	float3 e2 = p2 - p0;
	float2 d1 = uv1 - uv0;
	float2 d2 = uv2 - uv0;
	float det = d1.x * d2.y - d2.x * d1.y;
	if (fabs(det) < 1.0e-12) return make_tangent(n);
	float3 dpdu = (e1 * d2.y - e2 * d1.y) / det;
	float3 t = dpdu - n * dot(n, dpdu);
	if (length(t) < 1.0e-12) return make_tangent(n);
	return normalize(t);
}

static float triangle_area(float3 p0, float3 p1, float3 p2) {
	return 0.5 * length(cross(p1 - p0, p2 - p0));
}

static float light_pdf_solid_angle(float dist2, float cos_light, float area, int light_count) {
	if (light_count <= 0 || cos_light <= 0.0 || area <= 0.0) {
		return 0.0;
	}
	return dist2 / (cos_light * area * float(light_count));
}

static float3 sample_quad_light(GPUQuadLight light, float3 from_point, thread uint& seed, thread float3& light_pos, thread float& light_dist, thread float3& light_normal, thread float& pdf) {
	float r1 = rng_float(seed);
	float r2 = rng_float(seed);
	float3 pos = light.position.xyz + r1 * light.u.xyz + r2 * light.v.xyz;
	light_normal = normalize(cross(light.u.xyz, light.v.xyz));
	float3 to_light = pos - from_point;
	light_dist = length(to_light);
	float3 dir = to_light / light_dist;
	float cos_light = max(fabs(dot(light_normal, -dir)), 0.001);
	float area = length(cross(light.u.xyz, light.v.xyz));
	pdf = (light_dist * light_dist) / (cos_light * area);
	return dir;
}

static float3 sample_sphere_light(GPUSphereLight light, float3 from_point, thread uint& seed, thread float3& light_pos, thread float& light_dist, thread float3& light_normal, thread float& pdf) {
	float3 center = light.position.xyz;
	float radius = max(light.radius, 0.001);
	float3 to_center = center - from_point;
	float center_dist = length(to_center);
	float3 dir_to_center = to_center / center_dist;

	float sin_theta_max = radius / center_dist;
	float cos_theta_max = sqrt(max(1.0 - sin_theta_max * sin_theta_max, 0.0));

	float eps1 = rng_float(seed);
	float eps2 = rng_float(seed);
	float cos_theta = 1.0 - eps1 * (1.0 - cos_theta_max);
	float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
	float phi = 2.0 * PI * eps2;

	float3 helper = fabs(dir_to_center.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
	float3 tangent = normalize(cross(helper, dir_to_center));
	float3 bitangent = cross(dir_to_center, tangent);

	float3 local_dir = cos_theta * dir_to_center + sin_theta * (cos(phi) * tangent + sin(phi) * bitangent);

	light_dist = center_dist * cos_theta - sqrt(max(radius * radius - center_dist * center_dist * sin_theta * sin_theta, 0.0));
	light_pos = from_point + light_dist * local_dir;
	light_normal = normalize(light_pos - center);
	float3 surface_to_eye = -local_dir;
	if (dot(light_normal, surface_to_eye) < 0.0) light_normal = -light_normal;

	pdf = 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
	return local_dir;
}

static float3 sample_disc_light(GPUDiscLight light, float3 from_point, thread uint& seed, thread float& light_dist, thread float3& light_normal, thread float& pdf) {
	float3 center = light.position.xyz;
	float radius = max(light.position.w, 0.001);
	float3 n = normalize(light.normal.xyz);
	float r = radius * sqrt(rng_float(seed));
	float phi = 2.0 * PI * rng_float(seed);
	float3 t = make_tangent(n);
	float3 b = cross(n, t);
	float3 pos = center + r * (cos(phi) * t + sin(phi) * b);
	float area = PI * radius * radius;
	float3 to_light = pos - from_point;
	light_dist = length(to_light);
	float3 dir = to_light / light_dist;
	if (dot(n, dir) > 0.0) n = -n;
	light_normal = n;
	float cos_light = max(fabs(dot(n, dir)), 0.001);
	pdf = (light_dist * light_dist) / (cos_light * area);
	return dir;
}

static float3 sample_cylinder_light(GPUCylinderLight light, float3 from_point, thread uint& seed, thread float& light_dist, thread float3& light_normal, thread float& pdf) {
	float3 base = light.position.xyz;
	float radius = max(light.position.w, 0.001);
	float3 axis = normalize(light.axis.xyz);
	float height = max(light.axis.w, 0.001);
	float3 t = make_tangent(axis);
	float3 b = cross(axis, t);
	float tt = height * rng_float(seed);
	float phi = 2.0 * PI * rng_float(seed);
	float3 radial = cos(phi) * t + sin(phi) * b;
	float3 pos = base + tt * axis + radius * radial;
	float area = 2.0 * PI * radius * height;
	float3 to_light = pos - from_point;
	light_dist = length(to_light);
	float3 dir = to_light / light_dist;
	light_normal = radial;
	float cos_light = max(fabs(dot(radial, dir)), 0.001);
	pdf = (light_dist * light_dist) / (cos_light * area);
	return dir;
}

// ── HDRI environment (dome light) ───────────────────────────────────────────

static float3 rotate_y_msl(float3 d, float a) {
	if (a == 0.0) return d;
	float ca = cos(a), sa = sin(a);
	return float3(d.x * ca + d.z * sa, d.y, -d.x * sa + d.z * ca);
}

static float env_lum(float3 c) {
	return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

static float3 env_texel(device const float* px, int w, int h, int x, int y) {
	int xi = ((x % w) + w) % w;
	int yi = clamp(y, 0, h - 1);
	int idx = (yi * w + xi) * 3;
	return float3(px[idx], px[idx + 1], px[idx + 2]);
}

static float3 env_lookup(constant GPUSceneData& scene, device const float* px, float3 dir) {
	float3 d = normalize(rotate_y_msl(dir, -scene.env_rotation));
	float theta = acos(clamp(d.y, -1.0, 1.0));
	float phi = atan2(d.z, d.x);
	float u = (phi + PI) / (2.0 * PI);
	float v = theta / PI;
	int w = scene.env_width, h = scene.env_height;
	float fx = u * float(w) - 0.5;
	float fy = v * float(h) - 0.5;
	int x0 = int(floor(fx)), y0 = int(floor(fy));
	float tx = fx - float(x0), ty = fy - float(y0);
	float3 c00 = env_texel(px, w, h, x0, y0);
	float3 c10 = env_texel(px, w, h, x0 + 1, y0);
	float3 c01 = env_texel(px, w, h, x0, y0 + 1);
	float3 c11 = env_texel(px, w, h, x0 + 1, y0 + 1);
	float3 top = mix(c00, c10, tx);
	float3 bot = mix(c01, c11, tx);
	return mix(top, bot, ty) * scene.env_intensity;
}

static float env_pdf(constant GPUSceneData& scene, device const float* px, float3 dir) {
	if (scene.env_func_int <= 0.0) return 0.0;
	float3 d = normalize(rotate_y_msl(dir, -scene.env_rotation));
	float theta = acos(clamp(d.y, -1.0, 1.0));
	float sin_theta = sin(theta);
	if (sin_theta <= 0.0) return 0.0;
	float phi = atan2(d.z, d.x);
	float u = (phi + PI) / (2.0 * PI);
	float v = theta / PI;
	int w = scene.env_width, h = scene.env_height;
	int col = clamp(int(u * float(w)), 0, w - 1);
	int row = clamp(int(v * float(h)), 0, h - 1);
	int idx = (row * w + col) * 3;
	float lum = env_lum(float3(px[idx], px[idx + 1], px[idx + 2]));
	return lum / (2.0 * PI * PI * scene.env_func_int);
}

static float sample_cdf_msl(device const float* cdf, int offset, int n, float xi, thread int& bucket) {
	int lo = 0, hi = n;
	while (lo + 1 < hi) {
		int mid = (lo + hi) / 2;
		if (cdf[offset + mid] <= xi) lo = mid; else hi = mid;
	}
	bucket = lo;
	float c0 = cdf[offset + lo], c1 = cdf[offset + lo + 1];
	float du = (c1 > c0) ? (xi - c0) / (c1 - c0) : 0.0;
	return (float(lo) + du) / float(n);
}

static float3 env_sample(constant GPUSceneData& scene, device const float* px, device const float* marg, device const float* cond, thread uint& seed, thread float3& radiance, thread float& pdf) {
	int w = scene.env_width, h = scene.env_height;
	float xi1 = rng_float(seed), xi2 = rng_float(seed);
	int row;
	float v = sample_cdf_msl(marg, 0, h, xi1, row);
	int col;
	float u = sample_cdf_msl(cond, row * (w + 1), w, xi2, col);
	float theta = v * PI;
	float phi = u * 2.0 * PI - PI;
	float sin_theta = sin(theta);
	float3 base = float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
	float3 dir = rotate_y_msl(base, scene.env_rotation);
	if (sin_theta <= 0.0 || scene.env_func_int <= 0.0) {
		radiance = float3(0.0);
		pdf = 0.0;
		return dir;
	}
	int cc = clamp(int(u * float(w)), 0, w - 1);
	int rr = clamp(int(v * float(h)), 0, h - 1);
	int idx = (rr * w + cc) * 3;
	float lum = env_lum(float3(px[idx], px[idx + 1], px[idx + 2]));
	float map_pdf = lum * sin_theta / scene.env_func_int;
	pdf = map_pdf / (2.0 * PI * PI * sin_theta);
	radiance = env_lookup(scene, px, dir);
	return dir;
}

	// ── Photon hash grid ───────────────────────────────────────────────────────

static int photon_hash_cell(float3 cell) {
	// NOTE: hash the *integer* cell coordinate. Reinterpreting the float bits
	// (as_type<int>) collapses the grid: integer-valued floats have their low
	// mantissa bits zero, so after the prime multiply + low-bit mask every
	// cell maps to bucket 0.
	uint h = (uint(int(cell.x)) * 73856093u) ^
	         (uint(int(cell.y)) * 19349663u) ^
	         (uint(int(cell.z)) * 83492791u);
	return int(h & uint(PHOTON_GRID_SIZE - 1));
}

// Radiance estimate from the global photon map. The grid is a counting-sort
// hash grid: `grid_offsets[cell]` is the start of that bucket inside
// `grid_sorted`, and `grid_counts[cell]` its length — so *every* photon in
// the neighbourhood is visited (no per-cell cap). Density is estimated with a
// Jensen cone filter of support `radius`; with cone slope k=1 the filter
// normalization is 1/((1 - 2/3)·π r²) = 3/(π r²).
static float3 photon_query(
	device const Photon* photons,
	device const int* grid_offsets,
	device const int* grid_counts,
	device const int* grid_sorted,
	float3 pos, float3 normal,
	float radius, float3 albedo,
	thread int& found_out
) {
	found_out = 0;
	if (radius <= 0.0) return float3(0.0);

	float3 base_cell = floor(pos / radius);
	float3 accum = 0.0;
	float norm = 3.0 / (PI * radius * radius);
	float3 brdf = albedo * INV_PI;
	int found = 0;

	for (int ix = -1; ix <= 1; ix++) {
		for (int iy = -1; iy <= 1; iy++) {
			for (int iz = -1; iz <= 1; iz++) {
				float3 ncell = base_cell + float3(float(ix), float(iy), float(iz));
				int cell = photon_hash_cell(ncell);
				int start = grid_offsets[cell];
				int count = grid_counts[cell];

				for (int j = 0; j < count; j++) {
					int pi = grid_sorted[start + j];
					Photon ph = photons[pi];

					float3 ph_cell = float3(ph.position.w, ph.incident.w, ph.power.w);
					if (any(ph_cell != ncell)) continue;

					float3 delta = ph.position.xyz - pos;
					float dist = length(delta);
					if (dist > radius) continue;

					// Front-side validity: keep photons that arrived onto the
					// same side as the shading normal. This is a rejection
					// test, not an energy weight (the arrival cosine is already
					// baked into the stored flux).
					if (dot(normal, ph.incident.xyz) <= 0.0) continue;

					float w = 1.0 - dist / radius;   // cone filter
					accum += brdf * ph.power.xyz * w;
					found++;
				}
			}
		}
	}
	found_out = found;
	return accum * norm;
}

	// ── Irradiance Cache (hash grid + deferred write) ────────────────────────────

static int gi_hash_cell(float3 cell) {
	// Hash the *integer* cell coordinate. Reinterpreting the float bits
	// (as_type<int>) collapses the grid: integer-valued floats have their low
	// mantissa bits zero, so after the prime multiply + low-bit mask every
	// cell maps to bucket 0 (same bug that was fixed in photon_hash_cell).
	uint h = (uint(int(cell.x)) * 73856093u) ^
	         (uint(int(cell.y)) * 19349663u) ^
	         (uint(int(cell.z)) * 83492791u);
	return int(h & uint(GI_GRID_SIZE - 1));
}

static float3 gi_cache_query(
	device const GICachePoint* cache,
	device const int* grid_cells,
	device const atomic_int* grid_counts,
	int num_cells,
	float3 pos,
	float3 normal,
	float max_dist,
	float normal_angle,
	thread int& sample_count,
	thread float& rel_stddev
) {
	sample_count = 0;
	rel_stddev = 1.0e6;
	if (num_cells <= 0 || max_dist <= 0.0) return float3(0.0);

	float3 base_cell = floor(pos / max_dist);

	float3 result = 0.0;
	float total_weight = 0.0;
	float lum_sum = 0.0;
	float lum2_sum = 0.0;

	for (int ix = -1; ix <= 1; ix++) {
		for (int iy = -1; iy <= 1; iy++) {
			for (int iz = -1; iz <= 1; iz++) {
				float3 ncell = base_cell + float3(float(ix), float(iy), float(iz));
				int cell = gi_hash_cell(ncell);
				int count = min(atomic_load_explicit(&grid_counts[cell], memory_order_relaxed), GI_MAX_PER_CELL);

				for (int j = 0; j < count; j++) {
					int ci = grid_cells[cell * GI_MAX_PER_CELL + j];
					GICachePoint cp = cache[ci];

					float3 cp_cell = float3(cp.position.w, cp.normal.w, cp.irradiance.w);
					if (any(cp_cell != ncell)) continue;

					float3 delta = cp.position.xyz - pos;
					float dist = length(delta);
					if (dist > max_dist) continue;

					float normal_sim = max(dot(normal, cp.normal.xyz), 0.0);
					if (normal_sim < normal_angle) continue;

					float radial = max(1.0 - dist / max_dist, 0.0);
					float w = radial * radial * normal_sim * normal_sim;
					result += cp.irradiance.xyz * w;
					total_weight += w;
					float lum = luminance(cp.irradiance.xyz);
					lum_sum += lum * w;
					lum2_sum += lum * lum * w;
					sample_count++;
				}
			}
		}
	}

	if (total_weight > 0.0) {
		float mean_lum = lum_sum / total_weight;
		float mean_lum2 = lum2_sum / total_weight;
		float variance = max(mean_lum2 - mean_lum * mean_lum, 0.0);
		rel_stddev = sqrt(variance) / max(mean_lum, 1.0e-4);
		return result / total_weight;
	}
	return float3(0.0);
}

static void gi_cache_store(
	device GICachePoint* cache,
	device atomic_int* counter,
	device int* grid_cells,
	device atomic_int* grid_counts,
	float cell_size,
	float3 pos,
	float3 normal,
	float3 irradiance
) {
	int idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
	idx = idx % GI_CACHE_MAX_POINTS;

	float3 cell_coord = floor(pos / cell_size);
	cache[idx].position = float4(pos, cell_coord.x);
	cache[idx].normal = float4(normal, cell_coord.y);
	cache[idx].irradiance = float4(irradiance, cell_coord.z);

	int cell = gi_hash_cell(cell_coord);
	int count = atomic_fetch_add_explicit(&grid_counts[cell], 1, memory_order_relaxed);
	if (count < GI_MAX_PER_CELL) {
		grid_cells[cell * GI_MAX_PER_CELL + count] = idx;
	}
}

static void gi_cache_deferred_write(
	device GICachePoint* cache,
	device atomic_int* counter,
	device int* grid_cells,
	device atomic_int* grid_counts,
	float cell_size,
	thread int& cache_pending,
	thread float3& cache_p_pos,
	thread float3& cache_p_normal,
	thread float3& cache_p_throughput,
	thread float3& cache_p_accum_before,
	float3 accumulated
) {
	if (!cache_pending) return;
	float3 delta = accumulated - cache_p_accum_before;
	float3 throughput = max(cache_p_throughput, 1e-8);
	float3 irradiance = delta * PI / throughput;
	if (luminance(irradiance) > 0.0) {
		gi_cache_store(cache, counter, grid_cells, grid_counts,
			cell_size,
			cache_p_pos, cache_p_normal, irradiance);
	}
	cache_pending = 0;
}

static float3 emissive_radiance(GPUMaterial mat) {
	float3 color = mat.emission.xyz;
	if (luminance(color) <= 0.0) {
		color = mat.albedo.xyz;
	}

	float strength = mat.params1.y;
	if (strength <= 0.0) {
		strength = 20.0;
	}
	return color * strength;
}

// ── Ray tracing kernel (triangle AS, built-in intersection) ─────────────────

kernel void raytraceKernel(
	uint2                              tid            [[thread_position_in_grid]],
	constant GPUSceneData&             scene          [[buffer(0)]],
	device const GPUMaterial*          materials      [[buffer(1)]],
	device float4*                     output         [[buffer(2)]],
	primitive_acceleration_structure   accel          [[buffer(3)]],
	device const TriVertex*            vertices       [[buffer(4)]],
	device const uint*                 indices        [[buffer(5)]],
	device const GPULightTriangle*     tri_lights     [[buffer(6)]],
	device const GPUQuadLight*         quad_lights    [[buffer(7)]],
	device const GPUSphereLight*       sphere_lights  [[buffer(8)]],
	device const int*                  mat_indices    [[buffer(9)]],
	device GICachePoint*               gi_cache       [[buffer(10)]],
	device atomic_int*                 gi_counter     [[buffer(11)]],
	device int*                        gi_grid_cells  [[buffer(12)]],
	device atomic_int*                 gi_grid_counts [[buffer(13)]],
	device const Photon*               photons        [[buffer(14)]],
	device const atomic_int*           photon_counter [[buffer(15)]],
	device const int*                  photon_grid_offsets [[buffer(16)]],
	device const int*                  photon_grid_counts  [[buffer(17)]],
	device const uchar*                tex_pixels     [[buffer(18)]],
	device const int*                  photon_grid_sorted  [[buffer(19)]],
	device const GPUDiscLight*         disc_lights    [[buffer(20)]],
	device const GPUCylinderLight*     cylinder_lights [[buffer(21)]],
	device const GPUPunctualLight*     punctual_lights [[buffer(22)]],
	device const float*                env_pixels     [[buffer(23)]],
	device const float*                env_marginal   [[buffer(24)]],
	device const float*                env_conditional [[buffer(25)]]
) {
	if (tid.x >= uint(scene.image_width) ||
	    tid.y >= uint(scene.image_height)) return;

	uint pixel_idx = tid.y * uint(scene.image_width) + tid.x;
	uint seed = scene.seed + pixel_idx;

	float3 pixel_color = 0.0;

	for (int s = 0; s < scene.samples_per_pixel; s++) {
		float u = (float(tid.x) + rng_float(seed)) / float(scene.image_width - 1);
		float v = (float(tid.y) + rng_float(seed)) / float(scene.image_height - 1);

		float3 ro = scene.origin.xyz;
		float3 rd = scene.lower_left.xyz
			+ u * scene.horizontal.xyz
			+ v * scene.vertical.xyz
			- ro;

		// Depth of Field
		if (scene.lens_radius > 0.0) {
			float3 rd_disk = rng_in_unit_disk(seed) * scene.lens_radius;
			float3 offset = rd_disk.x * scene.u.xyz + rd_disk.y * scene.v.xyz;
			ro += offset;
			rd -= offset;
		}

		ray r;
		r.origin = ro;
		r.direction = rd;
		r.min_distance = 0.001;
		r.max_distance = INFINITY;

		float3 ray_color = 1.0;
		float3 accumulated = 0.0;
		float last_bsdf_pdf = 0.0;
		bool last_was_delta = false;
		bool debug_found = false;

		// Denoiser-guide state (debug modes 20/21/22). The guides must
		// describe the surface the pixel actually shows, so on a mirror or
		// glass hit we follow the specular/refractive bounce and only record
		// the guide once we reach a non-delta (diffuse/glossy) surface. This
		// is what lets the À-Trous edge-stops and albedo demodulation keep
		// the reflected/refracted image sharp instead of smearing it to milk.
		float  guide_dist = 0.0;    // accumulated path length to the visible surface
		float3 guide_tint = 1.0;    // product of specular tints along the chain

		// Deferred irradiance cache state
		int cache_pending = 0;
		float3 cache_p_pos;
		float3 cache_p_normal;
		float3 cache_p_throughput;
		float3 cache_p_accum_before; // captures accumulated BEFORE NEE at the cache bounce

		for (int depth = 0; depth < scene.max_depth; depth++) {
			// Metal's built-in triangle intersector
			intersector<> i;
			i.assume_geometry_type(geometry_type::triangle);
			auto result = i.intersect(r, accel);

			if (result.type == intersection_type::none || result.distance >= INFINITY || result.distance <= 0.0) {
				// Guide passes: the ray escaped to the background (directly, or
				// through a mirror/glass). Record the background as the visible
				// "surface": zero normal, a far depth, and an albedo of the
				// accumulated specular tint so demodulation leaves the sky
				// reflection/refraction as illumination.
				if (scene.debug_mode >= 20 && scene.debug_mode <= 23) {
					if (scene.debug_mode == 20)      accumulated = float3(0.0);
					else if (scene.debug_mode == 21) accumulated = float3(guide_dist + 1.0e4);
					else if (scene.debug_mode == 22) accumulated = guide_tint;
					else                             accumulated = float3(0.0); // background emits nothing
					break;
				}
				float3 unit_dir = normalize(r.direction);
				float3 bg;
				if (scene.has_env != 0) {
					// HDRI dome. MIS-weight against environment NEE on a
					// BSDF-sampled diffuse/glossy bounce; camera and delta
					// bounces take the environment at full weight.
					bg = env_lookup(scene, env_pixels, unit_dir);
					if (depth > 0 && !last_was_delta) {
						float epdf = env_pdf(scene, env_pixels, unit_dir);
						bg *= power_heuristic(last_bsdf_pdf, epdf);
					}
				} else {
					if (scene.hide_default_sky != 0) {
						bg = float3(0.0);
					} else {
						float t = 0.5 * (unit_dir.y + 1.0);
						bg = (1.0 - t) * float3(1.0) + t * float3(0.5, 0.7, 1.0);
					}
				}
				accumulated += ray_color * bg;
				gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
					scene.gi_cache_distance,
					cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
					accumulated);
				break;
			}

			uint pid = result.primitive_id;
			float hit_dist = result.distance;
			float3 hit_point = r.origin + hit_dist * r.direction;

			uint base_idx = pid * 3;
			uint i0 = indices[base_idx];
			uint i1 = indices[base_idx + 1];
			uint i2 = indices[base_idx + 2];

			float3 p0 = vertices[i0].position.xyz;
			float3 p1 = vertices[i1].position.xyz;
			float3 p2 = vertices[i2].position.xyz;

			float3 edge1 = p1 - p0;
			float3 edge2 = p2 - p0;
			float3 geom_normal = normalize(cross(edge1, edge2));
			bool front_face = dot(r.direction, geom_normal) < 0.0;

			// Smooth shading via barycentric interpolation of vertex normals
			float3 edge_cross = cross(edge1, edge2);
			float full_area = length(edge_cross);
			float bu = length(cross(p1 - hit_point, p2 - hit_point)) / max(full_area, 1e-12);
			float bv = length(cross(p2 - hit_point, p0 - hit_point)) / max(full_area, 1e-12);
			float bw = length(cross(p0 - hit_point, p1 - hit_point)) / max(full_area, 1e-12);
			float sum = bu + bv + bw;
			bu /= sum; bv /= sum; bw /= sum;

			float3 vn0 = vertices[i0].normal.xyz;
			float3 vn1 = vertices[i1].normal.xyz;
			float3 vn2 = vertices[i2].normal.xyz;
			float3 shading_normal = normalize(bu * vn0 + bv * vn1 + bw * vn2);
			if (!front_face) shading_normal = -shading_normal;

			// Barycentric UV interpolation: compute the UV at the hit
			// point from the per-vertex UVs. Only meaningful when
			// the assigned material has a texture.
			float2 uv0 = vertices[i0].uv.xy;
			float2 uv1 = vertices[i1].uv.xy;
			float2 uv2 = vertices[i2].uv.xy;
			float2 hit_uv = float2(0.0);
			bool has_hit_uv = false;
			if (vertices[i0].uv.z > 0.5) {
				hit_uv = bu * uv0 + bv * uv1 + bw * uv2;
				has_hit_uv = true;
			}

			int midx = mat_indices[pid];
			GPUMaterial mat = materials[midx];
			int mat_kind = int(mat.params0.x);

			// Resolve the material's maps at the hit UV. Base color and
			// metallic-roughness modulate the constant factors; the normal
			// map replaces the interpolated shading normal.
			if (has_hit_uv) {
				if (mat.tex_info.w > 0.5) {
					float3 sampled = sample_tex_rgba8(tex_pixels, mat.tex_info, hit_uv, true);
					mat.albedo = float4(mat.albedo.xyz * sampled, 1.0);
				}
				if (mat.mr_info.w > 0.5) {
					// glTF packs occlusion/roughness/metallic into R/G/B.
					float3 mr = sample_tex_rgba8(tex_pixels, mat.mr_info, hit_uv, false);
					mat.params0.w = clamp(mat.params0.w * mr.y, 0.0, 1.0);
					mat.params1.x = clamp(mat.params1.x * mr.z, 0.0, 1.0);
				}
				if (mat.nrm_info.w > 0.5) {
					float3 ts = sample_tex_rgba8(tex_pixels, mat.nrm_info, hit_uv, false) * 2.0 - 1.0;
					ts.xy *= mat.params2.z; // normal_scale
					shading_normal = perturb_normal(
						shading_normal, p0, p1, p2, uv0, uv1, uv2, ts);
					// A perturbed normal can point away from the viewer at
					// grazing angles; fold it back so the BSDF stays valid.
					if (dot(shading_normal, geom_normal) * (front_face ? 1.0 : -1.0) < 0.0) {
						shading_normal = front_face ? geom_normal : -geom_normal;
					}
				}
			}
			// The direct-light estimator uses a local diffusion approximation;
			// the scatter phase below adds the non-local random-walk component.
			if (mat.params4.w > 0.0) {
				mat.albedo = float4(mix(mat.albedo.xyz, mat.params4.xyz, mat.params4.w), 1.0);
			}

			// UV-aligned tangent for anisotropic shading (unused when the
			// material is isotropic). Derived after the shading normal is
			// finalized so it lies in the perturbed surface's plane.
			float3 shading_tangent = derive_tangent(
				shading_normal, p0, p1, p2, uv0, uv1, uv2, has_hit_uv);
			// Evaluate the subsurface lobe at a sampled *exit* point. This is a
			// BSSRDF surface-diffusion event (Christensen-Burley profile), not a
			// local diffuse BRDF at the camera-facing entry point.
			if (mat.params4.w > 0.0 && front_face) {
				float3 sss_point, sss_normal;
				if (subsurface_sample_exit(mat, hit_point, shading_normal, accel, vertices, indices, seed, sss_point, sss_normal)) {
					hit_point = sss_point;
					shading_normal = sss_normal;
					shading_tangent = make_tangent(sss_normal);
				}
			}

			// Consolidation (Phase D): a pure-diffuse Principled surface — what
			// normalize_material produces from a Lambertian — routes to the
			// diffuse GI-cache/photon fast path (kind 0). This reuses the
			// existing Lambertian shading + NEE + guide machinery unchanged, so
			// a converted Lambertian renders byte-identically to the old kind.
			if (material_is_lambertian_like(mat)) {
				mat_kind = 0;
			}

			// GGX degenerates at zero roughness (D collapses to a delta), which
			// leaves perfectly smooth metals black. Keep a sliver of spread.
			if (mat_kind == 3) {
				mat.params0.w = max(mat.params0.w, 0.015);
			}

			// Roughness cutoff: treat high-roughness materials as fully diffuse.
			// Glossy bias: damp the effective roughness for cheaper blurry
			// reflections. 0 = full GGX, 1 = flat mirror approximation.
			if (mat_kind == 3) {
				if (scene.glossy_bias > 0.0) {
					mat.params0.w = clamp(mat.params0.w * (1.0 - scene.glossy_bias) + scene.glossy_bias, 0.0, 1.0);
				}
				if (scene.roughness_cutoff > 0.0) {
					if (mat.params0.w > scene.roughness_cutoff) {
						mat_kind = 0;
					}
				}
			}

			// Denoiser guides (modes 20/21/22): follow perfect specular /
			// refractive bounces so the guide describes the surface actually
			// seen through the glass or mirror, not its skin. A delta hit adds
			// its tint and defers the write; the first non-delta (diffuse or
			// glossy) hit — or a miss, handled above — records the guide.
			if (scene.debug_mode >= 20 && scene.debug_mode <= 23) {
				guide_dist += hit_dist;
				bool surf_delta =
					(mat_kind == 1) ||                                    // metal
					(mat_kind == 2) ||                                    // dielectric
					(mat_kind == 3 && pbr_alpha_sq(mat.params0.w) < 1.0e-4); // near-mirror GGX
				if (surf_delta && depth < scene.max_depth - 1) {
					if (mat_kind == 1) guide_tint *= mat.albedo.xyz;      // metals tint the reflection
					// Fall through: the material-scatter section below sets the
					// specular bounce (deterministically in guide passes), and
					// the loop continues to the next hit.
				} else {
					if (scene.debug_mode == 20)      accumulated = shading_normal;
					else if (scene.debug_mode == 21) accumulated = float3(guide_dist);
					else if (scene.debug_mode == 22) accumulated = guide_tint * mat.albedo.xyz;
					else {
						// Emission guide: the noise-free radiance the beauty adds
						// at this surface (an emissive map, or a pure emitter).
						// Held out of demodulation so the À-Trous filter never
						// blurs self-emitted detail — HUD/screen glow, emissive
						// textures — into the surrounding lighting.
						float3 e = float3(0.0);
						if (mat_kind == 4) {
							e = emissive_radiance(mat);
						} else if (mat.emis_info.w > 0.5 && has_hit_uv) {
							e = mat.emission.xyz *
								sample_tex_rgba8(tex_pixels, mat.emis_info, hit_uv, true);
						}
						accumulated = guide_tint * e;
					}
					break;
				}
			}

			// Debug modes: output diagnostic data on first bounce
			if (depth == 0 && scene.debug_mode > 0 && scene.debug_mode < 20) {
				if (scene.debug_mode == 1) {
					// Albedo
					accumulated = mat.albedo.xyz;
				} else if (scene.debug_mode == 2) {
					// Normal (mapped to [0,1])
					accumulated = shading_normal * 0.5 + 0.5;
				} else if (scene.debug_mode == 3) {
					// Depth (normalized)
					float d = result.distance / 20.0;
					accumulated = float3(d, d, d);
				} else if (scene.debug_mode == 4) {
					// Primitive ID (color-coded)
					float r = float(pid & 0xFF) / 255.0;
					float g = float((pid >> 8) & 0xFF) / 255.0;
					float b = float((pid >> 16) & 0xFF) / 255.0;
					accumulated = float3(r, g, b);
				} else if (scene.debug_mode == 5) {
					// Direct light contribution only.
					accumulated = 0.0;
				} else if (scene.debug_mode == 6) {
					// Light count visible to the shader.
					float c = min(float(scene.tri_light_count) / 4.0, 1.0);
					accumulated = float3(c, 0.0, 0.0);
				} else if (scene.debug_mode == 7) {
					// Direct light candidate geometry before shadowing.
					accumulated = 0.0;
				} else if (scene.debug_mode == 8) {
					// Shadow visibility only: green means unoccluded.
					accumulated = 0.0;
				} else if (scene.debug_mode == 14) {
					// UV coordinates.
					accumulated = has_hit_uv ? float3(fract(hit_uv), 0.0) : float3(1.0, 0.0, 1.0);
				} else if (scene.debug_mode == 15) {
					// Raw albedo texture sample before material factors.
					if (mat.tex_info.w > 0.5 && has_hit_uv) {
						accumulated = sample_tex_rgba8(tex_pixels, mat.tex_info, hit_uv, true);
					} else {
						accumulated = float3(1.0, 0.0, 1.0);
					}
				} else if (scene.debug_mode == 16) {
					// Roughness (green) and metallic (blue) after map lookup.
					accumulated = float3(0.0, mat.params0.w, mat.params1.x);
				} else if (scene.debug_mode == 20) {
					// Denoiser guide: raw first-hit shading normal (may be
					// negative). Not tonemapped — see the write below.
					accumulated = shading_normal;
				} else if (scene.debug_mode == 21) {
					// Denoiser guide: raw first-hit view distance in .x.
					accumulated = float3(result.distance);
				} else if (scene.debug_mode == 22) {
					// Denoiser guide: effective first-hit albedo (base color
					// factor × base color texture). Used for demodulation so
					// the filter smooths illumination, not texture detail.
					accumulated = mat.albedo.xyz;
				}
				if (
					scene.debug_mode != 5 &&
					scene.debug_mode != 7 &&
					scene.debug_mode != 8 &&
					scene.debug_mode != 9 &&
					scene.debug_mode != 10 &&
					scene.debug_mode != 11 &&
					scene.debug_mode != 12 &&
					scene.debug_mode != 13
				) break;
			}

			// Snapshot accumulated before NEE for deferred cache write
			float3 accum_at_hit = accumulated;

			if (mat_kind == 4) {
				// Emissive
				float weight = 1.0;
				if (depth > 0 && !last_was_delta && scene.tri_light_count > 0) {
					float light_area = triangle_area(p0, p1, p2);
					float cos_light = max(fabs(dot(geom_normal, -normalize(r.direction))), 0.0);
					float light_pdf = light_pdf_solid_angle(hit_dist * hit_dist, cos_light, light_area, scene.tri_light_count);
					weight = power_heuristic(last_bsdf_pdf, light_pdf);
				}
				accumulated += ray_color * emissive_radiance(mat) * weight;
				gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
					scene.gi_cache_distance,
					cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
					accumulated);
				break;
			}

			// An emissive map turns an ordinary BSDF surface into a weak
			// emitter: its radiance is added on top of the shading below.
			// It is not registered as a light source, so it lights nothing
			// but itself and contributes no NEE samples.
			if (mat.emis_info.w > 0.5 && has_hit_uv) {
				float3 emis = mat.emission.xyz *
					sample_tex_rgba8(tex_pixels, mat.emis_info, hit_uv, true);
				accumulated += ray_color * emis;
			}

			// Direct light sampling (Next-Event Estimation)
			if ((mat_kind == 0 || mat_kind == 3) && !(scene.debug_mode == 9 && depth == 0)) {
				int total_lights = scene.tri_light_count + scene.quad_light_count + scene.sphere_light_count
					+ scene.disc_light_count + scene.cylinder_light_count + scene.punctual_light_count
					+ (scene.has_env != 0 ? 1 : 0);
				if (total_lights > 0) {
					int light_types_sampled = 0;
					float3 wo = normalize(-r.direction);

					// Triangle lights
					if (scene.tri_light_count > 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							uint li = min(uint(rng_float(seed) * float(scene.tri_light_count)), uint(scene.tri_light_count - 1));
							GPULightTriangle ltri = tri_lights[li];
							float3 lp0 = ltri.p0.xyz;
							float3 lp1 = ltri.p1.xyz;
							float3 lp2 = ltri.p2.xyz;

							float r1 = rng_float(seed);
							float r2 = rng_float(seed);
							float sqrt_r1 = sqrt(r1);
							float3 light_pos = (1.0 - sqrt_r1) * lp0 + (sqrt_r1 * (1.0 - r2)) * lp1 + (sqrt_r1 * r2) * lp2;

							float3 to_light = light_pos - hit_point;
							float light_dist = length(to_light);
							float3 light_dir = to_light / light_dist;
							float3 light_normal = normalize(cross(lp1 - lp0, lp2 - lp0));
							float light_area = triangle_area(lp0, lp1, lp2);
							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							float cos_light = max(fabs(dot(light_normal, -light_dir)), 0.0);

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float dist2 = max(light_dist * light_dist, 1.0e-6);
								float light_pdf = light_pdf_solid_angle(dist2, cos_light, light_area, scene.tri_light_count);
								float3 direct = nee_contribution(mat, mat_kind, wo, shading_normal, shading_tangent, light_dir, cos_surf, light_pdf, ltri.emission.xyz);
								accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
							}
						}
					}

					// Quad lights
					if (scene.quad_light_count > 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							uint li = min(uint(rng_float(seed) * float(scene.quad_light_count)), uint(scene.quad_light_count - 1));
							GPUQuadLight ql = quad_lights[li];

							float3 light_pos, light_normal;
							float light_dist, light_pdf_val;
							float3 light_dir = sample_quad_light(ql, hit_point, seed, light_pos, light_dist, light_normal, light_pdf_val);
							light_pdf_val /= float(scene.quad_light_count);

							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							if (cos_surf <= 0.0 || light_pdf_val <= 0.0) continue;

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float3 direct = nee_contribution(mat, mat_kind, wo, shading_normal, shading_tangent, light_dir, cos_surf, light_pdf_val, ql.emission.xyz);
								accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
							}
						}
					}

					// Sphere lights
					if (scene.sphere_light_count > 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							uint li = min(uint(rng_float(seed) * float(scene.sphere_light_count)), uint(scene.sphere_light_count - 1));
							GPUSphereLight sl = sphere_lights[li];

							float3 light_pos, light_normal;
							float light_dist, light_pdf_val;
							float3 light_dir = sample_sphere_light(sl, hit_point, seed, light_pos, light_dist, light_normal, light_pdf_val);
							light_pdf_val /= float(scene.sphere_light_count);

							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							if (cos_surf <= 0.0 || light_pdf_val <= 0.0) continue;

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float3 direct = nee_contribution(mat, mat_kind, wo, shading_normal, shading_tangent, light_dir, cos_surf, light_pdf_val, sl.emission.xyz);
								accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
							}
						}
					}

					// Disc lights
					if (scene.disc_light_count > 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							uint li = min(uint(rng_float(seed) * float(scene.disc_light_count)), uint(scene.disc_light_count - 1));
							GPUDiscLight dl = disc_lights[li];

							float3 light_normal;
							float light_dist, light_pdf_val;
							float3 light_dir = sample_disc_light(dl, hit_point, seed, light_dist, light_normal, light_pdf_val);
							light_pdf_val /= float(scene.disc_light_count);

							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							if (cos_surf <= 0.0 || light_pdf_val <= 0.0) continue;

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float3 direct = nee_contribution(mat, mat_kind, wo, shading_normal, shading_tangent, light_dir, cos_surf, light_pdf_val, dl.emission.xyz);
								accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
							}
						}
					}

					// Cylinder lights
					if (scene.cylinder_light_count > 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							uint li = min(uint(rng_float(seed) * float(scene.cylinder_light_count)), uint(scene.cylinder_light_count - 1));
							GPUCylinderLight cl = cylinder_lights[li];

							float3 light_normal;
							float light_dist, light_pdf_val;
							float3 light_dir = sample_cylinder_light(cl, hit_point, seed, light_dist, light_normal, light_pdf_val);
							light_pdf_val /= float(scene.cylinder_light_count);

							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							if (cos_surf <= 0.0 || light_pdf_val <= 0.0) continue;

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float3 direct = nee_contribution(mat, mat_kind, wo, shading_normal, shading_tangent, light_dir, cos_surf, light_pdf_val, cl.emission.xyz);
								accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
							}
						}
					}

					// Punctual (point / spot / distant) delta lights. Evaluated
					// once each, with no MIS.
					if (scene.punctual_light_count > 0) {
						light_types_sampled++;
						for (int li = 0; li < scene.punctual_light_count; li++) {
							GPUPunctualLight pl = punctual_lights[li];
							int kind = int(pl.params.x);

							float3 light_dir;
							float light_dist;
							float3 radiance;
							if (kind == 2) {
								// Distant: direction the light travels is pl.direction.
								light_dir = -normalize(pl.direction.xyz);
								light_dist = INFINITY;
								radiance = pl.emission.xyz;
							} else {
								float3 to_light = pl.position.xyz - hit_point;
								light_dist = length(to_light);
								if (light_dist <= 0.0) continue;
								light_dir = to_light / light_dist;
								radiance = pl.emission.xyz / (light_dist * light_dist);
								if (kind == 1) {
									// Spot cone falloff.
									float3 axis = normalize(pl.direction.xyz);
									float cos_angle = dot(axis, -light_dir);
									float cos_inner = pl.params.y;
									float cos_outer = pl.params.z;
									float atten = 0.0;
									if (cos_angle >= cos_inner) atten = 1.0;
									else if (cos_angle > cos_outer) {
										float t = (cos_angle - cos_outer) / (cos_inner - cos_outer);
										atten = t * t * (3.0 - 2.0 * t);
									}
									radiance *= atten;
								}
							}

							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							if (cos_surf <= 0.0) continue;

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = (kind == 2) ? INFINITY : max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float3 direct = nee_contribution_delta(mat, mat_kind, wo, shading_normal, shading_tangent, light_dir, cos_surf, radiance);
								accumulated += ray_color * direct;
							}
						}
					}

					// Environment (dome) NEE.
					if (scene.has_env != 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							float3 eradiance;
							float epdf;
							float3 light_dir = env_sample(scene, env_pixels, env_marginal, env_conditional, seed, eradiance, epdf);
							if (epdf <= 0.0) continue;

							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							if (cos_surf <= 0.0) continue;

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = INFINITY;

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float3 direct = nee_contribution(mat, mat_kind, wo, shading_normal, shading_tangent, light_dir, cos_surf, epdf, eradiance);
								accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
							}
						}
					}
				}
			}

			if (
				depth == 0 &&
				(scene.debug_mode == 5 ||
				 scene.debug_mode == 7 ||
				 scene.debug_mode == 8)
			) {
				break;
			}

			// Scatter
			if (mat_kind == 0) {
				// Lambertian (pure diffuse) — GI cache and photons apply.
				if (scene.gi_cache_enabled && depth > 1 && !last_was_delta) {
					int cache_sample_count = 0;
					float cache_rel_stddev = 1.0e6;
					float3 cached = gi_cache_query(
						gi_cache, gi_grid_cells, gi_grid_counts,
						GI_GRID_SIZE,
						hit_point, shading_normal,
						scene.gi_cache_distance,
						scene.gi_cache_normal_angle,
						cache_sample_count,
						cache_rel_stddev
					);
					float cached_lum = luminance(cached);
					bool cache_accepted =
						cached_lum > 0.0 &&
						cache_sample_count >= GI_CACHE_MIN_SAMPLES &&
						cache_rel_stddev <= GI_CACHE_MAX_REL_STDDEV;

					if (scene.debug_mode == 12) {
						accumulated = cache_sample_count > 0 ?
							debug_heat(float(cache_sample_count) / float(GI_CACHE_MIN_SAMPLES * 2)) :
							float3(0.5, 0.0, 0.0);
						debug_found = true;
						ray_color = 0.0;
						cache_pending = 0;
						break;
					}

					if (scene.debug_mode == 13) {
						float confidence = 1.0 - clamp(cache_rel_stddev / GI_CACHE_MAX_REL_STDDEV, 0.0, 1.0);
						accumulated = cache_sample_count > 0 ? debug_heat(confidence) : float3(0.5, 0.0, 0.0);
						debug_found = true;
						ray_color = 0.0;
						cache_pending = 0;
						break;
					}

					if (cache_accepted) {
						if (scene.debug_mode == 10) {
							accumulated = debug_heat(cached_lum * 0.1);
							debug_found = true;
							ray_color = 0.0;
							cache_pending = 0;
							break;
						}
						accumulated += ray_color * cached * mat.albedo.xyz * INV_PI;
						ray_color = 0.0;
						break;
					}
				}

				// Photon mapping: the global photon map supplies indirect
				// diffuse illumination. When enough photons are found we take
				// the density estimate as the indirect radiance and terminate
				// the diffuse walk here (biased radiance estimate, no final
				// gather) — otherwise fall through to brute-force path tracing
				// so photon-sparse regions still converge.
				if (scene.photon_enabled && depth > 1) {
					int pc = min(int(atomic_load_explicit(photon_counter, memory_order_relaxed)), PHOTON_MAX_COUNT);
					if (pc > 0) {
						int photon_found = 0;
						float3 photon_contrib = photon_query(
							photons, photon_grid_offsets, photon_grid_counts,
							photon_grid_sorted,
							hit_point, shading_normal,
							scene.photon_radius, mat.albedo.xyz,
							photon_found
						);
						if (scene.debug_mode == 11) {
							accumulated = photon_found >= PHOTON_MIN_FOUND ?
								debug_heat(luminance(photon_contrib) * 0.5) :
								float3(0.0);
							debug_found = true;
							ray_color = 0.0;
							cache_pending = 0;
							break;
						}
						if (photon_found >= PHOTON_MIN_FOUND) {
							accumulated += ray_color * photon_contrib;
							ray_color = 0.0;
							cache_pending = 0;
							break;
						}
					}
				}

				float bsdf_pdf = 0.0;
				float3 scatter_dir = cosine_sample_hemisphere(shading_normal, seed, bsdf_pdf);
				if (near_zero(scatter_dir) || bsdf_pdf <= 0.0) scatter_dir = shading_normal;
				r.origin = hit_point;
				r.direction = scatter_dir;
				r.min_distance = 0.001;
				ray_color *= mat.albedo.xyz;
				last_bsdf_pdf = max(bsdf_pdf, 0.0);
				last_was_delta = false;

				if (scene.gi_cache_enabled && depth > 1) {
					cache_p_pos = hit_point;
					cache_p_normal = shading_normal;
					cache_p_throughput = ray_color;
					cache_p_accum_before = accum_at_hit;
					cache_pending = 1;
				}
			} else if (mat_kind == 3) {
				// Principled BSDF (GGX specular + Fresnel + diffuse).
				// GI cache and photons are skipped for glossy surfaces;
				// the BSDF's diffuse component is still added to the path
				// via the normal scatter.
				//
				// `glossy_bias` damps the effective roughness for a
				// cheaper, blurrier reflection. 0 = full GGX, 1 = flat mirror.
				float3 wo = normalize(-r.direction);

				// Glass lobe: with probability spec_trans the surface transmits
				// (microfacet reflect/refract). The refracted ray goes below the
				// surface, so this is handled before — and separate from — the
				// reflection lobes' cos_i>0 requirement.
				float spec_trans = mat.params3.x;
				if (spec_trans > 0.0 && rng_float(seed) < spec_trans) {
					float3 gwi, gtp;
					bool gok = principled_sample_glass(mat, wo, shading_normal, front_face, seed, gwi, gtp);
					if (!gok) {
						gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
							scene.gi_cache_distance,
							cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
							accumulated);
						break;
					}
					r.origin = hit_point;
					r.direction = gwi;
					r.min_distance = 0.001;
					ray_color *= gtp;
					last_bsdf_pdf = 0.0;
					// Treat as a specular event (skip the GI cache); rough glass
					// isn't strictly delta but is handled as such here.
					last_was_delta = true;
				} else {
				float3 f = float3(0.0);
				float bsdf_pdf = 0.0;
				float eff_roughness = mat.params0.w;
				if (scene.glossy_bias > 0.0) {
					eff_roughness = clamp(mat.params0.w * (1.0 - scene.glossy_bias) + scene.glossy_bias, 0.0, 1.0);
					GPUMaterial biased_mat = mat;
					biased_mat.params0.w = eff_roughness;
					principled_sample(biased_mat, wo, shading_normal, shading_tangent, seed, r.direction, f, bsdf_pdf);
				} else {
					principled_sample(mat, wo, shading_normal, shading_tangent, seed, r.direction, f, bsdf_pdf);
				}
				float cos_i = dot(r.direction, shading_normal);
				if (bsdf_pdf <= 0.0 || cos_i <= 0.0) {
					// The sampled direction went below the horizon. Terminate
					// the path, but keep whatever radiance it already carried:
					// zeroing `accumulated` here would throw away every NEE
					// contribution gathered on the way in. With normal maps a
					// degenerate sample is common, so discarding turned the
					// surface black.
					gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
						scene.gi_cache_distance,
						cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
						accumulated);
					break;
				}
				r.origin = hit_point;
				r.min_distance = 0.001;
				// Throughput = f * cos_i / pdf (cancel the pdf from the
				// next bounce's PDF). This absorbs the BSDF evaluation.
				ray_color *= f * cos_i / bsdf_pdf;
				last_bsdf_pdf = bsdf_pdf;
				// Treat very low roughness as a delta-like path for the
				// purpose of skipping the GI cache.
				float eff_alpha_sq = pbr_alpha_sq(eff_roughness);
				last_was_delta = eff_alpha_sq < 1.0e-4;
				}
			} else if (mat_kind == 1) {
				// Metal
				float3 reflected = reflect(r.direction, shading_normal);
				// Guide passes follow a clean mirror direction (no fuzz) so the
				// guide isn't averaged over scattered reflection lobes.
				float fuzz = (scene.debug_mode >= 20 && scene.debug_mode <= 22) ? 0.0 : min(mat.params0.y, 1.0);
				r.origin = hit_point;
				r.direction = reflected + fuzz * rng_in_unit_sphere(seed);
				r.min_distance = 0.001;
				if (dot(r.direction, shading_normal) <= 0.0) {
					// Fuzz scattered the reflection below the surface: absorb
					// the ray, but keep the radiance accumulated so far.
					gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
						scene.gi_cache_distance,
						cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
						accumulated);
					break;
				}
				ray_color *= mat.albedo.xyz;
				last_bsdf_pdf = 0.0;
				last_was_delta = true;
			} else if (mat_kind == 2) {
				// Dielectric
				float ir = mat.params0.z;
				float refraction_ratio = front_face ? (1.0 / ir) : ir;
				float3 unit_dir = normalize(r.direction);
				float cos_theta = min(dot(-unit_dir, shading_normal), 1.0);
				float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
				bool cannot_refract = refraction_ratio * sin_theta > 1.0;
				bool guide_pass = scene.debug_mode >= 20 && scene.debug_mode <= 22;
				float3 dir;
				if (cannot_refract) {
					dir = reflect(unit_dir, shading_normal);
				} else if (guide_pass) {
					// Guide passes follow the transmitted ray, the dominant
					// visible content through glass, rather than a random
					// Fresnel reflect/refract split that would blur the guide.
					dir = refract(unit_dir, shading_normal, refraction_ratio);
				} else if (schlick_reflectance(cos_theta, refraction_ratio) > rng_float(seed)) {
					dir = reflect(unit_dir, shading_normal);
				} else {
					dir = refract(unit_dir, shading_normal, refraction_ratio);
				}
				r.origin = hit_point;
				r.direction = dir;
				r.min_distance = 0.001;
				last_bsdf_pdf = 0.0;
				last_was_delta = true;
			}

			// Firefly clamping
			if (scene.max_radiance > 0.0) {
				float lum = luminance(ray_color);
				if (lum > scene.max_radiance) {
					ray_color *= (scene.max_radiance / lum);
				}
			}
		}
		// Path reached max depth — flush any deferred cache point
		gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
			scene.gi_cache_distance,
			cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
			accumulated);

		if (
			(scene.debug_mode == 10 ||
			 scene.debug_mode == 11 ||
			 scene.debug_mode == 12 ||
			 scene.debug_mode == 13) &&
			!debug_found
		) {
			accumulated = 0.0;
		}

		pixel_color += accumulated;
	}

	pixel_color /= float(scene.samples_per_pixel);
	// Write raw linear scene radiance — no tonemap. A tonemap is a display look,
	// not part of the render: baking Reinhard here desaturated highlights and
	// flattened contrast (washed-out vs. Karma) and left EXR clamped to [0,1)
	// instead of true HDR. The host clamps + applies the sRGB OETF for the 8-bit
	// PNG/viewport buffer; EXR takes this linear buffer as-is.
	output[pixel_idx] = float4(pixel_color, 1.0);
}

// ── Photon Emission Kernel ──────────────────────────────────────────────────

kernel void photonEmitKernel(
	uint                               tid            [[thread_position_in_grid]],
	constant GPUSceneData&             scene          [[buffer(0)]],
	device const GPUMaterial*          materials      [[buffer(1)]],
	primitive_acceleration_structure   accel          [[buffer(3)]],
	device const TriVertex*            vertices       [[buffer(4)]],
	device const uint*                 indices        [[buffer(5)]],
	device const GPULightTriangle*     tri_lights     [[buffer(6)]],
	device const GPUQuadLight*         quad_lights    [[buffer(7)]],
	device const GPUSphereLight*       sphere_lights  [[buffer(8)]],
	device const int*                  mat_indices    [[buffer(9)]],
	device Photon*                     photons        [[buffer(14)]],
	device atomic_int*                 photon_counter [[buffer(15)]]
) {
	if (tid >= uint(scene.photon_count) || scene.photon_count <= 0) return;

	uint seed = scene.seed + tid * 1973u;

	int tlc = scene.tri_light_count;
	int qlc = scene.quad_light_count;
	int slc = scene.sphere_light_count;
	int total_lights = tlc + qlc + slc;
	if (total_lights == 0) return;

	int li = int(tid) % total_lights;

	float3 origin, lnormal, emission;
	float area = 0.0;

	if (li < tlc) {
		GPULightTriangle lt = tri_lights[li];
		float r1 = rng_float(seed);
		float r2 = rng_float(seed);
		float sqrt_r1 = sqrt(r1);
		origin = (1.0 - sqrt_r1) * lt.p0.xyz + (sqrt_r1 * (1.0 - r2)) * lt.p1.xyz + (sqrt_r1 * r2) * lt.p2.xyz;
		lnormal = normalize(cross(lt.p1.xyz - lt.p0.xyz, lt.p2.xyz - lt.p0.xyz));
		emission = lt.emission.xyz;
		area = triangle_area(lt.p0.xyz, lt.p1.xyz, lt.p2.xyz);
	} else if (li < tlc + qlc) {
		GPUQuadLight ql = quad_lights[li - tlc];
		float r1 = rng_float(seed);
		float r2 = rng_float(seed);
		origin = ql.position.xyz + r1 * ql.u.xyz + r2 * ql.v.xyz;
		lnormal = normalize(cross(ql.u.xyz, ql.v.xyz));
		emission = ql.emission.xyz;
		area = length(cross(ql.u.xyz, ql.v.xyz));
	} else {
		GPUSphereLight sl = sphere_lights[li - tlc - qlc];
		float3 dir = rng_unit_vector(seed);
		origin = sl.position.xyz + dir * sl.radius;
		lnormal = dir;
		emission = sl.emission.xyz;
		area = 4.0 * PI * sl.radius * sl.radius;
	}

	// Cosine-weighted emission direction
	float pdf_dir;
	float3 dir = cosine_sample_hemisphere(lnormal, seed, pdf_dir);

	float3 power = emission * (PI * area * float(total_lights)) / float(scene.photon_count);

	ray r;
	r.origin = origin + dir * 0.002;
	r.direction = dir;
	r.min_distance = 0.001;
	r.max_distance = INFINITY;

	for (int bounce = 0; bounce < scene.photon_max_bounces; bounce++) {
		intersector<> i;
		i.assume_geometry_type(geometry_type::triangle);
		auto result = i.intersect(r, accel);

		if (result.type == intersection_type::none || result.distance >= INFINITY || result.distance <= 0.0) break;

		uint pid = result.primitive_id;
		float hit_dist = result.distance;
		float3 hit_point = r.origin + hit_dist * r.direction;

		uint base_idx = pid * 3;
		uint i0 = indices[base_idx];
		uint i1 = indices[base_idx + 1];
		uint i2 = indices[base_idx + 2];

		float3 p0 = vertices[i0].position.xyz;
		float3 p1 = vertices[i1].position.xyz;
		float3 p2 = vertices[i2].position.xyz;

		float3 edge1 = p1 - p0;
		float3 edge2 = p2 - p0;
		float3 geom_normal = normalize(cross(edge1, edge2));
		bool front_face = dot(r.direction, geom_normal) < 0.0;

		float3 edge_cross = cross(edge1, edge2);
		float full_area = length(edge_cross);
		float bu = length(cross(p1 - hit_point, p2 - hit_point)) / max(full_area, 1e-12);
		float bv = length(cross(p2 - hit_point, p0 - hit_point)) / max(full_area, 1e-12);
		float bw = length(cross(p0 - hit_point, p1 - hit_point)) / max(full_area, 1e-12);
		float sum = bu + bv + bw;
		bu /= sum; bv /= sum; bw /= sum;

		float3 vn0 = vertices[i0].normal.xyz;
		float3 vn1 = vertices[i1].normal.xyz;
		float3 vn2 = vertices[i2].normal.xyz;
		float3 shading_normal = normalize(bu * vn0 + bv * vn1 + bw * vn2);
		if (!front_face) shading_normal = -shading_normal;

		int midx = mat_indices[pid];
		GPUMaterial mat = materials[midx];
		int mat_kind = int(mat.params0.x);

		// Apply glossy_bias and roughness cutoff the same way the main
		// kernel does, so photon paths match what the camera sees.
		if (mat_kind == 3) {
			if (scene.glossy_bias > 0.0) {
				mat.params0.w = clamp(mat.params0.w * (1.0 - scene.glossy_bias) + scene.glossy_bias, 0.0, 1.0);
			}
			if (scene.roughness_cutoff > 0.0) {
				if (mat.params0.w > scene.roughness_cutoff) {
					mat_kind = 0;
				}
			}
		}

		// For Principled, the metal/dielectric split comes from
		// `metallic` rather than the material kind: a fully metallic
		// Principled should reflect like a mirror until first diffuse
		// hit, while a dielectric Principled should still bounce through
		// a thin material layer.
		if (mat_kind == 3 && mat.params1.x >= 1.0) {
			// Treat as metal: GGX-style reflect. For a fully metallic
			// surface the diffuse lobe is zero, so the path is purely
			// specular until a non-metallic hit.
			float roughness = mat.params0.w;
			float alpha = max(roughness, 0.0);
			float3 tangent = make_tangent(shading_normal);
			float3 bitangent = cross(shading_normal, tangent);
			float3 wo = normalize(-r.direction);
			float cos_o = dot(wo, shading_normal);
			float3 wo_local = float3(dot(wo, tangent), dot(wo, bitangent), cos_o);
			float u1 = rng_float(seed);
			float u2 = rng_float(seed);
			float3 wh_local = sample_ggx_vndf_local(wo_local, alpha, u1, u2);
			float3 wh_world = normalize(wh_local.x * tangent + wh_local.y * bitangent + wh_local.z * shading_normal);
			float3 reflected = reflect_over(r.direction, wh_world);
			r.origin = hit_point;
			r.direction = reflected;
			r.min_distance = 0.001;
			if (dot(r.direction, shading_normal) <= 0.0) break;
			// Power scaling: F0 for a metal is the base color. We use
			// the F0 of the material at the current angle.
			float3 f0 = principled_f0(mat);
			float wo_dot_wh = max(dot(wo, wh_world), 0.0);
			float3 F = schlick_fresnel_color(wo_dot_wh, f0);
			power *= F;
		} else if (mat_kind == 0 || mat_kind == 3) {
			// Diffuse — store photon
			int idx = atomic_fetch_add_explicit(photon_counter, 1, memory_order_relaxed);
			if (idx < PHOTON_MAX_COUNT) {
				float3 cell_coord = floor(hit_point / scene.photon_radius);
				photons[idx].position = float4(hit_point, cell_coord.x);
				photons[idx].incident = float4(-r.direction, cell_coord.y);
				photons[idx].power = float4(power, cell_coord.z);
			}
			break;
		} else if (mat_kind == 1) {
			// Metal — reflect
			float3 reflected = reflect(r.direction, shading_normal);
			float fuzz = min(mat.params0.y, 1.0);
			r.origin = hit_point;
			r.direction = reflected + fuzz * rng_in_unit_sphere(seed);
			r.min_distance = 0.001;
			if (dot(r.direction, shading_normal) <= 0.0) break;
			power *= mat.albedo.xyz;
		} else if (mat_kind == 2) {
			// Dielectric — transmit / reflect
			float ir = mat.params0.z;
			float refraction_ratio = front_face ? (1.0 / ir) : ir;
			float3 unit_dir = normalize(r.direction);
			float cos_theta = min(dot(-unit_dir, shading_normal), 1.0);
			float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
			bool cannot_refract = refraction_ratio * sin_theta > 1.0;
			float3 dir_out;
			if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > rng_float(seed)) {
				dir_out = reflect(unit_dir, shading_normal);
			} else {
				dir_out = refract(unit_dir, shading_normal, refraction_ratio);
			}
			r.origin = hit_point;
			r.direction = dir_out;
			r.min_distance = 0.001;
		} else if (mat_kind == 4) {
			break;
		}

		// Russian roulette
		float lum = luminance(power);
		if (lum < 0.001) {
			if (rng_float(seed) > lum / 0.001) break;
			power *= 0.001 / lum;
		}
	}
}

// ── Photon Grid Build (counting sort) ───────────────────────────────────────
//
// The grid is built in two GPU passes with a CPU exclusive prefix sum in
// between (see render_gpu_darwin.odin):
//   1. photonCountKernel  — per-photon bucket + per-bucket count.
//   2. [CPU] prefix sum    — grid_counts → grid_offsets.
//   3. photonScatterKernel — place each photon index into grid_sorted at
//                            grid_offsets[bucket] + running fill cursor.
// This keeps *all* photons (no fixed per-cell cap), so the density estimate
// sees the true photon density.

static int ph_grid_cell(float3 pos, float radius) {
	// Must match photon_hash_cell(floor(pos/radius)).
	return photon_hash_cell(floor(pos / radius));
}

kernel void photonCountKernel(
	uint                               tid            [[thread_position_in_grid]],
	device const Photon*               photons        [[buffer(0)]],
	device const atomic_int*           photon_counter [[buffer(1)]],
	device int*                        photon_cell    [[buffer(2)]],
	device atomic_int*                 grid_counts    [[buffer(3)]],
	constant GPUSceneData&             scene          [[buffer(4)]]
) {
	int count = min(int(atomic_load_explicit(photon_counter, memory_order_relaxed)), PHOTON_MAX_COUNT);
	if (tid >= uint(count) || scene.photon_radius <= 0.0) return;

	int cell = ph_grid_cell(photons[tid].position.xyz, scene.photon_radius);
	photon_cell[tid] = cell;
	atomic_fetch_add_explicit(&grid_counts[cell], 1, memory_order_relaxed);
}

kernel void photonScatterKernel(
	uint                               tid            [[thread_position_in_grid]],
	device const atomic_int*           photon_counter [[buffer(1)]],
	device const int*                  photon_cell    [[buffer(2)]],
	device const int*                  grid_offsets   [[buffer(3)]],
	device atomic_int*                 grid_fill      [[buffer(4)]],
	device int*                        grid_sorted    [[buffer(5)]],
	constant GPUSceneData&             scene          [[buffer(6)]]
) {
	int count = min(int(atomic_load_explicit(photon_counter, memory_order_relaxed)), PHOTON_MAX_COUNT);
	if (tid >= uint(count) || scene.photon_radius <= 0.0) return;

	int cell = photon_cell[tid];
	int slot = atomic_fetch_add_explicit(&grid_fill[cell], 1, memory_order_relaxed);
	grid_sorted[grid_offsets[cell] + slot] = int(tid);
}
