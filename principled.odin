package main

import m "core:math/linalg/glsl"

PI_F64 :: 3.14159265358979323846
INV_PI_F64 :: 0.31830988618379067154

// ── GGX helpers ─────────────────────────────────────────────────────────────

// `roughness` is the user-facing value in [0,1]. We square it to get alpha^2
// for the GGX distribution — the "perceptual roughness" mapping recommended
// by Disney / Heitz.
ggx_alpha_sq :: proc(roughness: f64) -> f64 {
	a := clamp(roughness, 0.0, 1.0)
	return a * a
}

// Trowbridge-Reitz / GGX normal distribution.
ggx_D :: proc(cos_h: f64, alpha_sq: f64) -> f64 {
	if cos_h <= 0.0 || alpha_sq <= 0.0 {
		return 0.0
	}
	a2 := alpha_sq
	denom := cos_h * cos_h * (a2 - 1.0) + 1.0
	return a2 / max(PI_F64 * denom * denom, 1.0e-30)
}

// Smith G1 term for a single direction.
ggx_G1 :: proc(cos_v: f64, alpha_sq: f64) -> f64 {
	if cos_v <= 0.0 {
		return 0.0
	}
	a2 := alpha_sq
	cos2 := cos_v * cos_v
	tan2 := (1.0 - cos2) / max(cos2, 1.0e-30)
	return 2.0 / (1.0 + m.sqrt(1.0 + a2 * tan2))
}

// Smith G2 term: G1(wo) * G1(wi).
ggx_G :: proc(cos_o: f64, cos_i: f64, alpha_sq: f64) -> f64 {
	return ggx_G1(cos_o, alpha_sq) * ggx_G1(cos_i, alpha_sq)
}

// Schlick Fresnel for an arbitrary F0. Returns the full spectral Fresnel.
schlick_fresnel :: proc(cos_theta: f64, f0: Color) -> Color {
	ct := clamp(cos_theta, 0.0, 1.0)
	pow5 := (1.0 - ct)
	pow5 = pow5 * pow5
	pow5 = pow5 * pow5 * (1.0 - ct)
	return f0 + (Color{1.0, 1.0, 1.0} - f0) * pow5
}

// ── F0 / F90 helpers ───────────────────────────────────────────────────────

// Dielectric F0 from a "specular" parameter in [0,1] (0.5 → 0.08, 1.0 → 0.16).
dielectric_f0 :: proc(specular: f64) -> f64 {
	return 0.08 * specular
}

// Build the spectral F0 (used for Schlick) from the Principled parameters.
// Metallic materials tint the F0 with the base color (modulated by specular_tint).
principled_f0 :: proc(mat: Material) -> Color {
	dielectric_part := dielectric_f0(mat.specular) * mat.specular_tint
	if mat.metallic >= 1.0 {
		return mat.albedo
	}
	return m.lerp(dielectric_part, mat.albedo, mat.metallic)
}

// F90 is the reflectance at grazing angle. For dielectrics it's 1.0; for
// metals it's the base color (the Schlick curve never reaches 1.0 for metals
// because F0 = F90 for pure metals in this simplified model).
principled_f90 :: proc(mat: Material, f0: Color) -> Color {
	if mat.metallic >= 1.0 {
		return f0
	}
	return Color{1.0, 1.0, 1.0}
}

// ── GGX VNDF importance sampling (Heitz 2017) ──────────────────────────────

// Sample a microfacet normal visible from `wo`. `u` is two uniform samples
// in [0,1). `alpha` is the per-axis squared roughness (we treat the surface
// as isotropic in the base implementation).
sample_ggx_vndf :: proc(wo: Vec3, alpha: f64, u1, u2: f64) -> Vec3 {
	// Stretch wo into the unit sphere
	v := Vec3{wo.x * alpha, wo.y * alpha, wo.z}
	if m.length(v) < 1.0e-12 {
		v = Vec3{0.0, 0.0, 1.0}
	}
	v = m.normalize(v)

	// Build orthonormal basis around v
	t1: Vec3
	if m.abs(v.z) < 0.9999 {
		t1 = m.normalize(m.cross(v, Vec3{0.0, 1.0, 0.0}))
	} else {
		t1 = Vec3{1.0, 0.0, 0.0}
	}
	t2 := m.cross(t1, v)

	// Sample point on the visible disk of normal vectors
	a := 1.0 / (1.0 + v.z)
	r := m.sqrt(u1)
	phi: f64
	if u2 < a {
		phi = u2 / a * PI_F64
	} else {
		phi = PI_F64 + (u2 - a) / (1.0 - a) * PI_F64
	}
	s1 := r * m.cos(phi)
	s2 := r * m.sin(phi) * m.sqrt(max(1.0 - s1 * s1, 0.0))

	// Reproject onto the hemisphere
	n_h := s1 * t1 + s2 * t2 + m.sqrt(max(1.0 - s1 * s1 - s2 * s2, 0.0)) * v

	// Unstretch
	return m.normalize(Vec3{n_h.x * alpha, n_h.y * alpha, n_h.z})
}

// Sample a microfacet normal from the NDF directly (used for NEE-style
// evaluation paths where the sample is not constrained to be visible).
sample_ggx_ndf :: proc(alpha_sq: f64, u1, u2: f64) -> Vec3 {
	r1 := clamp(u1, 1.0e-6, 1.0 - 1.0e-6)
	cos_h := m.sqrt((1.0 - r1) / (1.0 + r1 * (alpha_sq - 1.0)))
	sin_h := m.sqrt(max(1.0 - cos_h * cos_h, 0.0))
	phi := 2.0 * PI_F64 * u2
	return Vec3{sin_h * m.cos(phi), sin_h * m.sin(phi), cos_h}
}

// PDF of `wh` under the GGX NDF. Used by the BRDF-side MIS weight.
ggx_pdf :: proc(cos_h: f64, alpha_sq: f64) -> f64 {
	return ggx_D(cos_h, alpha_sq) * cos_h
}

// ── Disney-style evaluate / sample / pdf ───────────────────────────────────

// Reflect `wo` over the surface normal `n` and return the microfacet
// half-vector for the reflection.
reflect_half :: proc(wo: Vec3, n: Vec3) -> Vec3 {
	return m.normalize(-wo + 2.0 * m.dot(wo, n) * n)
}

// Reflection direction from incoming `wi` over half-vector `wh`.
reflect_over :: proc(wi: Vec3, wh: Vec3) -> Vec3 {
	return wi - 2.0 * m.dot(wi, wh) * wh
}

// Evaluate the BSDF (radiance) and the solid-angle sampling PDF for an
// outgoing direction `wo` given an incoming direction `wi`. Both vectors
// point away from the surface.
principled_evaluate :: proc(mat: Material, wo, wi, n: Vec3) -> (f: Color, pdf: f64) {
	cos_o := m.dot(wo, n)
	cos_i := m.dot(wi, n)
	if cos_o <= 0.0 || cos_i <= 0.0 {
		return Color{0.0, 0.0, 0.0}, 0.0
	}
	wh := m.normalize(wo + wi)
	if m.length(wh) < 1.0e-12 {
		return Color{0.0, 0.0, 0.0}, 0.0
	}
	cos_h := m.dot(wh, n)
	if cos_h <= 0.0 {
		return Color{0.0, 0.0, 0.0}, 0.0
	}
	wo_dot_wh := max(m.dot(wo, wh), 0.0)
	if wo_dot_wh <= 0.0 {
		return Color{0.0, 0.0, 0.0}, 0.0
	}

	alpha_sq := ggx_alpha_sq(mat.roughness)
	D := ggx_D(cos_h, alpha_sq)
	G := ggx_G(cos_o, cos_i, alpha_sq)
	f0 := principled_f0(mat)
	F := schlick_fresnel(wo_dot_wh, f0)

	specular := F * (D * G) / (4.0 * cos_o * cos_i)
	diffuse := Color{0.0, 0.0, 0.0}
	if mat.metallic < 1.0 {
		// Lambertian with energy conservation: diffuse weighted by (1 - F)
		diffuse = (1.0 - mat.metallic) * mat.albedo * (Color{1.0, 1.0, 1.0} - F) * INV_PI_F64
	}

	f = diffuse + specular

	// Combined sampling PDF. The diffuse lobe uses cosine sampling
	// (pdf_d = cos_i / PI), the specular lobe uses the visible-normal
	// distribution (pdf_s = G1(cos_o) * vndf_pdf(wh) / (4 * wo . wh)).
	// We use a 50/50 weight for the MIS across lobes — good enough for
	// the bias-vs-quality budget.
	weight_spec := 0.5
	weight_diff := 1.0 - weight_spec

	vndf := G / ggx_G1(cos_o, alpha_sq)  // = G1(cos_i)
	pdf_s := vndf * D / (4.0 * wo_dot_wh)
	pdf_d := cos_i * INV_PI_F64

	pdf = weight_spec * pdf_s + weight_diff * pdf_d
	return f, pdf
}

// Convenience wrapper: PDF at a direction for a Principled surface.
// Used by NEE to compute MIS weights against the light sampling PDF.
principled_pdf_simple :: proc(mat: Material, wo, wi, n: Vec3) -> f64 {
	cos_o := m.dot(wo, n)
	cos_i := m.dot(wi, n)
	if cos_o <= 0.0 || cos_i <= 0.0 {
		return 0.0
	}
	wh := m.normalize(wo + wi)
	if m.length(wh) < 1.0e-12 {
		return 0.0
	}
	cos_h := m.dot(wh, n)
	wo_dot_wh := max(m.dot(wo, wh), 0.0)
	if cos_h <= 0.0 || wo_dot_wh <= 0.0 {
		return 0.0
	}
	alpha_sq := ggx_alpha_sq(mat.roughness)
	D := ggx_D(cos_h, alpha_sq)
	G1_o := ggx_G1(cos_o, alpha_sq)
	pdf_s := G1_o * D / (4.0 * wo_dot_wh)
	pdf_d := cos_i * INV_PI_F64
	return 0.5 * pdf_s + 0.5 * pdf_d
}

// Sample a direction and return (f, pdf) for the next ray.
principled_sample :: proc(mat: Material, wo, n: Vec3, rng: ^Rng) -> (wi: Vec3, f: Color, pdf: f64) {
	cos_o := m.dot(wo, n)
	if cos_o <= 0.0 {
		return Vec3{0.0, 0.0, 1.0}, Color{0.0, 0.0, 0.0}, 0.0
	}

	alpha_sq := ggx_alpha_sq(mat.roughness)
	alpha := m.sqrt(max(alpha_sq, 0.0))

	// Stochastic lobe selection: pick specular with probability proportional
	// to its expected reflectance, otherwise diffuse.
	f0 := principled_f0(mat)
	f0_lum := 0.2126 * f0.x + 0.7152 * f0.y + 0.0722 * f0.z
	metallic := clamp(mat.metallic, 0.0, 1.0)
	// For metals the diffuse lobe is zero, so we always sample specular.
	if metallic >= 1.0 {
		wi, f, pdf = sample_specular_lobe(mat, wo, n, alpha, alpha_sq, cos_o, rng)
		return wi, f, pdf
	}
	// Otherwise pick specular with probability proportional to F0 vs (1-metallic).
	p_spec := clamp(f0_lum / max(f0_lum + (1.0 - metallic), 1.0e-3), 0.0, 1.0)
	if rng_f64(rng) < p_spec {
		wi_spec, f_spec, pdf_s := sample_specular_lobe(mat, wo, n, alpha, alpha_sq, cos_o, rng)
		// Combined PDF: weight specular and diffuse lobe pdfs.
		cos_i := m.dot(wi_spec, n)
		pdf_d := cos_i * INV_PI_F64 if cos_i > 0.0 else 0.0
		pdf = p_spec * pdf_s + (1.0 - p_spec) * pdf_d
		return wi_spec, f_spec, pdf
	} else {
		wi_diff, f_diff, pdf_d := sample_diffuse_lobe(mat, wo, n, rng)
		cos_i := m.dot(wi_diff, n)
		// Need to compute the specular pdf at this direction.
		wh := m.normalize(wo + wi_diff)
		wo_dot_wh := max(m.dot(wo, wh), 0.0)
		cos_h := m.dot(wh, n)
		if cos_h <= 0.0 || wo_dot_wh <= 0.0 {
			return wi_diff, f_diff, pdf_d
		}
		D := ggx_D(cos_h, alpha_sq)
		G1_o := ggx_G1(cos_o, alpha_sq)
		pdf_s := G1_o * D / (4.0 * wo_dot_wh)
		pdf = p_spec * pdf_s + (1.0 - p_spec) * pdf_d
		return wi_diff, f_diff, pdf
	}
}

sample_specular_lobe :: proc(mat: Material, wo, n: Vec3, alpha, alpha_sq, cos_o: f64, rng: ^Rng) -> (wi: Vec3, f: Color, pdf: f64) {
	u1 := rng_f64(rng)
	u2 := rng_f64(rng)
	// Build a tangent frame for the surface.
	tangent := make_tangent_vec(n)
	bitangent := m.cross(n, tangent)
	// Project wo into tangent space (z = n).
	wo_local := Vec3{m.dot(wo, tangent), m.dot(wo, bitangent), cos_o}
	wh_local := sample_ggx_vndf(wo_local, alpha, u1, u2)
	wh_world := m.normalize(wh_local.x * tangent + wh_local.y * bitangent + wh_local.z * n)

	wi = reflect_over(wo, wh_world)
	cos_i := m.dot(wi, n)
	if cos_i <= 0.0 {
		return Vec3{0.0, 0.0, 1.0}, Color{0.0, 0.0, 0.0}, 0.0
	}
	wo_dot_wh := max(m.dot(wo, wh_world), 0.0)
	cos_h := m.dot(wh_world, n)

	D := ggx_D(cos_h, alpha_sq)
	G := ggx_G(cos_o, cos_i, alpha_sq)
	f0 := principled_f0(mat)
	F := schlick_fresnel(wo_dot_wh, f0)
	specular := F * (D * G) / (4.0 * cos_o * cos_i)
	diffuse := Color{0.0, 0.0, 0.0}
	if mat.metallic < 1.0 {
		diffuse = (1.0 - mat.metallic) * mat.albedo * (Color{1.0, 1.0, 1.0} - F) * INV_PI_F64
	}
	f = diffuse + specular

	G1_o := ggx_G1(cos_o, alpha_sq)
	pdf_s := G1_o * D / (4.0 * wo_dot_wh)
	cos_i_clamped := max(cos_i, 0.0)
	pdf_d := cos_i_clamped * INV_PI_F64
	pdf = 0.5 * pdf_s + 0.5 * pdf_d
	return wi, f, pdf
}

sample_diffuse_lobe :: proc(mat: Material, wo, n: Vec3, rng: ^Rng) -> (wi: Vec3, f: Color, pdf: f64) {
	// Cosine-weighted hemisphere sampling
	cos_o := max(m.dot(wo, n), 0.0)
	u1 := rng_f64(rng)
	u2 := rng_f64(rng)
	r := m.sqrt(u1)
	phi := 2.0 * PI_F64 * u2
	x := r * m.cos(phi)
	y := r * m.sin(phi)
	z := m.sqrt(max(1.0 - u1, 0.0))
	tangent := make_tangent_vec(n)
	bitangent := m.cross(n, tangent)
	wi = m.normalize(x * tangent + y * bitangent + z * n)
	cos_i := m.dot(wi, n)
	if cos_i <= 0.0 {
		return Vec3{0.0, 0.0, 1.0}, Color{0.0, 0.0, 0.0}, 0.0
	}

	// Evaluate f and pdf at the sampled direction
	wh := m.normalize(wo + wi)
	wo_dot_wh := max(m.dot(wo, wh), 0.0)
	cos_h := m.dot(wh, n)
	alpha_sq := ggx_alpha_sq(mat.roughness)
	f0 := principled_f0(mat)
	D := ggx_D(cos_h, alpha_sq) if cos_h > 0.0 else 0.0
	G := ggx_G(cos_o, cos_i, alpha_sq)
	F := schlick_fresnel(wo_dot_wh, f0) if wo_dot_wh > 0.0 else f0

	specular := F * (D * G) / (4.0 * cos_o * cos_i) if cos_o > 0.0 && cos_i > 0.0 else Color{0.0, 0.0, 0.0}
	diffuse := Color{0.0, 0.0, 0.0}
	if mat.metallic < 1.0 {
		diffuse = (1.0 - mat.metallic) * mat.albedo * (Color{1.0, 1.0, 1.0} - F) * INV_PI_F64
	}
	f = diffuse + specular

	G1_o := ggx_G1(cos_o, alpha_sq)
	pdf_s := G1_o * D / (4.0 * wo_dot_wh) if wo_dot_wh > 0.0 else 0.0
	pdf_d := cos_i * INV_PI_F64
	pdf = 0.5 * pdf_s + 0.5 * pdf_d
	return wi, f, pdf
}
