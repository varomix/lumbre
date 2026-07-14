package lumbre_core

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

// ── Anisotropic GGX (Phase B) ───────────────────────────────────────────────
// All operate in the surface tangent frame: a vector's components are
// (x along tangent, y along bitangent, z along normal). With ax == ay these
// reduce *exactly* to the isotropic forms above, which is why the isotropic
// render path stays byte-identical (it is only ever entered when
// anisotropic == 0). See plans/PRINCIPLED_BSDF.md Phase B.

// Per-axis GGX alphas from roughness + anisotropic. In this codebase the GGX
// alpha is the roughness itself (ggx_alpha_sq = roughness^2), so the Disney
// aspect stretch is applied to roughness directly. anisotropic == 0 -> ax == ay.
aniso_alphas :: proc(mat: Material) -> (ax, ay: f64) {
	a := clamp(mat.roughness, 0.0, 1.0)
	aniso := clamp(mat.anisotropic, 0.0, 1.0)
	aspect := m.sqrt(1.0 - 0.9 * aniso)
	ax = max(a / aspect, 1.0e-4)
	ay = max(a * aspect, 1.0e-4)
	return
}

// Anisotropic GGX NDF for a half-vector in tangent-local coords.
ggx_D_aniso :: proc(h: Vec3, ax, ay: f64) -> f64 {
	if h.z <= 0.0 {
		return 0.0
	}
	d := (h.x * h.x) / (ax * ax) + (h.y * h.y) / (ay * ay) + h.z * h.z
	return 1.0 / max(PI_F64 * ax * ay * d * d, 1.0e-30)
}

// Smith Lambda for the anisotropic GGX, `w` in tangent-local coords.
ggx_lambda_aniso :: proc(w: Vec3, ax, ay: f64) -> f64 {
	if w.z <= 0.0 {
		return 0.0
	}
	t2 := (ax * ax * w.x * w.x + ay * ay * w.y * w.y) / (w.z * w.z)
	return 0.5 * (-1.0 + m.sqrt(1.0 + t2))
}

ggx_G1_aniso :: proc(w: Vec3, ax, ay: f64) -> f64 {
	return 1.0 / (1.0 + ggx_lambda_aniso(w, ax, ay))
}

ggx_G_aniso :: proc(wo, wi: Vec3, ax, ay: f64) -> f64 {
	return ggx_G1_aniso(wo, ax, ay) * ggx_G1_aniso(wi, ax, ay)
}

// Heitz 2018 anisotropic VNDF sample. `wo_local` is the outgoing direction in
// tangent-local coords; returns the sampled half-vector, also tangent-local.
sample_ggx_vndf_aniso :: proc(wo_local: Vec3, ax, ay: f64, u1, u2: f64) -> Vec3 {
	vh := m.normalize(Vec3{ax * wo_local.x, ay * wo_local.y, wo_local.z})
	lensq := vh.x * vh.x + vh.y * vh.y
	t1: Vec3 = ---
	if lensq > 1.0e-12 {
		t1 = Vec3{-vh.y, vh.x, 0.0} / m.sqrt(lensq)
	} else {
		t1 = Vec3{1.0, 0.0, 0.0}
	}
	t2v := m.cross(vh, t1)
	r := m.sqrt(u1)
	phi := 2.0 * PI_F64 * u2
	p1 := r * m.cos(phi)
	p2 := r * m.sin(phi)
	s := 0.5 * (1.0 + vh.z)
	p2 = (1.0 - s) * m.sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2
	nh := p1 * t1 + p2 * t2v + m.sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0)) * vh
	return m.normalize(Vec3{ax * nh.x, ay * nh.y, max(nh.z, 1.0e-6)})
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
//
// The BSDF is a weighted sum of lobes: Burley diffuse + sheen + GGX specular
// + clearcoat (transmission arrives in Phase C). `principled_eval_f` and
// `principled_pdf` are the single source of truth for the value and the
// combined sampling pdf; `principled_sample` picks one lobe, generates a
// direction, then evaluates the *full* f and pdf at it so MIS stays correct.
// See plans/PRINCIPLED_BSDF.md.

pow5 :: proc(x: f64) -> f64 {
	x2 := x * x
	return x2 * x2 * x
}

luminance_color :: proc(c: Color) -> f64 {
	return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

// Schlick Fresnel for a scalar F0 (clearcoat uses F0 = 0.04).
schlick_scalar :: proc(cos_theta: f64, f0: f64) -> f64 {
	return f0 + (1.0 - f0) * pow5(1.0 - clamp(cos_theta, 0.0, 1.0))
}

// Clearcoat's own GGX alpha^2. The coat roughness is floored so an authored
// clearcoat with roughness 0 is a very-glossy coat rather than a delta spike
// the sampler can't hit.
clearcoat_alpha_sq :: proc(mat: Material) -> f64 {
	r := clamp(mat.clearcoat_roughness, 0.03, 1.0)
	return r * r
}

// Reflect `wo` over the surface normal `n` and return the microfacet
// half-vector for the reflection.
reflect_half :: proc(wo: Vec3, n: Vec3) -> Vec3 {
	return m.normalize(-wo + 2.0 * m.dot(wo, n) * n)
}

// Reflect an *incident* direction (pointing toward the surface) about `wh`.
// To mirror a direction that points away from the surface, negate it first.
reflect_over :: proc(incident: Vec3, wh: Vec3) -> Vec3 {
	return incident - 2.0 * m.dot(incident, wh) * wh
}

// Relative selection weights of the three reflection lobes. With clearcoat = 0
// these reduce to the previous (1-metallic) vs f0_lum split, so a material
// without a coat samples exactly as before.
principled_lobe_weights :: proc(mat: Material) -> (w_diff, w_spec, w_cc: f64) {
	f0 := principled_f0(mat)
	metallic := clamp(mat.metallic, 0.0, 1.0)
	spec_trans := clamp(mat.spec_trans, 0.0, 1.0)
	w_diff = (1.0 - metallic) * (1.0 - spec_trans)
	w_spec = luminance_color(f0)
	w_cc = 0.25 * max(mat.clearcoat, 0.0)
	return
}

// Orthonormalized surface tangent: `t` projected into the plane of `n`, with
// a stable fallback when `t` is unusable (parallel to n / zero). Only the
// anisotropic path depends on the tangent *direction*; the isotropic path is
// rotationally symmetric and ignores it.
surface_tangent :: proc(n, t: Vec3) -> Vec3 {
	proj := t - n * m.dot(n, t)
	if m.length(proj) < 1.0e-6 {
		return make_tangent_vec(n)
	}
	return m.normalize(proj)
}

// Main specular lobe's D, G and VNDF sampling-pdf factor. Isotropic
// (byte-identical to Phase A) unless mat.anisotropic > 0, in which case it is
// evaluated in the (t, b, n) tangent frame with per-axis alphas.
principled_specular_dg :: proc(
	mat: Material, wo, wi, wh, n, t: Vec3, cos_o, cos_i, cos_h: f64,
) -> (D, G, pdf_s: f64) {
	if mat.anisotropic <= 0.0 {
		alpha_sq := ggx_alpha_sq(mat.roughness)
		D = ggx_D(cos_h, alpha_sq)
		G = ggx_G(cos_o, cos_i, alpha_sq)
		pdf_s = ggx_G1(cos_o, alpha_sq) * D / (4.0 * cos_o)
		return
	}
	b := m.cross(n, t)
	ax, ay := aniso_alphas(mat)
	wo_l := Vec3{m.dot(wo, t), m.dot(wo, b), cos_o}
	wi_l := Vec3{m.dot(wi, t), m.dot(wi, b), cos_i}
	wh_l := Vec3{m.dot(wh, t), m.dot(wh, b), cos_h}
	D = ggx_D_aniso(wh_l, ax, ay)
	G = ggx_G_aniso(wo_l, wi_l, ax, ay)
	pdf_s = ggx_G1_aniso(wo_l, ax, ay) * D / (4.0 * cos_o)
	return
}

// Full BSDF value for a reflection pair (both vectors above the surface).
// `t` is the surface tangent (only used when the material is anisotropic).
principled_eval_f :: proc(mat: Material, wo, wi, n, t: Vec3) -> Color {
	cos_o := m.dot(wo, n)
	cos_i := m.dot(wi, n)
	if cos_o <= 0.0 || cos_i <= 0.0 {
		return Color{0.0, 0.0, 0.0}
	}
	wh := m.normalize(wo + wi)
	if m.length(wh) < 1.0e-12 {
		return Color{0.0, 0.0, 0.0}
	}
	cos_h := m.dot(wh, n)
	cos_d := max(m.dot(wo, wh), 0.0) // difference angle; = dot(wi, wh)
	if cos_h <= 0.0 || cos_d <= 0.0 {
		return Color{0.0, 0.0, 0.0}
	}
	metallic := clamp(mat.metallic, 0.0, 1.0)

	// Specular (GGX + Schlick metallic Fresnel), anisotropy-aware.
	tan := surface_tangent(n, t)
	D, G, _ := principled_specular_dg(mat, wo, wi, wh, n, tan, cos_o, cos_i, cos_h)
	f0 := principled_f0(mat)
	F := schlick_fresnel(cos_d, f0)
	specular := F * (D * G) / (4.0 * cos_o * cos_i)

	// Burley 2012 diffuse: roughness-dependent retroreflection. Weighted by
	// (1 - F) so the diffuse and specular lobes split the incident energy
	// instead of both firing at full strength at grazing (where F -> 1),
	// which otherwise gains energy. This is the Frostbite-style energy split,
	// not pure Disney 2012 (which is knowingly non-conserving); verified by
	// the furnace test in bsdf_energy_test.odin.
	diffuse := Color{0.0, 0.0, 0.0}
	sheen := Color{0.0, 0.0, 0.0}
	if metallic < 1.0 {
		// Burley diffuse retroreflection, weighted by (1 - F). We deliberately
		// do NOT apply Frostbite's energy_factor renormalization: it conserves
		// energy exactly but darkens a matte white surface to ~0.66
		// reflectance, breaking Lumbre's "albedo = reflectance" contract (and
		// it would darken the Cornell walls once Lambertian folds into
		// Principled in Phase D). The cost is a bounded retroreflective
		// overshoot (<=~1.13) in the narrow grazing lobe — a known Disney-
		// diffuse property, safe under the renderer's firefly clamping. See
		// plans/PRINCIPLED_BSDF.md and the furnace test in bsdf_energy_test.odin.
		fd90 := 0.5 + 2.0 * mat.roughness * cos_d * cos_d
		fl := 1.0 + (fd90 - 1.0) * pow5(1.0 - cos_i)
		fv := 1.0 + (fd90 - 1.0) * pow5(1.0 - cos_o)
		one_minus_F := Color{1.0, 1.0, 1.0} - F
		// (1 - spec_trans): a glass surface has no diffuse lobe.
		diffuse = (1.0 - metallic) * (1.0 - clamp(mat.spec_trans, 0.0, 1.0)) * mat.albedo * INV_PI_F64 * fl * fv * one_minus_F

		// Sheen: additive grazing lobe, no separate sampling lobe.
		if mat.sheen > 0.0 {
			sheen = (1.0 - metallic) * mat.sheen * mat.sheen_tint * pow5(1.0 - cos_d)
		}
	}

	base := diffuse + sheen + specular

	// Clearcoat: thin dielectric coat at F0 = 0.04, its own roughness. The
	// coat reflects `clearcoat * Fcc` of the light, so the layers beneath it
	// are attenuated by (1 - clearcoat*Fcc) rather than the coat lobe simply
	// adding on top (which would gain energy at grazing).
	if mat.clearcoat > 0.0 {
		acc := clearcoat_alpha_sq(mat)
		dcc := ggx_D(cos_h, acc)
		gcc := ggx_G(cos_o, cos_i, acc)
		fcc := schlick_scalar(cos_d, 0.04)
		cc := 0.25 * mat.clearcoat * dcc * fcc * gcc / (4.0 * cos_o * cos_i)
		atten := 1.0 - mat.clearcoat * fcc
		return base * atten + Color{cc, cc, cc}
	}

	return base
}

// Combined solid-angle sampling pdf at `wi`, matching the lobe weights and
// per-lobe pdfs `principled_sample` draws from. `t` is the surface tangent.
principled_pdf :: proc(mat: Material, wo, wi, n, t: Vec3) -> f64 {
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
	cos_d := max(m.dot(wo, wh), 0.0)
	if cos_h <= 0.0 || cos_d <= 0.0 {
		return 0.0
	}

	w_diff, w_spec, w_cc := principled_lobe_weights(mat)
	total := w_diff + w_spec + w_cc
	if total <= 0.0 {
		return 0.0
	}

	tan := surface_tangent(n, t)
	_, _, pdf_s := principled_specular_dg(mat, wo, wi, wh, n, tan, cos_o, cos_i, cos_h)
	pdf_d := cos_i * INV_PI_F64
	pdf_cc := 0.0
	if w_cc > 0.0 {
		acc := clearcoat_alpha_sq(mat)
		pdf_cc = ggx_G1(cos_o, acc) * ggx_D(cos_h, acc) / (4.0 * cos_o)
	}
	return (w_diff * pdf_d + w_spec * pdf_s + w_cc * pdf_cc) / total
}

// Evaluate the BSDF value and the combined sampling PDF for a fixed (wo, wi).
// Kept for the NEE path (returns both in one call).
principled_evaluate :: proc(mat: Material, wo, wi, n, t: Vec3) -> (f: Color, pdf: f64) {
	return principled_eval_f(mat, wo, wi, n, t), principled_pdf(mat, wo, wi, n, t)
}

// PDF-only wrapper, used by NEE to compute MIS weights against the light pdf.
principled_pdf_simple :: proc(mat: Material, wo, wi, n, t: Vec3) -> f64 {
	return principled_pdf(mat, wo, wi, n, t)
}

// Sample an isotropic GGX reflection direction with GGX alpha `alpha`,
// returning the world-space `wi`. Uses the arbitrary make_tangent_vec basis,
// which is fine because the isotropic lobe is rotationally symmetric.
sample_ggx_reflection :: proc(wo, n: Vec3, alpha, cos_o: f64, rng: ^Rng) -> Vec3 {
	u1 := rng_f64(rng)
	u2 := rng_f64(rng)
	tangent := make_tangent_vec(n)
	bitangent := m.cross(n, tangent)
	wo_local := Vec3{m.dot(wo, tangent), m.dot(wo, bitangent), cos_o}
	wh_local := sample_ggx_vndf(wo_local, alpha, u1, u2)
	wh_world := m.normalize(wh_local.x * tangent + wh_local.y * bitangent + wh_local.z * n)
	return reflect_over(-wo, wh_world)
}

// Sample an anisotropic GGX reflection in the surface (t, b, n) frame. `t`
// must be an orthonormalized surface tangent — the anisotropy axis.
sample_ggx_reflection_aniso :: proc(wo, n, t: Vec3, ax, ay, cos_o: f64, rng: ^Rng) -> Vec3 {
	u1 := rng_f64(rng)
	u2 := rng_f64(rng)
	b := m.cross(n, t)
	wo_local := Vec3{m.dot(wo, t), m.dot(wo, b), cos_o}
	wh_local := sample_ggx_vndf_aniso(wo_local, ax, ay, u1, u2)
	wh_world := m.normalize(wh_local.x * t + wh_local.y * b + wh_local.z * n)
	return reflect_over(-wo, wh_world)
}

// Sample a direction and return (f, pdf) for the next ray. Picks a lobe by
// weight, then evaluates the full BSDF and combined pdf at the result.
// `t` is the surface tangent (only used when the material is anisotropic).
principled_sample :: proc(mat: Material, wo, n, t: Vec3, rng: ^Rng) -> (wi: Vec3, f: Color, pdf: f64) {
	cos_o := m.dot(wo, n)
	if cos_o <= 0.0 {
		return Vec3{0.0, 0.0, 1.0}, Color{0.0, 0.0, 0.0}, 0.0
	}

	w_diff, w_spec, w_cc := principled_lobe_weights(mat)
	total := w_diff + w_spec + w_cc
	if total <= 0.0 {
		return Vec3{0.0, 0.0, 1.0}, Color{0.0, 0.0, 0.0}, 0.0
	}

	pick := rng_f64(rng) * total
	if pick < w_diff {
		// Cosine-weighted diffuse hemisphere sample.
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
	} else if pick < w_diff + w_spec {
		if mat.anisotropic <= 0.0 {
			alpha := m.sqrt(max(ggx_alpha_sq(mat.roughness), 0.0))
			wi = sample_ggx_reflection(wo, n, alpha, cos_o, rng)
		} else {
			ax, ay := aniso_alphas(mat)
			wi = sample_ggx_reflection_aniso(wo, n, surface_tangent(n, t), ax, ay, cos_o, rng)
		}
	} else {
		// Clearcoat is always isotropic.
		alpha_cc := m.sqrt(max(clearcoat_alpha_sq(mat), 0.0))
		wi = sample_ggx_reflection(wo, n, alpha_cc, cos_o, rng)
	}

	if m.dot(wi, n) <= 0.0 {
		return Vec3{0.0, 0.0, 1.0}, Color{0.0, 0.0, 0.0}, 0.0
	}
	f = principled_eval_f(mat, wo, wi, n, t)
	pdf = principled_pdf(mat, wo, wi, n, t)
	return wi, f, pdf
}

// ── Specular transmission (glass) — Phase C ─────────────────────────────────
//
// A microfacet dielectric (Walter et al. 2007): sample a GGX microfacet
// normal, compute the dielectric Fresnel there, and stochastically reflect or
// refract off it. With VNDF sampling the D/Fresnel/Jacobian terms cancel and
// the single-sample throughput is `tint * G2/G1` (tint = white for the surface
// reflection, albedo for the tinted transmission). At roughness 0 the
// microfacet normal is the geometric normal and G2/G1 -> 1, so this reduces
// exactly to the perfect glass the legacy Dielectric kind produces (the Phase
// D regression target). Treated as a specular event: no NEE, no GI cache.

// Below this perceptual roughness the GGX lobe is narrower than a pixel for
// the scenes we target.  Continuing to sample a new microfacet normal at each
// interface of a thick or concave solid then turns what should look like clear
// glass into accumulated stochastic blur.  Use the exact dielectric event in
// that range instead.  This is also a much lower-variance representation of
// the limiting GGX distribution.
GLASS_DELTA_ROUGHNESS :: 0.02

// Sample an (isotropic) GGX microfacet normal about `n`, returned in world
// space. Requires cos_o = dot(wo, n) > 0.
sample_ggx_microfacet :: proc(wo, n: Vec3, alpha, cos_o: f64, rng: ^Rng) -> Vec3 {
	u1 := rng_f64(rng)
	u2 := rng_f64(rng)
	tangent := make_tangent_vec(n)
	bitangent := m.cross(n, tangent)
	wo_local := Vec3{m.dot(wo, tangent), m.dot(wo, bitangent), cos_o}
	wh_local := sample_ggx_vndf(wo_local, alpha, u1, u2)
	return m.normalize(wh_local.x * tangent + wh_local.y * bitangent + wh_local.z * n)
}

// Samples the glass lobe. `n` is the shading normal oriented toward `wo`
// (dot(wo,n) > 0), `front_face` selects the eta ratio. Returns the outgoing
// direction (which is below the surface for a transmission event), the path
// throughput multiplier, and ok=false on a degenerate/masked sample.
principled_sample_glass :: proc(
	mat: Material, wo, n: Vec3, front_face: bool, rng: ^Rng,
) -> (wi: Vec3, throughput: Color, ok: bool) {
	cos_o := m.dot(wo, n)
	if cos_o <= 0.0 {
		return {}, {}, false
	}
	eta := 1.0 / mat.ir if front_face else mat.ir
	alpha_sq := ggx_alpha_sq(max(mat.roughness, 0.001))
	alpha := m.sqrt(alpha_sq)

	// A near-delta lobe must not acquire a different random normal at every
	// total-internal-reflection bounce.  In particular, the shader ball's
	// authored 0.01625 roughness falls in this range.
	wh := n if mat.roughness <= GLASS_DELTA_ROUGHNESS else sample_ggx_microfacet(wo, n, alpha, cos_o, rng)
	cos_ow := m.dot(wo, wh)
	if cos_ow <= 0.0 {
		return {}, {}, false
	}

	incident := -wo
	sin_ow := m.sqrt(max(1.0 - cos_ow * cos_ow, 0.0))
	cannot_refract := eta * sin_ow > 1.0
	tint: Color
	if cannot_refract || reflectance(cos_ow, eta) > rng_f64(rng) {
		wi = reflect(incident, wh)
		tint = Color{1.0, 1.0, 1.0} // dielectric surface reflection is untinted
		// A microfacet reflection can go below the geometric surface (masked).
		if m.dot(wi, n) <= 0.0 {
			return {}, {}, false
		}
	} else {
		wi = refract(incident, wh, eta)
		tint = mat.albedo // transmission carries the base-color tint
		if m.dot(wi, n) >= 0.0 {
			// Refraction that failed to cross the surface — treat as masked.
			return {}, {}, false
		}
	}

	// VNDF microfacet weight: tint * G2/G1, with G on absolute cosines so it is
	// valid for the below-surface transmission direction too.
	cos_i_abs := m.abs(m.dot(wi, n))
	g2 := ggx_G(cos_o, cos_i_abs, alpha_sq)
	g1 := ggx_G1(cos_o, alpha_sq)
	gratio := g2 / g1 if g1 > 0.0 else 0.0
	throughput = tint * gratio
	return wi, throughput, true
}

// ── Material consolidation (Phase D) ────────────────────────────────────────
//
// Converts the legacy RTIOW material kinds to Principled parameters so the
// renderer standardizes on one BSDF. Lambertian maps to a diffuse-only
// Principled (specular 0), which the GPU kernel still routes to the
// GI-cache/photon fast path via `material_is_lambertian_like` — so a converted
// Lambertian shades byte-identically to the old Lambertian kind. Metal becomes
// a metallic GGX surface (fuzz -> roughness); Dielectric becomes a smooth
// glass (spec_trans 1), which the Phase C transmission lobe renders. Emissive
// and already-Principled materials are unchanged. See plans/PRINCIPLED_BSDF.md.
normalize_material :: proc(mat: Material) -> Material {
	out := mat
	switch mat.kind {
	case .Lambertian:
		out.kind = .Principled
		out.roughness = 1.0
		out.metallic = 0.0
		out.specular = 0.0 // no specular lobe -> stays "Lambertian-like"
		out.spec_trans = 0.0
		out.clearcoat = 0.0
		out.sheen = 0.0
		out.anisotropic = 0.0
	case .Metal:
		out.kind = .Principled
		out.metallic = 1.0
		out.roughness = clamp(mat.fuzz, 0.0, 1.0) // fuzz blur ~ GGX roughness
		out.specular = 0.5
		out.spec_trans = 0.0
	case .Dielectric:
		out.kind = .Principled
		out.spec_trans = 1.0
		out.roughness = 0.0
		out.metallic = 0.0
		out.specular = 0.5
		// A dielectric with no authored tint transmits clear.
		if mat.albedo == (Color{0, 0, 0}) {
			out.albedo = Color{1, 1, 1}
		}
	case .Principled, .Emissive:
	// unchanged
	}
	return out
}

// True when a Principled material is a pure diffuse surface — the case the GI
// cache and photon map were tuned for. Matches exactly what `normalize_material`
// produces for a Lambertian, and deliberately excludes glTF/USD Principled
// materials (which default specular to 0.5), so their look is unchanged.
material_is_lambertian_like :: proc(mat: Material) -> bool {
	return(
		mat.kind == .Principled &&
		mat.metallic < 0.01 &&
		mat.spec_trans < 0.01 &&
		mat.specular < 0.01 &&
		mat.clearcoat < 0.01 &&
		mat.sheen < 0.01 &&
		mat.roughness > 0.99 \
	)
}
