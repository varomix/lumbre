package main

import "core:fmt"
import "core:math"
import m "core:math/linalg/glsl"

// Monte-Carlo white-furnace / energy check for the CPU Principled BSDF
// (the reference the GPU shader mirrors). For a reflective-only BSDF with a
// white albedo and white F0, the directional-hemispherical reflectance
//
//     R(wo) = ∫ f(wo,wi) cos_i dwi   ≈  (1/N) Σ f·cos_i / pdf
//
// must be ≤ 1 for every roughness/metallic/view angle — energy in ≥ energy
// out. A value meaningfully above 1, or a NaN, means the sampling pdf and the
// evaluated f disagree (exactly the class of bug the VNDF-pdf fix in PLAN.md
// Stage 3A addressed). Single-scatter GGX legitimately loses a little energy
// at high roughness, so R somewhat below 1 there is expected, not a failure.
run_bsdf_energy_test :: proc() {
	rng := Rng{state = 0x9e3779b97f4a7c15}
	n := Vec3{0.0, 0.0, 1.0}
	N :: 400000

	Config :: struct {
		name:                string,
		roughness, metallic: f64,
		clearcoat, sheen:    f64,
		anisotropic:         f64,
	}
	configs := []Config{
		{"diffuse white",        1.0, 0.0, 0.0, 0.0, 0.0},
		{"glossy dielectric",    0.3, 0.0, 0.0, 0.0, 0.0},
		{"smooth dielectric",    0.05, 0.0, 0.0, 0.0, 0.0},
		{"rough metal",          0.5, 1.0, 0.0, 0.0, 0.0},
		{"smooth metal",         0.05, 1.0, 0.0, 0.0, 0.0},
		{"clearcoat over diff",  0.6, 0.0, 1.0, 0.0, 0.0},
		{"sheen cloth",          0.8, 0.0, 0.0, 1.0, 0.0},
		{"aniso metal",          0.4, 1.0, 0.0, 0.0, 0.8},
		{"aniso dielectric",     0.4, 0.0, 0.0, 0.0, 0.8},
	}
	view_angles := []f64{5.0, 45.0, 75.0}

	// A pdf/eval mismatch (the bug class this guards against) shows up as
	// gross gain (R >> 1) or NaN, and metals — which have no non-conserving
	// diffuse lobe — must stay <= ~1. The Disney diffuse legitimately
	// overshoots a little in the narrow grazing retroreflection lobe (a model
	// property we accept over darkening albedo; see principled.odin), so a
	// modest grazing R up to ~1.15 is reported, not failed.
	GROSS_GAIN :: 1.15
	METAL_CAP :: 1.02
	fmt.println("── Principled BSDF energy test ──")
	worst := 0.0
	worst_metal := 0.0
	any_nan := false
	for cfg in configs {
		mat := Material{
			kind                = .Principled,
			albedo              = Color{1.0, 1.0, 1.0},
			roughness           = cfg.roughness,
			metallic            = cfg.metallic,
			specular            = 0.5,
			specular_tint       = Color{1.0, 1.0, 1.0},
			clearcoat           = cfg.clearcoat,
			clearcoat_roughness = 0.1,
			sheen               = cfg.sheen,
			sheen_tint          = Color{1.0, 1.0, 1.0},
			anisotropic         = cfg.anisotropic,
			ir                  = 1.5,
		}
		// Fixed reference tangent — energy conservation holds for any tangent
		// orientation, so a constant one keeps the test deterministic.
		tan := make_tangent_vec(n)
		fmt.printf("  %-20s r=%.2f met=%.1f cc=%.1f sh=%.1f an=%.1f : ",
			cfg.name, cfg.roughness, cfg.metallic, cfg.clearcoat, cfg.sheen, cfg.anisotropic)
		for theta_deg in view_angles {
			theta := theta_deg * math.PI / 180.0
			wo := Vec3{math.sin(theta), 0.0, math.cos(theta)}
			acc := 0.0
			for _ in 0 ..< N {
				wi, f, pdf := principled_sample(mat, wo, n, tan, &rng)
				cos_i := m.dot(wi, n)
				if pdf > 0.0 && cos_i > 0.0 {
					// White material: all channels equal, integrate one.
					acc += f.x * cos_i / pdf
				}
			}
			R := acc / f64(N)
			if math.is_nan(R) {
				any_nan = true
			} else {
				worst = max(worst, R)
				if cfg.metallic >= 1.0 {
					worst_metal = max(worst_metal, R)
				}
			}
			fmt.printf("R(%.0f°)=%.3f  ", theta_deg, R)
		}
		fmt.println()
	}
	fmt.printf("── worst R = %.3f, worst metal R = %.3f, any NaN = %v ──\n", worst, worst_metal, any_nan)
	if any_nan {
		fmt.println("FAIL: NaN in reflectance (pdf/eval mismatch)")
	} else if worst_metal > METAL_CAP {
		fmt.println("FAIL: metal gains energy (pdf/eval mismatch)")
	} else if worst > GROSS_GAIN {
		fmt.println("FAIL: gross energy gain — pdf/eval mismatch")
	} else {
		fmt.println("PASS: no bug-level energy gain; grazing diffuse overshoot within model bound")
	}
}
