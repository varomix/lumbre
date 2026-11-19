package lumbre_core

import m "core:math"
import la "core:math/linalg"
import "core:testing"

// `sample_ggx_vndf` must draw half vectors from the distribution of visible
// normals, G1(wo) max(0, wo.h) D(h) / cos_o, which is the pdf the path tracer
// divides by. Compares the mean elevation of the reflected direction and the
// mean G1 along it -- the latter is the specular albedo estimate itself --
// between a stratified sweep of the sampler and a quadrature of that pdf.
// The sampler's basis was built around the wrong axis and its far half-disc
// scaled by sqrt(1 - s1^2) instead of v.z: at alpha 0.25 and 85 degrees the
// albedo came out 0.59 against 0.86, so grazing glossy reflections were 30%
// dark and skewed.
@(test)
test_vndf_sampler_matches_its_pdf :: proc(t: ^testing.T) {
	for alpha in ([]f64{0.05, 0.25, 0.6}) {
		for deg in ([]f64{30, 60, 75, 85}) {
			th := m.to_radians(deg)
			wo := Vec3{m.sin(th) * m.cos(0.7), m.sin(th) * m.sin(0.7), m.cos(th)}
			a2 := alpha * alpha

			N :: 256
			sampled_z, sampled_g: f64
			for j in 0 ..< N {
				for i in 0 ..< N {
					h := sample_ggx_vndf(wo, alpha, (f64(i) + 0.5) / N, (f64(j) + 0.5) / N)
					l := 2 * la.dot(wo, h) * h - wo
					if l.z <= 0 { continue }
					sampled_z += l.z / (N * N)
					sampled_g += ggx_G1(l.z, a2) / (N * N)
				}
			}

			T :: 1024
			P :: 128
			exact_z, exact_g: f64
			for j in 0 ..< T {
				th_h := (f64(j) + 0.5) / T * m.PI / 2
				for i in 0 ..< P {
					ph := (f64(i) + 0.5) / P * 2 * m.PI
					h := Vec3{m.sin(th_h) * m.cos(ph), m.sin(th_h) * m.sin(ph), m.cos(th_h)}
					wo_h := la.dot(wo, h)
					if wo_h <= 0 { continue }
					l := 2 * wo_h * h - wo
					if l.z <= 0 { continue }
					w := ggx_G1(wo.z, a2) * wo_h * ggx_D(h.z, a2) / wo.z *
						m.sin(th_h) * (m.PI / 2 / T) * (2 * m.PI / P)
					exact_z += w * l.z
					exact_g += w * ggx_G1(l.z, a2)
				}
			}
			testing.expectf(t, abs(sampled_z - exact_z) < 0.005 && abs(sampled_g - exact_g) < 0.005,
				"alpha %.2f at %.0f degrees: sampled (%.4f, %.4f), pdf says (%.4f, %.4f)",
				alpha, deg, sampled_z, sampled_g, exact_z, exact_g)
		}
	}
}
