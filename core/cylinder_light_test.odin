package lumbre_core

import "core:math"
import m "core:math/linalg/glsl"
import "core:testing"

// Irradiance at `p` on a surface facing `n`, estimated with the cylinder
// light's own sampler, as next-event estimation uses it. Unoccluded.
@(private = "file")
estimate_irradiance :: proc(light: Light, p: Point3, n: Vec3, samples: int) -> f64 {
	rng := Rng{state = 0x9e3779b97f4a7c15}
	sum := 0.0
	for _ in 0 ..< samples {
		s := sample_cylinder_light(light, p, &rng)
		if s.pdf <= 0 { continue }
		sum += light.intensity.x * max(m.dot(n, s.direction), 0) / s.pdf
	}
	return sum / f64(samples)
}

// A thin tube of radiance L seen side-on: each slice dx of it shows the point a
// strip 2r wide, so E = L 2r integral of cos_surface * sin(view to axis) / rho^2,
// the closed form for the half of the tube that faces the point.
@(private = "file")
thin_tube_irradiance :: proc(L, r, h, x0, x1: f64) -> f64 {
	n := 100000
	sum := 0.0
	for i in 0 ..< n {
		x := x0 + (f64(i) + 0.5) * (x1 - x0) / f64(n)
		rho2 := x * x + h * h
		sum += h * h / (rho2 * rho2)
	}
	return L * 2 * r * sum * (x1 - x0) / f64(n)
}

// Only the facing half of a cylinder lights a point: its far side is hidden
// behind the near one. Counting the whole surface by |cos| made cylinders twice
// as bright as this.
@(test)
test_cylinder_light_lights_from_its_facing_half :: proc(t: ^testing.T) {
	// Along x from -1.1 to 1.1, 0.5 above a floor point under its middle.
	light := make_cylinder_light({-1.1, 0.5, 0}, {1, 0, 0}, 0.02, 2.2, {4, 4, 4})
	got := estimate_irradiance(light, {0, 0, 0}, {0, 1, 0}, 400000)
	want := thin_tube_irradiance(4, 0.02, 0.5, -1.1, 1.1)
	testing.expectf(t, abs(got - want) < want * 0.01, "irradiance %v, want %v", got, want)

	// A fat tube up close: the facing half's grazing edges face away from the
	// point and must count for nothing, not for something.
	fat := make_cylinder_light({-1, 0.6, 0}, {1, 0, 0}, 0.4, 2, {1, 1, 1})
	close := estimate_irradiance(fat, {0, 0, 0}, {0, 1, 0}, 400000)
	// Irradiance from a long cylinder (Lambertian emitter) on a parallel strip
	// directly below, by brute force over the visible surface: integrate the
	// form factor kernel over every surface patch that faces the point.
	ref := 0.0
	{
		nphi, nx := 720, 400
		for j in 0 ..< nx {
			x := -1 + (f64(j) + 0.5) * 2 / f64(nx)
			for i in 0 ..< nphi {
				phi := (f64(i) + 0.5) * 2 * math.PI / f64(nphi)
				radial := Vec3{0, math.cos(phi), math.sin(phi)}
				q := Vec3{x, 0.6, 0} + radial * 0.4
				d := q
				dist := m.length(d)
				dir := d / dist
				cos_l := m.dot(radial, -dir)
				cos_s := dir.y
				if cos_l <= 0 || cos_s <= 0 { continue }
				da := (2 * math.PI * 0.4 / f64(nphi)) * (2 / f64(nx))
				ref += cos_l * cos_s / (dist * dist) * da
			}
		}
	}
	testing.expectf(t, abs(close - ref) < ref * 0.01, "close irradiance %v, want %v", close, ref)

	// Inside the tube the surface faces away everywhere: it emits outward.
	testing.expect_value(t, sample_cylinder_light(fat, {0, 0.6, 0}, &Rng{state = 1}).pdf, 0)
}

// A USD CylinderLight's transform places its centre; Lumbre's light is placed
// by the base of its axis, so the importer must step back half the length.
@(test)
test_usd_cylinder_light_is_centred :: proc(t: ^testing.T) {
	world := m.mat4Translate({1, 2, 3})
	light, ok := usd_make_light_from_params(Usd_Light_Params {
		world = world, kind = .Cylinder, intensity = 1, color = {1, 1, 1}, radius = 0.1, length = 4,
	})
	testing.expect(t, ok)
	centre := light.position + light.direction * (light.height * 0.5)
	testing.expectf(t, m.length(centre - Vec3{1, 2, 3}) < 1e-9, "centre %v, want (1, 2, 3)", centre)
	testing.expect_value(t, light.height, 4)
}
