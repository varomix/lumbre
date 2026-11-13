package lumbre_ltc

import "core:math"
import "core:math/linalg"
import "core:testing"

TABLE_BYTES :: #load(TABLE_FILE)

@(private = "file")
load_table :: proc() -> ^Table {
	assert(len(TABLE_BYTES) == size_of(Table))
	return (^Table)(raw_data(TABLE_BYTES))
}

// The table's nearest cell for a roughness and view cosine.
@(private = "file")
lookup :: proc(t: ^Table, roughness, cos_v: f32) -> Entry {
	x := int(math.round(clamp(roughness, 0, 1) * (SIZE - 1)))
	y := int(math.round(math.sqrt(clamp(1 - cos_v, 0, 1)) * (SIZE - 1)))
	return t[y * SIZE + x]
}

// A rectangle as seen from the origin: centre, and half its two edges.
@(private = "file")
Quad :: struct { centre, hu, hv: Vec3 }

@(private = "file")
corners :: proc(q: Quad) -> [4]Vec3 {
	return {q.centre - q.hu - q.hv, q.centre + q.hu - q.hv, q.centre + q.hu + q.hv, q.centre - q.hu + q.hv}
}

// Whether a ray from the origin along `d` hits the quad.
@(private = "file")
hits :: proc(q: Quad, d: Vec3) -> bool {
	n := linalg.cross(q.hu, q.hv)
	denom := linalg.dot(n, d)
	if abs(denom) < 1e-9 { return false }
	t := linalg.dot(n, q.centre) / denom
	if t <= 0 { return false }
	rel := d * t - q.centre
	a := linalg.dot(rel, q.hu) / linalg.dot(q.hu, q.hu)
	b := linalg.dot(rel, q.hv) / linalg.dot(q.hv, q.hv)
	return abs(a) <= 1 && abs(b) <= 1
}

// The reference: the BRDF times cosine integrated over the quad, by importance
// sampling the BRDF itself.
@(private = "file")
reference :: proc(v: Vec3, roughness: f32, q: Quad) -> f32 {
	n := 512
	sum: f64 = 0
	for j in 0 ..< n {
		for i in 0 ..< n {
			l := sample_brdf(v, roughness, (f32(i) + 0.5) / f32(n), (f32(j) + 0.5) / f32(n))
			if l.z <= 0 || !hits(q, l) { continue }
			value, pdf := brdf_eval(v, l, roughness)
			if pdf > 0 { sum += f64(value / pdf) }
		}
	}
	return f32(sum / f64(n * n))
}

@(private = "file")
estimate :: proc(table: ^Table, e: Entry, q: Quad) -> f32 {
	inv := entry_inverse(e)
	c := corners(q)
	transformed: [4]Vec3
	for p, k in c { transformed[k] = inv * p }
	return e.norm * polygon_form_factor(table, transformed[:])
}

// Every cell is a usable transform with a plausible magnitude.
@(test)
test_table_is_sane :: proc(t: ^testing.T) {
	table := load_table()
	for e, i in table {
		finite := true
		for x in ([]f32{e.inv.x, e.inv.y, e.inv.z, e.inv.w, e.norm, e.fresnel}) {
			finite &&= math.classify(x) == .Normal || x == 0
		}
		testing.expectf(t, finite, "cell %d not finite: %v", i, e)
		det := e.inv.x * e.inv.w - e.inv.y * e.inv.z
		testing.expectf(t, det > 0, "cell %d is not invertible: %v", i, e)
		testing.expectf(t, e.norm > 0 && e.norm <= 1.001 && e.fresnel >= 0 && e.fresnel <= e.norm, "cell %d magnitude %v", i, e)
	}
}

// The transformed cosine reproduces the BRDF's integral over lights of
// several sizes and places, across roughness and view angle.
@(test)
test_ltc_matches_the_brdf_over_quads :: proc(t: ^testing.T) {
	table := load_table()
	quads := [?]Quad {
		{{0, 0, 2}, {1, 0, 0}, {0, 1, 0}},       // large, overhead
		{{1.5, 0, 1}, {0.3, 0, 0}, {0, 0.6, 0}}, // small, toward the reflection
		{{-1, 0.5, 1}, {0, 0.5, 0}, {0, 0, 0.5}}, // to the side, crossing the horizon
	}
	worst: f32
	for roughness in ([]f32{0.1, 0.25, 0.5, 0.8}) {
		for cos_v in ([]f32{1, 0.8, 0.5, 0.2}) {
			v := Vec3{math.sqrt(1 - cos_v * cos_v), 0, cos_v}
			e := lookup(table, roughness, cos_v)
			for q in quads {
				ref := reference(v, roughness, q)
				est := estimate(table, e, q)
				// Relative to the lobe's whole magnitude: a light catching a
				// sliver of the lobe has a tiny integral, and a relative error
				// on that alone measures nothing anyone sees.
				err := abs(est - ref) / e.norm
				worst = max(worst, err)
			}
		}
	}
	// Measured at 0.050 (smooth, grazing), and mostly under 0.01 -- in line
	// with the paper's own fits.
	testing.expectf(t, worst < 0.07, "worst error %.3f of the lobe", worst)
}

// Diffuse is the identity transform: a quad straight overhead gives the
// textbook form factor of a rectangle.
@(test)
test_polygon_integral_is_the_form_factor :: proc(t: ^testing.T) {
	// A square of half-size a at height h, centred over the point.
	a: f32 = 1
	h: f32 = 1.5
	q := Quad{{0, 0, h}, {a, 0, 0}, {0, a, 0}}
	c := corners(q)
	table := load_table()
	got := polygon_integral_exact(c[:])
	approx := polygon_form_factor(table, c[:])
	x := a / h
	y := a / h
	want := (2 / math.PI) * (x / math.sqrt(1 + x * x) * math.atan(y / math.sqrt(1 + x * x)) +
		y / math.sqrt(1 + y * y) * math.atan(x / math.sqrt(1 + y * y)))
	testing.expectf(t, abs(got - want) < 1e-3, "form factor %v, want %v", got, want)
	// Wholly above the horizon the approximation has nothing to approximate.
	testing.expectf(t, abs(approx - want) < 2e-3, "approximate form factor %v, want %v", approx, want)
	// Seen from behind, the same.
	back := [4]Vec3{c[3], c[2], c[1], c[0]}
	testing.expectf(t, abs(polygon_integral_exact(back[:]) - want) < 1e-3, "reversed winding %v", polygon_integral_exact(back[:]))
	testing.expectf(t, abs(polygon_form_factor(table, back[:]) - want) < 2e-3, "reversed winding, approximate %v", polygon_form_factor(table, back[:]))
	// Entirely below the horizon: nothing.
	below := [4]Vec3{{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}}
	testing.expect_value(t, polygon_integral_exact(below[:]), 0)
	testing.expect(t, polygon_form_factor(table, below[:]) < 1e-3)
}

// Where a light crosses the horizon, the spherical-cap correction stands in
// for clipping. How far off it is, over quads of several sizes swept from
// overhead to below the horizon, measured against exact clipping, in units
// of the whole hemisphere's integral (1).
@(test)
test_horizon_approximation_error :: proc(t: ^testing.T) {
	table := load_table()
	worst: f32
	for size in ([]f32{0.2, 0.6, 1.5}) {
		for k in 0 ..= 40 {
			// Tilt the quad's centre from straight up (0) to below the horizon.
			a := f32(k) / 40 * 1.9
			centre := Vec3{math.sin(a), 0, math.cos(a)} * 2
			hu := Vec3{0, size, 0}
			hv := linalg.normalize(linalg.cross(centre, hu)) * size
			q := Quad{centre, hu, hv}
			c := corners(q)
			exact := polygon_integral_exact(c[:])
			approx := polygon_form_factor(table, c[:])
			worst = max(worst, abs(exact - approx))
		}
	}
	// Measured at 0.0018.
	testing.expectf(t, worst < 0.005, "worst horizon error %.4f", worst)
}
