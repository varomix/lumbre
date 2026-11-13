package lumbre_ltc

// Fits the table: for each roughness and view angle, the transform M whose
// cosine best matches the BRDF lobe. After the paper's fitLTC.cpp: the lobe
// is measured for its magnitude, Fresnel part and average direction; the
// transform is built in a frame aligned with that average direction, with
// three free parameters fitted by Nelder-Mead against an L3 error estimated
// by sampling both distributions. Each cell starts from its neighbour's
// solution, walking from rough to smooth and from normal to grazing, because
// the error surface for a sharp lobe is too narrow to find from scratch.

import "core:math"
import "core:math/linalg"

// Samples per axis for the lobe's moments and for the fit's error. Stratified
// rather than random, so the fit is deterministic.
MOMENT_SAMPLES :: 64
ERROR_SAMPLES :: 32

// The lobe's integrals for one view direction.
@(private = "file")
Moments :: struct {
	norm, fresnel: f32,
	average:       Vec3, // unit, in the xz plane
}

@(private = "file")
moments :: proc(v: Vec3, roughness: f32) -> Moments {
	out: Moments
	n := MOMENT_SAMPLES
	for j in 0 ..< n {
		for i in 0 ..< n {
			u1 := (f32(i) + 0.5) / f32(n)
			u2 := (f32(j) + 0.5) / f32(n)
			l := sample_brdf(v, roughness, u1, u2)
			value, pdf := brdf_eval(v, l, roughness)
			if pdf <= 0 { continue }
			w := value / pdf
			h := linalg.normalize(v + l)
			out.norm += w
			out.fresnel += w * math.pow(1 - max(linalg.dot(v, h), 0), 5)
			out.average += w * l
		}
	}
	count := f32(n * n)
	out.norm /= count
	out.fresnel /= count
	out.average.y = 0
	out.average = linalg.normalize(out.average)
	return out
}

// The transform for parameters p = (a, c, b) in the frame whose z axis is the
// lobe's average direction: M = frame * | a 0 b ; 0 c 0 ; 0 0 1 |.
@(private = "file")
transform :: proc(p: [3]f32, average: Vec3) -> Mat3 {
	x := Vec3{average.z, 0, -average.x}
	y := Vec3{0, 1, 0}
	z := average
	frame := Mat3{
		x.x, y.x, z.x,
		x.y, y.y, z.y,
		x.z, y.z, z.z,
	}
	scale := Mat3{
		p[0], 0,    p[2],
		0,    p[1], 0,
		0,    0,    1,
	}
	return frame * scale
}

// L3 difference between the lobe and the transformed cosine, by multiple
// importance sampling of both.
@(private = "file")
fit_error :: proc(t: Ltc, v: Vec3, roughness: f32) -> f32 {
	err: f64 = 0
	n := ERROR_SAMPLES
	for j in 0 ..< n {
		for i in 0 ..< n {
			u1 := (f32(i) + 0.5) / f32(n)
			u2 := (f32(j) + 0.5) / f32(n)
			for from_ltc in ([2]bool{true, false}) {
				l := from_ltc ? ltc_sample(t, u1, u2) : sample_brdf(v, roughness, u1, u2)
				brdf, pdf_brdf := brdf_eval(v, l, roughness)
				density := ltc_density(t, l)
				ltc_value := t.magnitude * density
				d := f64(abs(brdf - ltc_value))
				denom := f64(pdf_brdf) + f64(density)
				if denom > 0 {
					err += d * d * d / denom
				}
			}
		}
	}
	return f32(err / f64(n * n))
}

// Nelder-Mead over three parameters, minimizing `f`.
@(private = "file")
nelder_mead :: proc(start: [3]f32, step: f32, iterations: int, data: rawptr, f: proc(p: [3]f32, data: rawptr) -> f32) -> [3]f32 {
	simplex: [4][3]f32
	values: [4]f32
	simplex[0] = start
	for k in 0 ..< 3 {
		simplex[k + 1] = start
		simplex[k + 1][k] += step
	}
	for k in 0 ..< 4 { values[k] = f(simplex[k], data) }

	for _ in 0 ..< iterations {
		// Order best to worst.
		for a in 0 ..< 4 {
			for b in a + 1 ..< 4 {
				if values[b] < values[a] {
					values[a], values[b] = values[b], values[a]
					simplex[a], simplex[b] = simplex[b], simplex[a]
				}
			}
		}
		if abs(values[3] - values[0]) <= 1e-7 * max(abs(values[0]), 1e-12) {
			break
		}
		centroid := (simplex[0] + simplex[1] + simplex[2]) / 3
		reflected := centroid + (centroid - simplex[3])
		fr := f(reflected, data)
		if fr < values[0] {
			expanded := centroid + 2 * (centroid - simplex[3])
			fe := f(expanded, data)
			if fe < fr {
				simplex[3], values[3] = expanded, fe
			} else {
				simplex[3], values[3] = reflected, fr
			}
			continue
		}
		if fr < values[2] {
			simplex[3], values[3] = reflected, fr
			continue
		}
		contracted := centroid + 0.5 * (simplex[3] - centroid)
		fc := f(contracted, data)
		if fc < values[3] {
			simplex[3], values[3] = contracted, fc
			continue
		}
		for k in 1 ..< 4 {
			simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
			values[k] = f(simplex[k], data)
		}
	}
	best := 0
	for k in 1 ..< 4 { if values[k] < values[best] { best = k } }
	return simplex[best]
}

@(private = "file")
Fit_Data :: struct {
	v:         Vec3,
	roughness: f32,
	m:         Moments,
}

@(private = "file")
fit_objective :: proc(p: [3]f32, data: rawptr) -> f32 {
	d := (^Fit_Data)(data)
	// Degenerate or mirrored scales are not transforms of a cosine lobe.
	if p[0] <= 1e-5 || p[1] <= 1e-5 {
		return 1e30
	}
	m := transform(p, d.m.average)
	if abs(linalg.determinant(m)) < 1e-9 {
		return 1e30
	}
	return fit_error(ltc_make(m, d.m.norm), d.v, d.roughness)
}

// Fits one cell, starting from `guess`. Returns the parameters and the entry.
fit_cell :: proc(x, y: int, guess: [3]f32) -> (params: [3]f32, entry: Entry) {
	roughness := cell_roughness(x)
	cos_t := cell_cos_theta(y)
	v := Vec3{math.sqrt(1 - cos_t * cos_t), 0, cos_t}
	data := Fit_Data{v = v, roughness = roughness, m = moments(v, roughness)}

	params = nelder_mead(guess, 0.05, 200, &data, fit_objective)
	m := transform(params, data.m.average)
	inv := linalg.inverse(m)
	// Scaled so the middle entry is 1; see Entry.
	k := 1 / inv[1, 1]
	entry = Entry {
		inv     = {inv[0, 0] * k, inv[0, 2] * k, inv[2, 0] * k, inv[2, 2] * k},
		norm    = data.m.norm,
		fresnel = data.m.fresnel,
	}
	return
}

// The `sphere` entry for cell (x, y): a spherical cap whose unclipped
// form factor is `length` = y / (SIZE - 1) -- so its half-angle s has
// sin^2 s = length -- with its axis at elevation z = 2 x / (SIZE - 1) - 1.
// Returns the clamped cosine's integral over the cap, clipped at the horizon,
// divided by `length`. Midpoint quadrature over the cap, so deterministic.
sphere_cell :: proc(x, y: int) -> f32 {
	z := f32(x) / (SIZE - 1) * 2 - 1
	length := f32(y) / (SIZE - 1)
	if length <= 0 {
		return max(z, 0)
	}
	cos_s := math.sqrt(max(1 - length, 0))
	axis := Vec3{math.sqrt(max(1 - z * z, 0)), 0, z}
	t := Vec3{z, 0, -axis.x}
	b := Vec3{0, 1, 0}
	// Uniform in solid angle over the cap: cos(theta) uniform in [cos_s, 1].
	n := 256
	sum: f64 = 0
	for j in 0 ..< n {
		ct := cos_s + (1 - cos_s) * (f32(j) + 0.5) / f32(n)
		st := math.sqrt(max(1 - ct * ct, 0))
		for i in 0 ..< n {
			phi := 2 * math.PI * (f32(i) + 0.5) / f32(n)
			w := axis * ct + (t * math.cos(phi) + b * math.sin(phi)) * st
			sum += f64(max(w.z, 0))
		}
	}
	solid_angle := 2 * math.PI * (1 - cos_s)
	integral := f32(sum / f64(n * n)) * solid_angle / math.PI
	return integral / length
}

// Fits the whole table, row by row from the roughest lobe at normal incidence.
// `progress` is called once per roughness column, if given.
fit_table :: proc(table: ^Table, progress: proc(done, total: int) = nil) {
	previous_column_first: [3]f32 = {1, 1, 0}
	for xi := SIZE - 1; xi >= 0; xi -= 1 {
		guess := previous_column_first
		for y in 0 ..< SIZE {
			params, entry := fit_cell(xi, y, guess)
			table[y * SIZE + xi] = entry
			guess = params
			if y == 0 {
				previous_column_first = params
			}
		}
		if progress != nil {
			progress(SIZE - xi, SIZE)
		}
	}
}
