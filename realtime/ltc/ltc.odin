package lumbre_ltc

// Linearly transformed cosines (Heitz, Dupuy, Hill and Neubelt, 2016) for
// shading polygonal lights in the rasterizer.
//
// The specular lobe at a given roughness and view angle is approximated by a
// clamped cosine pushed through a 3x3 matrix M. Integrating that over a
// polygon has a closed form: transform the polygon's corners by M^-1 and sum
// one term per edge. So an area light's specular costs a table lookup and a
// handful of edge terms per pixel, instead of the sampling the path tracer
// does.
//
// The table is fitted here, to Lumbre's own specular term -- the GGX with
// separable Smith G and roughness used directly as alpha that
// shaders/shading.slang's eval_brdf and the path tracer share -- rather than
// taken from the paper's published fit, which assumes a different G and
// roughness mapping. `tools/ltc_fit` runs the fit and writes TABLE_FILE; the
// renderer loads that file at compile time.
//
// No GPU or SDL here, so the fit and the tests run anywhere.

import "core:math"
import "core:math/linalg"

SIZE :: 64

// Roughness below this is a mirror the fit cannot represent; the shader
// clamps to it, and so does the table's first row.
MIN_ROUGHNESS :: 0.03

// Per table cell: the inverse transform's four free entries, and the lobe's
// magnitude and Fresnel-weighted magnitude. See `Entry`.
TABLE_FILE :: "ltc_table.bin"

Vec3 :: [3]f32
Mat3 :: matrix[3, 3]f32

// One fitted cell, as the shader reads it from two RGBA textures.
//
// `inv` holds M^-1 scaled so its middle entry is 1, which is all the shader
// needs: scaling the transform does not change the normalized directions the
// edge integral works with. The other entries are zero by symmetry -- the
// view direction lies in the xz plane of the frame the lobe is fitted in.
//   M^-1 = | inv.x  0  inv.y |
//          |   0    1    0   |
//          | inv.z  0  inv.w |
//
// `norm` is the specular directional albedo with F = 1, and `fresnel` the
// part of it weighted by (1 - V.H)^5, so with Schlick's Fresnel the lobe's
// total is F0 * (norm - fresnel) + fresnel.
//
// `sphere` is a separate table sharing the texture: indexed not by roughness
// and view but by a polygon's vector form factor -- its elevation z on the
// x axis, its length on the y axis -- it holds how much of a spherical cap
// with that form factor survives clipping at the horizon, as a fraction of
// the unclipped length. See `polygon_form_factor`.
Entry :: struct {
	inv:     [4]f32,
	norm:    f32,
	fresnel: f32,
	sphere:  f32,
	_pad:    f32,
}

Table :: [SIZE * SIZE]Entry

// The table's axes: x is roughness, y is sqrt(1 - cos theta_v). The square
// root spends more rows near grazing, where the lobe changes fastest.
cell_roughness :: proc(x: int) -> f32 {
	return max(f32(x) / (SIZE - 1), MIN_ROUGHNESS)
}

cell_cos_theta :: proc(y: int) -> f32 {
	s := f32(y) / (SIZE - 1)
	// Not quite 90 degrees: at exactly grazing the view direction lies in the
	// surface and there is no lobe to fit.
	return max(1 - s * s, 0.0005)
}

// ── Lumbre's specular BRDF, mirroring eval_brdf in shading.slang ────────────
//
// Tangent frame, normal +z. Returns f * cos(theta_l) with F = 1, and the pdf
// of `sample_brdf` producing `l`.

brdf_eval :: proc(v, l: Vec3, roughness: f32) -> (value, pdf: f32) {
	if v.z <= 0 || l.z <= 0 {
		return 0, 0
	}
	h := linalg.normalize(v + l)
	a2 := roughness * roughness
	d := ggx_d(h.z, a2)
	g := smith_g1(v.z, a2) * smith_g1(l.z, a2)
	value = d * g / (4 * v.z)
	pdf = d * h.z / (4 * max(linalg.dot(v, h), 1e-7))
	return
}

ggx_d :: proc(cos_h, a2: f32) -> f32 {
	if cos_h <= 0 { return 0 }
	denom := cos_h * cos_h * (a2 - 1) + 1
	return a2 / (math.PI * denom * denom)
}

smith_g1 :: proc(cos_v, a2: f32) -> f32 {
	if cos_v <= 0 { return 0 }
	c2 := cos_v * cos_v
	tan2 := (1 - c2) / max(c2, 1e-30)
	return 2 / (1 + math.sqrt(1 + a2 * tan2))
}

// A direction reflected about a GGX half vector, alpha = roughness.
sample_brdf :: proc(v: Vec3, roughness: f32, u1, u2: f32) -> Vec3 {
	a2 := roughness * roughness
	phi := 2 * math.PI * u1
	cos_t := math.sqrt((1 - u2) / (1 + (a2 - 1) * u2))
	sin_t := math.sqrt(max(1 - cos_t * cos_t, 0))
	h := Vec3{sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t}
	return 2 * linalg.dot(v, h) * h - v
}

// ── the transformed cosine ───────────────────────────────────────────────────

// The fitted distribution: a clamped cosine pushed through `m`.
Ltc :: struct {
	m:         Mat3,
	inv:       Mat3,
	det_inv:   f32,
	magnitude: f32,
}

ltc_make :: proc(m: Mat3, magnitude: f32) -> Ltc {
	inv := linalg.inverse(m)
	return {m = m, inv = inv, det_inv = abs(linalg.determinant(inv)), magnitude = magnitude}
}

// Density of direction `l` (unit), without the magnitude.
ltc_density :: proc(t: Ltc, l: Vec3) -> f32 {
	lo := t.inv * l
	len := linalg.length(lo)
	lo /= len
	cos_o := max(lo.z, 0) / math.PI
	// The Jacobian of normalize(M^-1 l): det(M^-1) / |M^-1 l|^3.
	return cos_o * t.det_inv / (len * len * len)
}

ltc_sample :: proc(t: Ltc, u1, u2: f32) -> Vec3 {
	r := math.sqrt(u1)
	phi := 2 * math.PI * u2
	lo := Vec3{r * math.cos(phi), r * math.sin(phi), math.sqrt(max(1 - u1, 0))}
	return linalg.normalize(t.m * lo)
}

// ── polygon integral, mirroring ltc_polygon in shading.slang ────────────────

// The integral of a clamped cosine (normalized: it integrates to 1 over the
// hemisphere) over a polygon given by its corners as directions from the
// shading point, already in the cosine's own frame. Two-sided: a polygon
// seen from behind counts the same.
//
// What the shader does: no clipping. The edges are summed unclipped into the
// vector form factor F, which is exact for the unclipped polygon; the horizon
// is then accounted for as it would be for a spherical cap with the same F,
// from the `sphere` table. Exact while the polygon is wholly above the
// horizon, and close where it crosses -- see the tests. Clipping exactly
// instead needs arrays indexed at run time, which a GPU keeps in memory
// rather than registers.
polygon_form_factor :: proc(table: ^Table, corners: []Vec3) -> f32 {
	f: Vec3
	n := len(corners)
	for i in 0 ..< n {
		f += edge_vector(linalg.normalize(corners[i]), linalg.normalize(corners[(i + 1) % n]))
	}
	length := linalg.length(f)
	if length <= 0 {
		return 0
	}
	// Two-sided: orient F toward the polygon whichever way it is wound.
	if linalg.dot(f, corners[0]) < 0 {
		f = -f
	}
	return length * sphere_lookup(table, f.z / length, length)
}

// The `sphere` table at elevation z and form factor `length`, bilinearly, as
// the shader's sampler reads it.
sphere_lookup :: proc(table: ^Table, z, length: f32) -> f32 {
	fx := clamp(z * 0.5 + 0.5, 0, 1) * (SIZE - 1)
	fy := clamp(length, 0, 1) * (SIZE - 1)
	x0 := min(int(fx), SIZE - 2)
	y0 := min(int(fy), SIZE - 2)
	tx := fx - f32(x0)
	ty := fy - f32(y0)
	at :: proc(t: ^Table, x, y: int) -> f32 { return t[y * SIZE + x].sphere }
	top := at(table, x0, y0) * (1 - tx) + at(table, x0 + 1, y0) * tx
	bottom := at(table, x0, y0 + 1) * (1 - tx) + at(table, x0 + 1, y0 + 1) * tx
	return top * (1 - ty) + bottom * ty
}

// The exact integral, clipped at the horizon: the reference the tests hold
// `polygon_form_factor` to.
polygon_integral_exact :: proc(corners: []Vec3) -> f32 {
	clipped: [16]Vec3
	n := clip_to_horizon(corners, clipped[:])
	if n < 3 {
		return 0
	}
	sum: f32 = 0
	for i in 0 ..< n {
		a := linalg.normalize(clipped[i])
		b := linalg.normalize(clipped[(i + 1) % n])
		sum += edge_integral(a, b)
	}
	return abs(sum)
}

// One edge's term: the z component of cross(a, b) * theta / sin(theta) / 2pi,
// with Hill's rational fit for theta / (2pi sin(theta)) -- the 1/2pi is part
// of the fit -- which stays accurate where acos in single precision does not.
edge_integral :: proc(a, b: Vec3) -> f32 {
	return edge_vector(a, b).z
}

edge_vector :: proc(a, b: Vec3) -> Vec3 {
	x := linalg.dot(a, b)
	y := abs(x)
	fa := 0.8543985 + (0.4965155 + 0.0145206 * y) * y
	fb := 3.4175940 + (4.1616724 + y) * y
	v := fa / fb
	theta_sin := x > 0 ? v : 0.5 / math.sqrt(max(1 - x * x, 1e-7)) - v
	return linalg.cross(a, b) * theta_sin
}

// Sutherland-Hodgman against z >= 0. Returns the clipped corner count.
clip_to_horizon :: proc(corners: []Vec3, out: []Vec3) -> int {
	n := 0
	count := len(corners)
	for i in 0 ..< count {
		a := corners[i]
		b := corners[(i + 1) % count]
		if a.z >= 0 {
			out[n] = a
			n += 1
		}
		if (a.z >= 0) != (b.z >= 0) {
			t := a.z / (a.z - b.z)
			out[n] = a + (b - a) * t
			out[n].z = 0
			n += 1
		}
	}
	return n
}

// The shading-side reconstruction of a table entry.
entry_inverse :: proc(e: Entry) -> Mat3 {
	return Mat3{
		e.inv.x, 0, e.inv.y,
		0,       1, 0,
		e.inv.z, 0, e.inv.w,
	}
}
