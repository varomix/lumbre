package lumbre_core

import m "core:math/linalg/glsl"

length_squared :: proc(v: Vec3) -> f64 {
	return m.dot(v, v)
}

near_zero :: proc(v: Vec3) -> bool {
	epsilon := 1.0e-8
	return m.abs(v.x) < epsilon && m.abs(v.y) < epsilon && m.abs(v.z) < epsilon
}

at :: proc(r: Ray, t: f64) -> Point3 {
	return r.origin + t * r.direction
}

set_face_normal :: proc(rec: ^Hit_Record, r: Ray, outward_normal: Vec3) {
	rec.front_face = m.dot(r.direction, outward_normal) < 0.0
	rec.normal = outward_normal if rec.front_face else -outward_normal
}

reflect :: proc(v, n: Vec3) -> Vec3 {
	return v - 2.0 * m.dot(v, n) * n
}

refract :: proc(uv, n: Vec3, etai_over_etat: f64) -> Vec3 {
	cos_theta := m.min(m.dot(-uv, n), 1.0)
	r_out_perp := etai_over_etat * (uv + cos_theta * n)
	r_out_parallel := -m.sqrt(m.abs(1.0 - length_squared(r_out_perp))) * n
	return r_out_perp + r_out_parallel
}

reflectance :: proc(cosine, refraction_index: f64) -> f64 {
	r0 := (1.0 - refraction_index) / (1.0 + refraction_index)
	r0 *= r0
	return r0 + (1.0 - r0) * m.pow(1.0 - cosine, 5.0)
}

hit_triangle :: proc(tri: Triangle, r: Ray, t_min, t_max: f64, rec: ^Hit_Record) -> bool {
	edge1 := tri.v1 - tri.v0
	edge2 := tri.v2 - tri.v0
	h := m.cross(r.direction, edge2)
	a := m.dot(edge1, h)
	if a > -1.0e-8 && a < 1.0e-8 {
		return false
	}
	f := 1.0 / a
	s := r.origin - tri.v0
	u := f * m.dot(s, h)
	if u < 0.0 || u > 1.0 {
		return false
	}
	q := m.cross(s, edge1)
	v := f * m.dot(r.direction, q)
	if v < 0.0 || u + v > 1.0 {
		return false
	}
	t := f * m.dot(edge2, q)
	if t <= t_min || t >= t_max {
		return false
	}
	rec.t = t
	rec.p = at(r, t)
	w := 1.0 - u - v
	rec.normal = m.normalize(w * tri.n0 + u * tri.n1 + v * tri.n2)
	rec.uv = w * tri.uv0 + u * tri.uv1 + v * tri.uv2
	rec.front_face = m.dot(r.direction, rec.normal) < 0.0
	rec.mat_idx = tri.mat_idx
	return true
}
