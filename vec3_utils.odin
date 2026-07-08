package main

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
