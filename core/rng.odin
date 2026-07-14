package lumbre_core

import m "core:math/linalg/glsl"

global_bvh_rng: Rng

rng_next :: proc(rng: ^Rng) -> u64 {
	rng.state += 0x9e3779b97f4a7c15
	z := rng.state
	z = (z ~ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ~ (z >> 27)) * 0x94d049bb133111eb
	z = z ~ (z >> 31)
	return z
}

rng_f64 :: proc(rng: ^Rng) -> f64 {
	return f64(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0)
}

rng_f64_range :: proc(rng: ^Rng, min, max: f64) -> f64 {
	return min + (max - min) * rng_f64(rng)
}

rng_vec3_range :: proc(rng: ^Rng, min, max: f64) -> Vec3 {
	return Vec3 {
		rng_f64_range(rng, min, max),
		rng_f64_range(rng, min, max),
		rng_f64_range(rng, min, max),
	}
}

rng_in_unit_sphere :: proc(rng: ^Rng) -> Vec3 {
	for {
		p := rng_vec3_range(rng, -1.0, 1.0)
		if length_squared(p) < 1.0 {
			return p
		}
	}
}

rng_in_unit_disk :: proc(rng: ^Rng) -> Vec3 {
	for {
		p := Vec3{rng_f64_range(rng, -1.0, 1.0), rng_f64_range(rng, -1.0, 1.0), 0.0}
		if length_squared(p) < 1.0 {
			return p
		}
	}
}

rng_unit_vector :: proc(rng: ^Rng) -> Vec3 {
	return m.normalize(rng_in_unit_sphere(rng))
}
