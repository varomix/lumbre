package main

import m "core:math/linalg/glsl"

degrees_to_radians :: proc(degrees: f64) -> f64 {
	return degrees * m.PI / 180.0
}

make_camera :: proc(
	lookfrom, lookat: Point3,
	vup: Vec3,
	vfov, aspect_ratio, aperture, focus_dist: f64,
) -> Camera {
	theta := degrees_to_radians(vfov)
	h := m.tan(theta / 2.0)
	viewport_height := 2.0 * h
	viewport_width := aspect_ratio * viewport_height

	w := m.normalize(lookfrom - lookat)
	u := m.normalize(m.cross(vup, w))
	v := m.cross(w, u)

	origin := lookfrom
	horizontal := focus_dist * viewport_width * u
	vertical := focus_dist * viewport_height * v
	lower_left_corner := origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w

	return Camera{origin, lower_left_corner, horizontal, vertical, u, v, aperture / 2.0}
}

get_ray :: proc(camera: Camera, s, t: f64, rng: ^Rng) -> Ray {
	rd := camera.lens_radius * rng_in_unit_disk(rng)
	offset := camera.u * rd.x + camera.v * rd.y

	return Ray {
		camera.origin + offset,
		camera.lower_left_corner +
		s * camera.horizontal +
		t * camera.vertical -
		camera.origin -
		offset,
	}
}
