package main

import "core:fmt"
import m "core:math/linalg/glsl"
import "core:math/rand"

Vec3 :: m.dvec3
Point3 :: m.dvec3
Color :: m.dvec3

Ray :: struct {
	origin:    Point3,
	direction: Vec3,
}

Camera :: struct {
	origin:            Point3,
	lower_left_corner: Point3,
	horizontal:        Vec3,
	vertical:          Vec3,
	u:                 Vec3,
	v:                 Vec3,
	lens_radius:       f64,
}

Material_Kind :: enum {
	Lambertian,
	Metal,
	Dielectric,
}

Material :: struct {
	kind:   Material_Kind,
	albedo: Color,
	fuzz:   f64,
	ir:     f64,
}

Hit_Record :: struct {
	p:          Point3,
	normal:     Vec3,
	t:          f64,
	front_face: bool,
	material:   Material,
}

Sphere :: struct {
	center:   Point3,
	radius:   f64,
	material: Material,
}

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

get_ray :: proc(camera: Camera, s, t: f64) -> Ray {
	rd := camera.lens_radius * random_in_unit_disk()
	offset := camera.u * rd.x + camera.v * rd.y

	return Ray{
		camera.origin + offset,
		camera.lower_left_corner + s * camera.horizontal + t * camera.vertical - camera.origin - offset,
	}
}

set_face_normal :: proc(rec: ^Hit_Record, r: Ray, outward_normal: Vec3) {
	rec.front_face = m.dot(r.direction, outward_normal) < 0.0
	rec.normal = outward_normal if rec.front_face else -outward_normal
}

random_f64 :: proc() -> f64 {
	return rand.float64()
}

random_f64_range :: proc(min, max: f64) -> f64 {
	return rand.float64_range(min, max)
}

random_vec3_range :: proc(min, max: f64) -> Vec3 {
	return Vec3{random_f64_range(min, max), random_f64_range(min, max), random_f64_range(min, max)}
}

random_in_unit_sphere :: proc() -> Vec3 {
	for {
		p := random_vec3_range(-1.0, 1.0)
		if length_squared(p) < 1.0 {
			return p
		}
	}
}

random_in_unit_disk :: proc() -> Vec3 {
	for {
		p := Vec3{random_f64_range(-1.0, 1.0), random_f64_range(-1.0, 1.0), 0.0}
		if length_squared(p) < 1.0 {
			return p
		}
	}
}

random_unit_vector :: proc() -> Vec3 {
	return m.normalize(random_in_unit_sphere())
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

write_color :: proc(pixel_color: Color, samples_per_pixel: i32) {
	scale := 1.0 / f64(samples_per_pixel)
	r := m.sqrt(scale * pixel_color.x)
	g := m.sqrt(scale * pixel_color.y)
	b := m.sqrt(scale * pixel_color.z)

	r = m.clamp(r, 0.0, 0.999)
	g = m.clamp(g, 0.0, 0.999)
	b = m.clamp(b, 0.0, 0.999)

	ir := i32(256.0 * r)
	ig := i32(256.0 * g)
	ib := i32(256.0 * b)

	fmt.printf("%d %d %d\n", ir, ig, ib)
}

hit_sphere :: proc(sphere: Sphere, r: Ray, ray_t_min, ray_t_max: f64, rec: ^Hit_Record) -> bool {
	oc := r.origin - sphere.center
	a := length_squared(r.direction)
	half_b := m.dot(oc, r.direction)
	c := length_squared(oc) - sphere.radius * sphere.radius

	discriminant := half_b * half_b - a * c
	if discriminant < 0.0 {
		return false
	}

	sqrtd := m.sqrt(discriminant)
	root := (-half_b - sqrtd) / a
	if root <= ray_t_min || ray_t_max <= root {
		root = (-half_b + sqrtd) / a
		if root <= ray_t_min || ray_t_max <= root {
			return false
		}
	}

	rec.t = root
	rec.p = at(r, rec.t)
	outward_normal := (rec.p - sphere.center) / sphere.radius
	set_face_normal(rec, r, outward_normal)
	rec.material = sphere.material
	return true
}

hit_world :: proc(r: Ray, ray_t_min, ray_t_max: f64, rec: ^Hit_Record) -> bool {
	material_ground := Material{.Lambertian, Color{0.8, 0.8, 0.0}, 0.0, 1.0}
	material_center := Material{.Lambertian, Color{0.1, 0.2, 0.5}, 0.0, 1.0}
	material_left := Material{.Dielectric, Color{1.0, 1.0, 1.0}, 0.0, 1.5}
	material_right := Material{.Metal, Color{0.8, 0.6, 0.2}, 0.0, 1.0}

	world := [?]Sphere {
		Sphere{Point3{0.0, 0.0, -1.0}, 0.5, material_center},
		Sphere{Point3{-1.0, 0.0, -1.0}, 0.5, material_left},
		Sphere{Point3{-1.0, 0.0, -1.0}, -0.4, material_left},
		Sphere{Point3{1.0, 0.0, -1.0}, 0.5, material_right},
		Sphere{Point3{0.0, -100.5, -1.0}, 100.0, material_ground},
	}

	temp_rec: Hit_Record
	hit_anything := false
	closest_so_far := ray_t_max

	for sphere in world {
		if hit_sphere(sphere, r, ray_t_min, closest_so_far, &temp_rec) {
			hit_anything = true
			closest_so_far = temp_rec.t
			rec^ = temp_rec
		}
	}

	return hit_anything
}

scatter :: proc(
	material: Material,
	r_in: Ray,
	rec: Hit_Record,
	attenuation: ^Color,
	scattered: ^Ray,
) -> bool {
	switch material.kind {
	case .Lambertian:
		scatter_direction := rec.normal + random_unit_vector()
		if near_zero(scatter_direction) {
			scatter_direction = rec.normal
		}

		scattered^ = Ray{rec.p, scatter_direction}
		attenuation^ = material.albedo
		return true

	case .Metal:
		fuzz := m.min(material.fuzz, 1.0)
		reflected := reflect(m.normalize(r_in.direction), rec.normal)
		scattered^ = Ray{rec.p, reflected + fuzz * random_in_unit_sphere()}
		attenuation^ = material.albedo
		return m.dot(scattered.direction, rec.normal) > 0.0

	case .Dielectric:
		attenuation^ = Color{1.0, 1.0, 1.0}
		refraction_ratio := 1.0 / material.ir if rec.front_face else material.ir

		unit_direction := m.normalize(r_in.direction)
		cos_theta := m.min(m.dot(-unit_direction, rec.normal), 1.0)
		sin_theta := m.sqrt(1.0 - cos_theta * cos_theta)

		cannot_refract := refraction_ratio * sin_theta > 1.0
		direction: Vec3
		if cannot_refract || reflectance(cos_theta, refraction_ratio) > random_f64() {
			direction = reflect(unit_direction, rec.normal)
		} else {
			direction = refract(unit_direction, rec.normal, refraction_ratio)
		}

		scattered^ = Ray{rec.p, direction}
		return true
	}

	return false
}

ray_color :: proc(r: Ray, depth: i32) -> Color {
	if depth <= 0 {
		return Color{0.0, 0.0, 0.0}
	}

	rec: Hit_Record
	if hit_world(r, 0.001, 1.0e30, &rec) {
		scattered: Ray
		attenuation: Color
		if scatter(rec.material, r, rec, &attenuation, &scattered) {
			return attenuation * ray_color(scattered, depth - 1)
		}

		return Color{0.0, 0.0, 0.0}
	}

	unit_direction := m.normalize(r.direction)
	a := 0.5 * (unit_direction.y + 1.0)
	return (1.0 - a) * Color{1.0, 1.0, 1.0} + a * Color{0.5, 0.7, 1.0}
}

main :: proc() {
	rand.reset(42)

	aspect_ratio := 16.0 / 9.0
	image_width := 400
	image_height := i32(f64(image_width) / aspect_ratio)
	samples_per_pixel := i32(100)
	max_depth := i32(50)

	lookfrom := Point3{3.0, 3.0, 2.0}
	lookat := Point3{0.0, 0.0, -1.0}
	vup := Vec3{0.0, 1.0, 0.0}
	dist_to_focus := m.length(lookfrom - lookat)
	aperture := 2.0

	camera := make_camera(
		lookfrom,
		lookat,
		vup,
		20.0,
		aspect_ratio,
		aperture,
		dist_to_focus,
	)

	fmt.printf("P3\n%d %d\n255\n", image_width, image_height)

	for j := image_height - 1; j >= 0; j -= 1 {
		for i := 0; i < image_width; i += 1 {
			pixel_color := Color{0.0, 0.0, 0.0}

			for sample := i32(0); sample < samples_per_pixel; sample += 1 {
				u := (f64(i) + random_f64()) / f64(image_width - 1)
				v := (f64(j) + random_f64()) / f64(image_height - 1)
				r := get_ray(camera, u, v)
				pixel_color += ray_color(r, max_depth)
			}

			write_color(pixel_color, samples_per_pixel)
		}
	}
}
