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

Material_Kind :: enum {
	Lambertian,
}

Material :: struct {
	kind:   Material_Kind,
	albedo: Color,
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
	return Vec3{
		random_f64_range(min, max),
		random_f64_range(min, max),
		random_f64_range(min, max),
	}
}

random_in_unit_sphere :: proc() -> Vec3 {
	for {
		p := random_vec3_range(-1.0, 1.0)
		if length_squared(p) < 1.0 {
			return p
		}
	}
}

random_unit_vector :: proc() -> Vec3 {
	return m.normalize(random_in_unit_sphere())
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
	material_ground := Material{.Lambertian, Color{0.8, 0.8, 0.0}}
	material_center := Material{.Lambertian, Color{0.7, 0.3, 0.3}}

	world := [?]Sphere {
		Sphere{Point3{0.0, 0.0, -1.0}, 0.5, material_center},
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

scatter :: proc(material: Material, r_in: Ray, rec: Hit_Record, attenuation: ^Color, scattered: ^Ray) -> bool {
	switch material.kind {
	case .Lambertian:
		scatter_direction := rec.normal + random_unit_vector()
		if near_zero(scatter_direction) {
			scatter_direction = rec.normal
		}

		scattered^ = Ray{rec.p, scatter_direction}
		attenuation^ = material.albedo
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
	aspect_ratio := 16.0 / 9.0
	image_width := 400
	image_height := i32(f64(image_width) / aspect_ratio)
	samples_per_pixel := i32(50)
	max_depth := i32(50)

	viewport_height := 2.0
	viewport_width := aspect_ratio * viewport_height
	focal_length := 1.0

	origin := Point3{0.0, 0.0, 0.0}
	horizontal := Vec3{viewport_width, 0.0, 0.0}
	vertical := Vec3{0.0, viewport_height, 0.0}
	lower_left_corner := origin - horizontal / 2.0 - vertical / 2.0 - Vec3{0.0, 0.0, focal_length}

	fmt.printf("P3\n%d %d\n255\n", image_width, image_height)

	for j := image_height - 1; j >= 0; j -= 1 {
		for i := 0; i < image_width; i += 1 {
			pixel_color := Color{0.0, 0.0, 0.0}

			for sample := i32(0); sample < samples_per_pixel; sample += 1 {
				u := (f64(i) + random_f64()) / f64(image_width - 1)
				v := (f64(j) + random_f64()) / f64(image_height - 1)
				r := Ray{origin, lower_left_corner + u * horizontal + v * vertical}
				pixel_color += ray_color(r, max_depth)
			}

			write_color(pixel_color, samples_per_pixel)
		}
	}
}
