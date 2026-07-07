package main

import "core:c"
import "core:fmt"
import m "core:math/linalg/glsl"
import "core:math/rand"
import stbi "vendor:stb/image"

Vec3 :: m.dvec3
Point3 :: m.dvec3
Color :: m.dvec3

MAX_SPHERES :: 500

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

	return Ray {
		camera.origin + offset,
		camera.lower_left_corner +
		s * camera.horizontal +
		t * camera.vertical -
		camera.origin -
		offset,
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

random_color :: proc() -> Color {
	return Color{random_f64(), random_f64(), random_f64()}
}

random_color_range :: proc(min, max: f64) -> Color {
	return Color {
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

write_color :: proc(pixel_color: Color, samples_per_pixel: i32, pixels: []u8, pixel_index: int) {
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

	pixels[pixel_index + 0] = u8(ir)
	pixels[pixel_index + 1] = u8(ig)
	pixels[pixel_index + 2] = u8(ib)
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

hit_world :: proc(world: []Sphere, r: Ray, ray_t_min, ray_t_max: f64, rec: ^Hit_Record) -> bool {
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

add_sphere :: proc(world: []Sphere, count: ^int, sphere: Sphere) {
	if count^ < len(world) {
		world[count^] = sphere
		count^ += 1
	}
}

random_scene :: proc(world: []Sphere) -> int {
	count := 0

	ground_material := Material{.Lambertian, Color{0.5, 0.5, 0.5}, 0.0, 1.0}
	add_sphere(world, &count, Sphere{Point3{0.0, -1000.0, 0.0}, 1000.0, ground_material})

	for a := -11; a < 11; a += 1 {
		for b := -11; b < 11; b += 1 {
			choose_mat := random_f64()
			center := Point3{f64(a) + 0.9 * random_f64(), 0.2, f64(b) + 0.9 * random_f64()}

			if m.length(center - Point3{4.0, 0.2, 0.0}) > 0.9 {
				if choose_mat < 0.8 {
					albedo := random_color() * random_color()
					material := Material{.Lambertian, albedo, 0.0, 1.0}
					add_sphere(world, &count, Sphere{center, 0.2, material})
				} else if choose_mat < 0.95 {
					albedo := random_color_range(0.5, 1.0)
					fuzz := random_f64_range(0.0, 0.5)
					material := Material{.Metal, albedo, fuzz, 1.0}
					add_sphere(world, &count, Sphere{center, 0.2, material})
				} else {
					material := Material{.Dielectric, Color{1.0, 1.0, 1.0}, 0.0, 1.5}
					add_sphere(world, &count, Sphere{center, 0.2, material})
				}
			}
		}
	}

	material_1 := Material{.Dielectric, Color{1.0, 1.0, 1.0}, 0.0, 1.5}
	material_2 := Material{.Lambertian, Color{0.4, 0.2, 0.1}, 0.0, 1.0}
	material_3 := Material{.Metal, Color{0.7, 0.6, 0.5}, 0.0, 1.0}

	add_sphere(world, &count, Sphere{Point3{0.0, 1.0, 0.0}, 1.0, material_1})
	add_sphere(world, &count, Sphere{Point3{-4.0, 1.0, 0.0}, 1.0, material_2})
	add_sphere(world, &count, Sphere{Point3{4.0, 1.0, 0.0}, 1.0, material_3})

	return count
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

ray_color :: proc(world: []Sphere, r: Ray, depth: i32) -> Color {
	if depth <= 0 {
		return Color{0.0, 0.0, 0.0}
	}

	rec: Hit_Record
	if hit_world(world, r, 0.001, 1.0e30, &rec) {
		scattered: Ray
		attenuation: Color
		if scatter(rec.material, r, rec, &attenuation, &scattered) {
			return attenuation * ray_color(world, scattered, depth - 1)
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
	image_width := i32(400)
	image_height := i32(f64(image_width) / aspect_ratio)
	samples_per_pixel := i32(50)
	max_depth := i32(20)

	world_storage: [MAX_SPHERES]Sphere
	world_count := random_scene(world_storage[:])
	world := world_storage[:world_count]

	lookfrom := Point3{13.0, 2.0, 3.0}
	lookat := Point3{0.0, 0.0, 0.0}
	vup := Vec3{0.0, 1.0, 0.0}
	dist_to_focus := 10.0
	aperture := 0.1

	camera := make_camera(lookfrom, lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus)

	bytes_per_pixel := i32(3)
	pixels := make([]u8, int(image_width * image_height * bytes_per_pixel))
	defer delete(pixels)

	for row := i32(0); row < image_height; row += 1 {
		j := image_height - 1 - row
		for i := i32(0); i < image_width; i += 1 {
			pixel_color := Color{0.0, 0.0, 0.0}

			for sample := i32(0); sample < samples_per_pixel; sample += 1 {
				u := (f64(i) + random_f64()) / f64(image_width - 1)
				v := (f64(j) + random_f64()) / f64(image_height - 1)
				r := get_ray(camera, u, v)
				pixel_color += ray_color(world, r, max_depth)
			}

			pixel_index := int((row * image_width + i) * bytes_per_pixel)
			write_color(pixel_color, samples_per_pixel, pixels, pixel_index)
		}
	}

	ok := stbi.write_png(
		cstring("image.png"),
		c.int(image_width),
		c.int(image_height),
		c.int(bytes_per_pixel),
		raw_data(pixels),
		c.int(image_width * bytes_per_pixel),
	)
	if ok == 0 {
		fmt.println("Failed to write image.png")
		return
	}

	fmt.println("Wrote image.png")
}
