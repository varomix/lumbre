package main

import m "core:math/linalg/glsl"
import "core:math/rand"

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
