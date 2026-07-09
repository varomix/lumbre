package main

import m "core:math/linalg/glsl"
import "core:math/rand"
import "core:fmt"
import "core:os"
import "core:strings"

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

	ground_material := Material{kind = .Lambertian, albedo = Color{0.5, 0.5, 0.5}, fuzz = 0.0, ir = 1.0}
	add_sphere(world, &count, Sphere{Point3{0.0, -1000.0, 0.0}, 1000.0, ground_material})

	for a := -11; a < 11; a += 1 {
		for b := -11; b < 11; b += 1 {
			choose_mat := random_f64()
			center := Point3{f64(a) + 0.9 * random_f64(), 0.2, f64(b) + 0.9 * random_f64()}

			if m.length(center - Point3{4.0, 0.2, 0.0}) > 0.9 {
				if choose_mat < 0.8 {
					albedo := random_color() * random_color()
					material := Material{kind = .Lambertian, albedo = albedo, fuzz = 0.0, ir = 1.0}
					add_sphere(world, &count, Sphere{center, 0.2, material})
				} else if choose_mat < 0.95 {
					albedo := random_color_range(0.5, 1.0)
					fuzz := random_f64_range(0.0, 0.5)
					material := Material{kind = .Metal, albedo = albedo, fuzz = fuzz, ir = 1.0}
					add_sphere(world, &count, Sphere{center, 0.2, material})
				} else {
					material := Material{kind = .Dielectric, albedo = Color{1.0, 1.0, 1.0}, fuzz = 0.0, ir = 1.5}
					add_sphere(world, &count, Sphere{center, 0.2, material})
				}
			}
		}
	}

	material_1 := Material{kind = .Dielectric, albedo = Color{1.0, 1.0, 1.0}, fuzz = 0.0, ir = 1.5}
	material_2 := Material{kind = .Lambertian, albedo = Color{0.4, 0.2, 0.1}, fuzz = 0.0, ir = 1.0}
	material_3 := Material{kind = .Metal, albedo = Color{0.7, 0.6, 0.5}, fuzz = 0.0, ir = 1.0}

	add_sphere(world, &count, Sphere{Point3{0.0, 1.0, 0.0}, 1.0, material_1})
	add_sphere(world, &count, Sphere{Point3{-4.0, 1.0, 0.0}, 1.0, material_2})
	add_sphere(world, &count, Sphere{Point3{4.0, 1.0, 0.0}, 1.0, material_3})

	return count
}

make_scene :: proc(cfg: Render_Config) -> (Scene, bool) {
	if cfg.scene_file != "" {
		lower := strings.to_lower(string(cfg.scene_file))
		is_gltf := strings.has_suffix(lower, ".gltf") || strings.has_suffix(lower, ".glb")
		data: ObjData
		ok: bool
		if is_gltf {
			data, ok = load_gltf(string(cfg.scene_file))
		} else {
			data, ok = load_obj(string(cfg.scene_file))
		}
		if !ok {
			return {}, false
		}
		// A file can carry geometry with no material at all (an .obj whose
		// .mtl is empty). Every triangle then indexes material 0, so give it
		// something to find instead of a zeroed, pitch-black struct.
		if len(data.materials) == 0 {
			data.materials = make([]Material, 1)
			data.materials[0] = Material{
				kind          = .Lambertian,
				albedo        = Color{0.8, 0.8, 0.8},
				ir            = 1.0,
				specular      = 0.5,
				specular_tint = Color{1.0, 1.0, 1.0},
			}
		}
		// Heuristic camera: if the scene is small (Cornell-box-like),
		// place the camera inside; otherwise use an exterior view.
		bounds_min := Vec3{1.0e30, 1.0e30, 1.0e30}
		bounds_max := Vec3{-1.0e30, -1.0e30, -1.0e30}
		for mesh in data.meshes {
			for tri in mesh.triangles {
			corners := [3]Vec3{tri.v0, tri.v1, tri.v2}
			for v in corners {
				bounds_min = m.min(bounds_min, v)
				bounds_max = m.max(bounds_max, v)
			}
			}
		}
		center := (bounds_min + bounds_max) * 0.5
		size := bounds_max - bounds_min
		max_dim := m.max(m.max(size.x, size.y), size.z)

		lookfrom: Point3 = Point3(center)
		lookat: Point3 = Point3(center)
		vfov: f64 = 40.0
		aperture: f64 = 0.0
		focus: f64 = 10.0

		// An interior camera only makes sense for an enclosed room. Count the
		// triangles that lie flat against each bounding-box face: a Cornell box
		// has walls on at least five of the six, while a lone object (a helmet,
		// a monkey head) has none. Without this check every small model gets
		// framed from inside its own bounding box, usually from behind.
		is_interior := false
		open_face := 0
		if max_dim > 0.0 && max_dim < 10.0 {
			face_counts := [6]int{}
			eps := max_dim * 1.0e-4
			for mesh in data.meshes {
				for tri in mesh.triangles {
					if m.abs(tri.v0.x - bounds_min.x) < eps && m.abs(tri.v1.x - bounds_min.x) < eps && m.abs(tri.v2.x - bounds_min.x) < eps {
						face_counts[0] += 1
					}
					if m.abs(tri.v0.x - bounds_max.x) < eps && m.abs(tri.v1.x - bounds_max.x) < eps && m.abs(tri.v2.x - bounds_max.x) < eps {
						face_counts[1] += 1
					}
					if m.abs(tri.v0.y - bounds_min.y) < eps && m.abs(tri.v1.y - bounds_min.y) < eps && m.abs(tri.v2.y - bounds_min.y) < eps {
						face_counts[2] += 1
					}
					if m.abs(tri.v0.y - bounds_max.y) < eps && m.abs(tri.v1.y - bounds_max.y) < eps && m.abs(tri.v2.y - bounds_max.y) < eps {
						face_counts[3] += 1
					}
					if m.abs(tri.v0.z - bounds_min.z) < eps && m.abs(tri.v1.z - bounds_min.z) < eps && m.abs(tri.v2.z - bounds_min.z) < eps {
						face_counts[4] += 1
					}
					if m.abs(tri.v0.z - bounds_max.z) < eps && m.abs(tri.v1.z - bounds_max.z) < eps && m.abs(tri.v2.z - bounds_max.z) < eps {
						face_counts[5] += 1
					}
				}
			}

			walls := 0
			for c in face_counts {
				if c >= 2 {
					walls += 1
				}
			}
			// Four walls is enough: a Cornell box is open at the front, and its
			// remaining wall is often a hair off the bounding box because the
			// original measurements are not exactly axis-aligned.
			is_interior = walls >= 4

			// Look in through the emptiest face, preferring +Z so a box that is
			// open on two sides is still viewed from the conventional front.
			preference := [6]int{5, 4, 1, 0, 3, 2}
			open_face = preference[0]
			for k in 1 ..< len(preference) {
				i := preference[k]
				if face_counts[i] < face_counts[open_face] {
					open_face = i
				}
			}
		}

		if is_interior {
			// Look in through the one face that has no wall.
			distance := max_dim * 1.5
			lookfrom = Point3(center)
			switch open_face {
			case 0: lookfrom.x = bounds_min.x - distance
			case 1: lookfrom.x = bounds_max.x + distance
			case 2: lookfrom.y = bounds_min.y - distance
			case 3: lookfrom.y = bounds_max.y + distance
			case 4: lookfrom.z = bounds_min.z - distance
			case 5: lookfrom.z = bounds_max.z + distance
			}
			lookat = Point3(center)
			vfov = 40.0
			aperture = 0.0
			focus = m.length(lookfrom - lookat)
		} else {
			// Exterior: three-quarter view from the front. glTF and OBJ both
			// put +Y up and the model's front toward +Z, so backing off along
			// +Z shows the face of the model rather than the back of its head.
			radius := 0.5 * m.length(size)
			if radius <= 0.0 {
				radius = 1.0
			}
			vfov = 35.0
			// Distance that fits a sphere of `radius` inside the vertical FOV.
			dist := radius / m.sin(degrees_to_radians(vfov * 0.5))
			dir := m.normalize(Vec3{0.55, 0.35, 1.0})
			lookfrom = Point3(center + dir * dist)
			lookat = Point3(center)
			aperture = 0.0
			focus = m.length(lookfrom - lookat)
		}

		has_emissive := false
		for mat in data.materials {
			if mat.kind == .Emissive {
				has_emissive = true
				break
			}
		}

		scene := Scene {
			meshes    = data.meshes,
			materials = data.materials,
			camera = make_camera(
				lookfrom, lookat,
				Vec3{0.0, 1.0, 0.0},
				vfov,
				f64(cfg.image_width) / f64(cfg.image_height),
				aperture,
				focus,
			),
		}

		if !has_emissive {
			hd := max_dim * 0.5
			scene.lights = make([]Light, 1)
			scene.lights[0] = make_area_light(
				center + Vec3{0.0, hd, hd},
				Vec3{hd * 0.8, 0.0, 0.0},
				Vec3{0.0, -hd * 0.8, 0.0},
				Color{hd * 0.3, hd * 0.3, hd * 0.3},
			)
		}

		build_default_scene_graph(&scene)
		return scene, true
	}

	// Default random sphere scene
	world_storage: [MAX_SPHERES]Sphere
	world_count := random_scene(world_storage[:])
	spheres := make([]Sphere, world_count)
	for i in 0 ..< world_count {
		spheres[i] = world_storage[i]
	}

	scene := Scene {
		spheres = spheres,
		camera  = make_camera(
			Point3{13.0, 2.0, 3.0},
			Point3{0.0, 0.0, 0.0},
			Vec3{0.0, 1.0, 0.0},
			20.0,
			f64(cfg.image_width) / f64(cfg.image_height),
			0.1,
			10.0,
		),
	}
	quad_light := make_area_light(
		Point3{-3.0, 5.0, -3.0},
		Vec3{6.0, 0.0, 0.0},
		Vec3{0.0, 0.0, 6.0},
		Color{15.0, 15.0, 15.0},
	)
	sphere_light := make_sphere_light(
		Point3{3.0, 3.0, 3.0},
		1.0,
		Color{10.0, 5.0, 2.0},
	)
	scene.lights = make([]Light, 2)
	scene.lights[0] = quad_light
	scene.lights[1] = sphere_light
	debug_print_lights(scene.lights)
	build_sphere_scene_graph(&scene)
	return scene, true
}

destroy_scene :: proc(scene: ^Scene) {
	for mesh in scene.meshes {
		delete(mesh.triangles)
		if mesh.name != "" {
			delete(mesh.name)
		}
	}
	for &mat in scene.materials {
		destroy_material_textures(&mat)
	}
	delete(scene.nodes)
	delete(scene.meshes)
	delete(scene.spheres)
	delete(scene.lights)
	delete(scene.materials)
}
