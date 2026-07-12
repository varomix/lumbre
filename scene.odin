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
		is_usd := strings.has_suffix(lower, ".usd") || strings.has_suffix(lower, ".usda") ||
			strings.has_suffix(lower, ".usdc") || strings.has_suffix(lower, ".usdz")
		data: ObjData
		usd_cameras: []Usd_Camera_Info
		usd_lights: []Usd_Light_Info
		ok: bool
		if is_gltf {
			data, ok = load_gltf(string(cfg.scene_file))
		} else if is_usd {
			data, usd_cameras, usd_lights, ok = load_usd(string(cfg.scene_file))
		} else {
			data, ok = load_obj(string(cfg.scene_file))
		}
		if !ok {
			return {}, false
		}
		defer {
			for cam in usd_cameras {
				delete(cam.name)
			}
			delete(usd_cameras)
			for lt in usd_lights {
				if lt.texture_file != "" {
					delete(lt.texture_file)
				}
			}
			delete(usd_lights)
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
		// Consolidate the legacy Lambertian/Metal/Dielectric kinds onto the
		// Principled model (Phase D). After this the mesh path only ever holds
		// Principled + Emissive materials.
		for &mat in data.materials {
			mat = normalize_material(mat)
		}
		// Diagnostic override: force anisotropy on Principled materials so the
		// anisotropic path can be exercised without an importer that authors it.
		if cfg.force_anisotropic != 0.0 {
			for &mat in data.materials {
				if mat.kind == .Principled {
					mat.anisotropic = cfg.force_anisotropic
				}
			}
		}
		if cfg.force_spec_trans != 0.0 {
			for &mat in data.materials {
				if mat.kind == .Principled {
					mat.spec_trans = cfg.force_spec_trans
				}
			}
		}
		// Heuristic camera: if the scene is small (Cornell-box-like),
		// place the camera inside; otherwise use an exterior view.
		//
		// Bounds must be measured in world space. A mesh's triangles are
		// stored untransformed, with the mesh's placement in its
		// transform, so a file that carries a scale on an Xform (common
		// in USD, where glTF exporters tend to bake it into the vertices)
		// would otherwise be framed at its unscaled size and put the
		// camera inside the geometry.
		bounds_min := Vec3{1.0e30, 1.0e30, 1.0e30}
		bounds_max := Vec3{-1.0e30, -1.0e30, -1.0e30}
		for mesh in data.meshes {
			for tri in mesh.triangles {
			corners := [3]Vec3{tri.v0, tri.v1, tri.v2}
			for v in corners {
				wv := transform_point(v, mesh.transform)
				bounds_min = m.min(bounds_min, wv)
				bounds_max = m.max(bounds_max, wv)
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
			// The six bounding-box faces as (pinned coordinate, its value,
			// which way points out of the box).
			face_axis := [6]int{0, 0, 1, 1, 2, 2}
			face_plane := [6]f64{
				bounds_min.x, bounds_max.x,
				bounds_min.y, bounds_max.y,
				bounds_min.z, bounds_max.z,
			}
			face_outward := [6]Vec3{
				{-1, 0, 0}, {1, 0, 0},
				{0, -1, 0}, {0, 1, 0},
				{0, 0, -1}, {0, 0, 1},
			}
			for mesh in data.meshes {
				for tri in mesh.triangles {
					// World space, like the bounds these are compared against.
					// Measuring untransformed triangles against world bounds
					// makes any mesh with a transform miss its own bounding-box
					// faces, so a translated box reads as a room with a hole in it.
					a := transform_point(tri.v0, mesh.transform)
					b := transform_point(tri.v1, mesh.transform)
					c := transform_point(tri.v2, mesh.transform)
					cr := m.cross(b - a, c - a)
					if m.length(cr) <= 0.0 {
						continue
					}
					normal := m.normalize(cr)
					for k in 0 ..< 6 {
						if m.abs(a[face_axis[k]] - face_plane[k]) >= eps ||
						   m.abs(b[face_axis[k]] - face_plane[k]) >= eps ||
						   m.abs(c[face_axis[k]] - face_plane[k]) >= eps {
							continue
						}
						// A room's walls face inward, towards the camera that
						// will sit between them. A solid box flush with the
						// scene bounds -- a UsdGeomCube, say -- puts triangles
						// on the very same planes but turns them outward, and
						// has no inside worth looking at.
						if m.dot(normal, face_outward[k]) < 0.0 {
							face_counts[k] += 1
						}
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

		// Looking straight along the up axis leaves the camera basis
		// degenerate, since cross(vup, view) is then zero. A room open at its
		// floor or ceiling is what reaches this.
		vup := Vec3{0.0, 1.0, 0.0}
		if view := lookat - lookfrom; m.length(view) > 0.0 && m.abs(m.normalize(view).y) > 0.999 {
			vup = Vec3{0.0, 0.0, 1.0}
		}

		aspect_ratio := f64(cfg.image_width) / f64(cfg.image_height)

		// A USD scene with its own Camera prim carries explicit framing
		// intent; use it instead of the auto-framing heuristic above, which
		// exists only because OBJ/glTF/camera-less USD scenes never carry
		// one. Default to the first Camera found in traversal order;
		// --usd-camera selects by prim name when a stage has several.
		camera := make_camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus)
		if len(usd_cameras) > 0 {
			chosen := 0
			if cfg.usd_camera_name != "" {
				chosen = -1
				for cam, i in usd_cameras {
					if cam.name == string(cfg.usd_camera_name) {
						chosen = i
						break
					}
				}
				if chosen == -1 {
					fmt.eprintln("Warning: --usd-camera", cfg.usd_camera_name, "not found in stage; using first camera")
					chosen = 0
				}
			}
			camera = usd_make_camera_from_info(usd_cameras[chosen], aspect_ratio)
		}

		scene := Scene {
			meshes    = data.meshes,
			materials = data.materials,
			camera    = camera,
		}

		// USD lights: a DomeLight becomes Scene.environment (unless --hdri
		// overrides it); every other kind converts to a Light. See
		// usd_light_import.odin.
		usd_dome_info: Usd_Light_Info
		has_usd_dome := false
		if len(usd_lights) > 0 {
			converted: [dynamic]Light
			for info in usd_lights {
				if info.kind == .Dome {
					if !has_usd_dome {
						usd_dome_info = info
						has_usd_dome = true
					}
					continue
				}
				if light, lok := usd_make_light_from_info(info); lok {
					append(&converted, light)
				}
			}
			if len(converted) > 0 {
				scene.lights = make([]Light, len(converted))
				copy(scene.lights, converted[:])
			}
			delete(converted)
		}
		if has_usd_dome {
			if cfg.hdri_file != "" {
				fmt.eprintln("Warning: scene has a USD DomeLight but --hdri was also given; --hdri wins")
			} else if env, eok := usd_dome_to_environment(usd_dome_info); eok {
				scene.environment = env
			} else {
				fmt.eprintln("Warning: failed to load USD DomeLight texture, continuing without it:", usd_dome_info.texture_file)
			}
		}

		// Only synthesize a fallback area light when the scene has no
		// lighting of its own at all: no emissive materials, no HDRI dome
		// (CLI or USD), no sun, and no USD analytic lights.
		if !has_emissive && cfg.hdri_file == "" && !has_usd_dome && !cfg.sun_enabled && len(scene.lights) == 0 {
			hd := max_dim * 0.5
			scene.lights = make([]Light, 1)
			scene.lights[0] = make_area_light(
				center + Vec3{0.0, hd, hd},
				Vec3{hd * 0.8, 0.0, 0.0},
				Vec3{0.0, -hd * 0.8, 0.0},
				Color{hd * 0.3, hd * 0.3, hd * 0.3},
			)
		}

		finalize_lighting(&scene, cfg)
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
	// Showcase one of each light type over the sphere field.
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
	disc_light := make_disc_light(
		Point3{-2.0, 4.0, 3.0},
		Vec3{0.4, -1.0, -0.3}, // normal aimed down at the spheres
		1.5,
		Color{12.0, 8.0, 6.0},
	)
	cylinder_light := make_cylinder_light(
		Point3{0.0, 0.5, -5.0},
		Vec3{0.0, 1.0, 0.0},
		0.25, 3.0,
		Color{4.0, 6.0, 12.0},
	)
	point_light := make_point_light(
		Point3{4.0, 3.0, -1.0},
		Color{40.0, 30.0, 20.0},
	)
	spot_light := make_spot_light(
		Point3{-4.0, 5.0, 2.0},
		Vec3{4.0, -5.0, -2.0}, // aim toward the origin
		degrees_to_radians(12.0),
		degrees_to_radians(22.0),
		Color{160.0, 160.0, 180.0},
	)
	scene.lights = make([]Light, 6)
	scene.lights[0] = quad_light
	scene.lights[1] = sphere_light
	scene.lights[2] = disc_light
	scene.lights[3] = cylinder_light
	scene.lights[4] = point_light
	scene.lights[5] = spot_light
	finalize_lighting(&scene, cfg)
	debug_print_lights(scene.lights)
	build_sphere_scene_graph(&scene)
	return scene, true
}

// Load the HDRI environment (if requested) and append the sun light. The dome
// itself is stored in `scene.environment` and lit via ray escape + environment
// NEE, so it needs no entry in `scene.lights`.
finalize_lighting :: proc(scene: ^Scene, cfg: Render_Config) {
	if cfg.hdri_file != "" {
		env, ok := load_environment(string(cfg.hdri_file), cfg.hdri_rotation, cfg.hdri_intensity)
		if ok {
			scene.environment = env
		} else {
			fmt.eprintln("Warning: failed to load HDRI, continuing without it:", cfg.hdri_file)
		}
	}

	if cfg.sun_enabled {
		half_angle := cfg.sun_angle * 0.5 * m.PI / 180.0
		sun := make_distant_light(cfg.sun_dir, half_angle, cfg.sun_color)
		combined := make([]Light, len(scene.lights) + 1)
		copy(combined[:len(scene.lights)], scene.lights)
		combined[len(scene.lights)] = sun
		delete(scene.lights)
		scene.lights = combined
	}
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
	destroy_environment(&scene.environment)
	delete(scene.nodes)
	delete(scene.meshes)
	delete(scene.spheres)
	delete(scene.lights)
	delete(scene.materials)
}
