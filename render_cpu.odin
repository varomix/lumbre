package main

import "core:c"
import "core:fmt"
import "core:sys/info"
import "core:thread"
import stbi "vendor:stb/image"
import m "core:math/linalg/glsl"

INV_PI :: 0.31830988618379067154

scatter :: proc(
	material: Material,
	r_in: Ray,
	rec: Hit_Record,
	attenuation: ^Color,
	scattered: ^Ray,
	rng: ^Rng,
	roughness_cutoff: f64 = 0.95,
) -> bool {
	mat_kind := material.kind
	if roughness_cutoff > 0.0 && mat_kind == .Principled {
		if material.roughness > roughness_cutoff {
			mat_kind = .Lambertian
		}
	}
	switch mat_kind {
	case .Lambertian, .Principled:
		scatter_direction := rec.normal + rng_unit_vector(rng)
		if near_zero(scatter_direction) {
			scatter_direction = rec.normal
		}
		scattered^ = Ray{rec.p, scatter_direction}
		attenuation^ = material.albedo
		return true

	case .Metal:
		fuzz := min(material.fuzz, 1.0)
		reflected := reflect(m.normalize(r_in.direction), rec.normal)
		scattered^ = Ray{rec.p, reflected + fuzz * rng_in_unit_sphere(rng)}
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
		if cannot_refract || reflectance(cos_theta, refraction_ratio) > rng_f64(rng) {
			direction = reflect(unit_direction, rec.normal)
		} else {
			direction = refract(unit_direction, rec.normal, refraction_ratio)
		}

		scattered^ = Ray{rec.p, direction}
		return true

	case .Emissive:
		return false
	}

	return false
}

hit_scene :: proc(spheres: []Sphere, sphere_nodes: []BVH_Node, sphere_bvh_root: i32, triangles: []Triangle, tri_nodes: []BVH_Node, tri_bvh_root: i32, mats: []Material, r: Ray) -> (Hit_Record, bool) {
	rec: Hit_Record
	hit_anything := false
	closest := 1.0e30

	if len(spheres) > 0 && sphere_bvh_root >= 0 {
		temp_rec: Hit_Record
		if hit_bvh(spheres, sphere_nodes, sphere_bvh_root, r, 0.001, closest, &temp_rec) {
			hit_anything = true
			closest = temp_rec.t
			rec = temp_rec
		}
	}

	if len(triangles) > 0 && tri_bvh_root >= 0 {
		temp_rec: Hit_Record
		if hit_triangle_bvh(triangles, tri_nodes, tri_bvh_root, r, 0.001, closest, &temp_rec) {
			if temp_rec.mat_idx >= 0 && int(temp_rec.mat_idx) < len(mats) {
				temp_rec.material = mats[temp_rec.mat_idx]
			}
			hit_anything = true
			closest = temp_rec.t
			rec = temp_rec
		}
	}

	return rec, hit_anything
}

ray_color :: proc(
	spheres: []Sphere, sphere_nodes: []BVH_Node, sphere_bvh_root: i32,
	triangles: []Triangle, tri_nodes: []BVH_Node, tri_bvh_root: i32,
	mats: []Material, lights: []Light,
	r: Ray, depth: i32, max_radiance: f64, rng: ^Rng,
	roughness_cutoff: f64 = 0.95,
) -> Color {
	if depth <= 0 {
		return Color{0.0, 0.0, 0.0}
	}

	rec, hit := hit_scene(spheres, sphere_nodes, sphere_bvh_root, triangles, tri_nodes, tri_bvh_root, mats, r)
	if hit {
		if rec.material.kind == .Emissive {
			return rec.material.emission * rec.material.emission_strength
		}

		radiance := Color{0.0, 0.0, 0.0}

		// Direct light sampling (NEE) for diffuse surfaces
		if rec.material.kind == .Lambertian || rec.material.kind == .Principled {
			light_count := total_light_count(lights)
			if light_count > 0 {
				for l in lights {
					if l.kind != .Quad && l.kind != .Sphere {
						continue
					}
					ls := sample_light(l, rec.p, rng)
					if ls.pdf <= 0.0 {
						continue
					}

					// Shadow ray
					shadow_ray := Ray{rec.p + 1.0e-4 * ls.direction, ls.direction}
					shadow_rec, shadow_hit := hit_scene(spheres, sphere_nodes, sphere_bvh_root, triangles, tri_nodes, tri_bvh_root, mats, shadow_ray)

					if !shadow_hit || shadow_rec.t > ls.distance - 1.0e-4 {
						cos_surf := max(m.dot(rec.normal, ls.direction), 0.0)
						if cos_surf > 0.0 {
							bsdf_pdf := cos_surf * INV_PI
							if bsdf_pdf > 0.0 {
								light_weight := ls.pdf
								mis_weight := light_weight * light_weight / (light_weight * light_weight + bsdf_pdf * bsdf_pdf)
								brdf := rec.material.albedo * INV_PI
								direct := ls.emission * brdf * cos_surf * mis_weight / ls.pdf
								radiance += direct
							}
						}
					}
				}
			}
		}

		scattered: Ray
		attenuation: Color
		if scatter(rec.material, r, rec, &attenuation, &scattered, rng, roughness_cutoff) {
			indirect := attenuation * ray_color(spheres, sphere_nodes, sphere_bvh_root, triangles, tri_nodes, tri_bvh_root, mats, lights, scattered, depth - 1, max_radiance, rng, roughness_cutoff)
			radiance += indirect
		}

		// Firefly clamping
		if max_radiance > 0.0 {
			lum := radiance.x * 0.2126 + radiance.y * 0.7152 + radiance.z * 0.0722
			if lum > max_radiance {
				scale := max_radiance / lum
				radiance = radiance * scale
			}
		}
		return radiance
	}

	unit_direction := m.normalize(r.direction)
	a := 0.5 * (unit_direction.y + 1.0)
	return (1.0 - a) * Color{1.0, 1.0, 1.0} + a * Color{0.5, 0.7, 1.0}
}

Render_Work :: struct {
	row_start:         i32,
	row_end:           i32,
	image_width:       i32,
	image_height:      i32,
	samples_per_pixel: i32,
	max_depth:         i32,
	max_radiance:      f64,
	bytes_per_pixel:   i32,
	pixels:            []u8,
	scene:             ^Scene,
	sphere_nodes:      []BVH_Node,
	sphere_bvh_root:   i32,
	triangles:         []Triangle,
	tri_nodes:         []BVH_Node,
	tri_bvh_root:      i32,
	materials:         []Material,
	lights:            []Light,
	roughness_cutoff:  f64,
	seed:              u64,
}

render_worker :: proc(data: rawptr) {
	work := cast(^Render_Work)data
	rng := Rng {
		state = work.seed,
	}

	for row := work.row_start; row < work.row_end; row += 1 {
		j := work.image_height - 1 - row
		for i := i32(0); i < work.image_width; i += 1 {
			pixel_color := Color{0.0, 0.0, 0.0}

			for sample := i32(0); sample < work.samples_per_pixel; sample += 1 {
				u := (f64(i) + rng_f64(&rng)) / f64(work.image_width - 1)
				v := (f64(j) + rng_f64(&rng)) / f64(work.image_height - 1)
				r := get_ray(work.scene.camera, u, v, &rng)
				pixel_color += ray_color(work.scene.spheres, work.sphere_nodes, work.sphere_bvh_root, work.triangles, work.tri_nodes, work.tri_bvh_root, work.materials, work.lights, r, work.max_depth, work.max_radiance, &rng, work.roughness_cutoff)
			}

			pixel_index := int((row * work.image_width + i) * work.bytes_per_pixel)
			write_color(pixel_color, work.samples_per_pixel, work.pixels, pixel_index)
		}
	}
}

render_cpu :: proc(
	scene: ^Scene,
	image_width, image_height: i32,
	samples_per_pixel, max_depth: i32,
	max_radiance: f64,
	file_output: cstring,
	roughness_cutoff: f64 = 0.95,
) {
	global_bvh_rng = Rng{state = 42}

	flattened := flatten_scene_graph(scene)
	defer destroy_flattened_scene(flattened)

	sphere_bvh_nodes: [MAX_BVH_NODES]BVH_Node
	sphere_node_count: i32 = 0
	sphere_bvh_root: i32 = -1
	var_sphere_slice: []BVH_Node

	if len(scene.spheres) > 0 {
		sphere_bvh_root = build_bvh(scene.spheres, &sphere_bvh_nodes, &sphere_node_count, 0, i32(len(scene.spheres)))
		var_sphere_slice = sphere_bvh_nodes[:sphere_node_count]
	}

	tri_bvh_nodes: [MAX_BVH_NODES]BVH_Node
	tri_node_count: i32 = 0
	tri_bvh_root: i32 = -1
	var_tri_slice: []BVH_Node

	if len(flattened.triangles) > 0 {
		tri_bvh_root = build_triangle_bvh(flattened.triangles, &tri_bvh_nodes, &tri_node_count, 0, i32(len(flattened.triangles)))
		var_tri_slice = tri_bvh_nodes[:tri_node_count]
	}

	bytes_per_pixel := i32(3)
	pixels := make([]u8, int(image_width * image_height * bytes_per_pixel))
	defer delete(pixels)

	_, logical_cpu, cpu_ok := info.cpu_core_count()
	num_threads := logical_cpu if cpu_ok else 4
	rows_per_thread := image_height / i32(num_threads)

	Work_State :: struct {
		thread: ^thread.Thread,
		work:   Render_Work,
	}

	states: [64]Work_State
	num_states := min(num_threads, len(states))
	num_states = max(num_states, 1)

	for i in 0 ..< num_states {
		row_start := i32(i) * rows_per_thread
		row_end := image_height if i == num_states - 1 else row_start + rows_per_thread

		states[i].work = Render_Work {
			row_start         = row_start,
			row_end           = row_end,
			image_width       = image_width,
			image_height      = image_height,
			samples_per_pixel = samples_per_pixel,
			max_depth         = max_depth,
			max_radiance      = max_radiance,
			bytes_per_pixel   = bytes_per_pixel,
			pixels            = pixels,
			scene             = scene,
			sphere_nodes      = var_sphere_slice,
			sphere_bvh_root   = sphere_bvh_root,
			triangles         = flattened.triangles,
			tri_nodes         = var_tri_slice,
			tri_bvh_root      = tri_bvh_root,
			materials         = flattened.materials,
			lights            = flattened.lights,
			roughness_cutoff  = roughness_cutoff,
			seed              = u64(i + 1),
		}

		states[i].thread = thread.create_and_start_with_data(&states[i].work, render_worker)
	}

	for i in 0 ..< num_states {
		thread.join(states[i].thread)
		thread.destroy(states[i].thread)
	}

	ok := stbi.write_png(
		file_output,
		c.int(image_width),
		c.int(image_height),
		c.int(bytes_per_pixel),
		raw_data(pixels),
		c.int(image_width * bytes_per_pixel),
	)
	if ok == 0 {
		fmt.eprintln("Failed to write %s", file_output)
		return
	}

	fmt.println("Wrote", file_output)
}
