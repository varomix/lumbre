package main

import "core:c"
import "core:fmt"
import "core:sys/info"
import "core:thread"
import stbi "vendor:stb/image"
import m "core:math/linalg/glsl"

scatter :: proc(
	material: Material,
	r_in: Ray,
	rec: Hit_Record,
	attenuation: ^Color,
	scattered: ^Ray,
	rng: ^Rng,
) -> bool {
	switch material.kind {
	case .Lambertian:
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
	}

	return false
}

ray_color :: proc(world: []Sphere, bvh_nodes: []BVH_Node, bvh_root: i32, r: Ray, depth: i32, rng: ^Rng) -> Color {
	if depth <= 0 {
		return Color{0.0, 0.0, 0.0}
	}

	rec: Hit_Record
	if hit_bvh(world, bvh_nodes, bvh_root, r, 0.001, 1.0e30, &rec) {
		scattered: Ray
		attenuation: Color
		if scatter(rec.material, r, rec, &attenuation, &scattered, rng) {
			return attenuation * ray_color(world, bvh_nodes, bvh_root, scattered, depth - 1, rng)
		}

		return Color{0.0, 0.0, 0.0}
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
	bytes_per_pixel:   i32,
	pixels:            []u8,
	world:             []Sphere,
	camera:            Camera,
	seed:              u64,
	bvh_nodes:         []BVH_Node,
	bvh_root:          i32,
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
				r := get_ray(work.camera, u, v, &rng)
				pixel_color += ray_color(work.world, work.bvh_nodes, work.bvh_root, r, work.max_depth, &rng)
			}

			pixel_index := int((row * work.image_width + i) * work.bytes_per_pixel)
			write_color(pixel_color, work.samples_per_pixel, work.pixels, pixel_index)
		}
	}
}

render_cpu :: proc(
	world: []Sphere,
	camera: Camera,
	image_width, image_height: i32,
	samples_per_pixel, max_depth: i32,
	file_output: cstring,
) {
	global_bvh_rng = Rng{state = 42}
	bvh_nodes: [MAX_BVH_NODES]BVH_Node
	bvh_node_count: i32 = 0
	bvh_root := build_bvh(world, &bvh_nodes, &bvh_node_count, 0, i32(len(world)))
	bvh_slice := bvh_nodes[:bvh_node_count]

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
			bytes_per_pixel   = bytes_per_pixel,
			pixels            = pixels,
			world             = world,
			camera            = camera,
			seed              = u64(i + 1),
			bvh_nodes         = bvh_slice,
			bvh_root          = bvh_root,
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
