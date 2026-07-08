package main

import "core:fmt"
import "core:math/rand"

USE_GPU :: true

main :: proc() {
	rand.reset(42)

	file_output: cstring = "render.png"

	aspect_ratio := 16.0 / 9.0
	image_width := i32(1024)
	image_height := i32(f64(image_width) / aspect_ratio)
	samples_per_pixel := i32(50)
	max_depth := i32(20)

	world_storage: [MAX_SPHERES]Sphere
	world_count := random_scene(world_storage[:])
	world := world_storage[:world_count]
	fmt.println("Spheres:", world_count)

	lookfrom := Point3{13.0, 2.0, 3.0}
	lookat := Point3{0.0, 0.0, 0.0}
	vup := Vec3{0.0, 1.0, 0.0}
	dist_to_focus := 10.0
	aperture := 0.1
	camera := make_camera(lookfrom, lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus)

	when ODIN_OS == .Darwin {
		if USE_GPU {
			render_gpu(world, camera, image_width, image_height, samples_per_pixel, max_depth, file_output)
			return
		}
	}

	render_cpu(world, camera, image_width, image_height, samples_per_pixel, max_depth, file_output)
}
