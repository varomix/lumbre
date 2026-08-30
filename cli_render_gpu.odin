package main

import "core:fmt"
import "core:strings"
import "output"

// CLI adapter: run the shared GPU renderer (in core) and write the result to
// disk. PNG/EXR output — including AOV layers — lives here, out of the renderer
// core, so the Houdini bridge never pulls in file-writing code.
render_gpu :: proc(
	scene: ^Scene,
	image_width, image_height: i32,
	samples_per_pixel, max_depth: i32,
	max_radiance: f64,
	file_output: cstring,
	debug_mode: i32 = 0,
	roughness_cutoff: f64 = 0.95,
	glossy_bias: f64 = 0.0,
	gi_cache_enabled: b32 = true,
	gi_cache_distance: f32 = 0.0,
	gi_cache_normal_angle: f32 = 0.5,
	photon_enabled: b32 = true,
	photon_count: i32 = 200000,
	photon_radius: f32 = 0.0,
	photon_bounces: i32 = 8,
	enable_aovs: b32 = false,
	exr_compress: b32 = false,
	denoise_enabled: b32 = false,
) {
	frame := gpu_render_frame(scene, image_width, image_height, samples_per_pixel, max_depth,
		max_radiance, debug_mode, roughness_cutoff, glossy_bias,
		gi_cache_enabled, gi_cache_distance, gi_cache_normal_angle,
		photon_enabled, photon_count, photon_radius, photon_bounces,
		enable_aovs, exr_compress, denoise_enabled)
	defer destroy_gpu_frame(&frame)

	fmt.println("Writing", file_output)

	// The writer is shared with the GUI so an offline render started from the
	// viewport lands the same bytes on disk as the same render from here.
	msg, ok := output.write_gpu_frame(&frame, string(file_output), bool(enable_aovs), bool(exr_compress))
	defer delete(msg)
	if !ok {
		fmt.eprintln(msg)
		return
	}
	fmt.println(msg)
}
