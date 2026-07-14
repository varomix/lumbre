package main

import "core:c"
import "core:fmt"
import "core:strings"
import stbi "vendor:stb/image"
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

	stbi.flip_vertically_on_write(true)
	path_str := string(file_output)
	if strings.has_suffix(path_str, ".exr") {
		// Beauty is always the first layer; enabled AOVs (albedo, normal,
		// depth, direct, indirect) are appended.
		img: output.EXR_Image
		output.exr_image_init(&img, image_width, image_height)
		img.compression = output.EXR_COMPRESSION_ZIP if exr_compress else output.EXR_COMPRESSION_NONE
		defer output.exr_destroy(&img)

		rgba_chans := []output.EXR_Channel{
			{name = "R", component = 0, pixel_type = 1, x_sampling = 1, y_sampling = 1},
			{name = "G", component = 1, pixel_type = 1, x_sampling = 1, y_sampling = 1},
			{name = "B", component = 2, pixel_type = 1, x_sampling = 1, y_sampling = 1},
			{name = "A", component = 3, pixel_type = 1, x_sampling = 1, y_sampling = 1},
		}
		output.exr_add_layer(&img, "", rgba_chans[:], frame.beauty_linear)

		if enable_aovs {
			aov_passes_seen := []int{1, 2, 3, 5, 9}
			aov_layer_names := make(map[int]string)
			defer delete(aov_layer_names)
			aov_layer_names[1] = "albedo"
			aov_layer_names[2] = "normal"
			aov_layer_names[3] = "depth"
			aov_layer_names[5] = "direct"
			aov_layer_names[9] = "indirect"
			for debug in aov_passes_seen {
				_, ok := frame.aov_results[debug]
				if !ok {
					continue
				}
				aov_chans := []output.EXR_Channel{
					{name = "R", component = 0, pixel_type = 1, x_sampling = 1, y_sampling = 1},
					{name = "G", component = 1, pixel_type = 1, x_sampling = 1, y_sampling = 1},
					{name = "B", component = 2, pixel_type = 1, x_sampling = 1, y_sampling = 1},
					{name = "A", component = 3, pixel_type = 1, x_sampling = 1, y_sampling = 1},
				}
				output.exr_add_layer(&img, aov_layer_names[debug], aov_chans[:], frame.aov_results[debug][:])
			}
		}

		if !output.exr_write_file(&img, path_str) {
			fmt.eprintln("Failed to write EXR")
			return
		}
		fmt.println("Wrote", file_output, "(EXR,", len(img.layers), "layers)")
		return
	}

	ok := stbi.write_png(
		file_output,
		c.int(image_width),
		c.int(image_height),
		3,
		raw_data(frame.pixels),
		c.int(image_width * 3),
	)
	if ok == 0 {
		fmt.eprintln("Failed to write PNG")
		return
	}
	fmt.println("Wrote", file_output)
}
