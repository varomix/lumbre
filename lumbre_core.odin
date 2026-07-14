package main

import "core:c"
import "core:fmt"
import stbi "vendor:stb/image"

// CLI render dispatch. `Lumbre_Core`, `lumbre_core_init`, and
// `lumbre_core_replace_triangles` are defined once in core/ and re-exported via
// core_aliases.odin; only the file-output adapter is CLI-specific.
//
// The renderer core produces an in-memory buffer (`render_cpu_to_buffer`); PNG
// and EXR writing stays here in the CLI adapter, out of the shared core.

lumbre_core_render_to_file :: proc(core: ^Lumbre_Core, output: cstring) {
	when ODIN_OS == .Darwin {
		if core.settings.use_gpu {
			render_gpu(&core.scene, core.settings.image_width, core.settings.image_height,
				core.settings.samples_per_pixel, core.settings.max_depth,
				core.settings.max_radiance, output, core.settings.debug_mode,
				core.settings.roughness_cutoff, core.settings.glossy_bias,
				core.settings.gi_cache_enabled, core.settings.gi_cache_distance,
				core.settings.gi_cache_normal_angle, core.settings.photon_enabled,
				core.settings.photon_count, core.settings.photon_radius,
				core.settings.photon_bounces, core.settings.enable_aovs,
				core.settings.exr_compress, core.settings.denoise_enabled)
			return
		}
	}

	buffer := render_cpu_to_buffer(&core.scene, core.settings.image_width, core.settings.image_height,
		core.settings.samples_per_pixel, core.settings.max_depth,
		core.settings.max_radiance, core.settings.roughness_cutoff,
		core.settings.glossy_bias)
	defer destroy_render_buffer(&buffer)
	write_png_buffer(output, buffer)
}

// CLI-only PNG writer for a core `Render_Buffer` (RGB8, top row first).
write_png_buffer :: proc(file_output: cstring, buffer: Render_Buffer) {
	bytes_per_pixel := i32(3)
	ok := stbi.write_png(
		file_output,
		c.int(buffer.width),
		c.int(buffer.height),
		c.int(bytes_per_pixel),
		raw_data(buffer.pixels),
		c.int(buffer.width * bytes_per_pixel),
	)
	if ok == 0 {
		fmt.eprintln("Failed to write", file_output)
		return
	}
	fmt.println("Wrote", file_output)
}
