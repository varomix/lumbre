package output

// Writes a rendered frame to disk, chosen by extension: PNG for the 8-bit
// sRGB beauty, EXR for linear beauty plus any AOV layers.
//
// This lives here, shared by the CLI and the GUI, rather than in either
// frontend: both need the same file on disk, and an offline render started
// from the viewport should be byte-identical to the same render from the
// command line. The renderer core still knows nothing about files.

import "core:c"
import "core:fmt"
import "core:strings"

import lc "../core"
import stbi "vendor:stb/image"

// AOV debug-mode ids and the layer names they are written under. Order is the
// order they appear in the file.
//
// One table, not two parallel ones: the mode and the name it writes under
// belong together, and keeping them in separate slices indexed by position
// made adding a channel a two-place edit that silently mismatched if only one
// side was updated.
@(private)
AOV_Layer :: struct {
	mode: int,
	name: string,
}

@(private)
AOV_LAYERS := []AOV_Layer {
	{1, "albedo"},
	{2, "normal"},
	{3, "depth"},
	{5, "direct"},
	{9, "indirect"},
}

@(private)
rgba_channels :: proc() -> []EXR_Channel {
	@(static) chans := [4]EXR_Channel {
		{name = "R", component = 0, pixel_type = 1, x_sampling = 1, y_sampling = 1},
		{name = "G", component = 1, pixel_type = 1, x_sampling = 1, y_sampling = 1},
		{name = "B", component = 2, pixel_type = 1, x_sampling = 1, y_sampling = 1},
		{name = "A", component = 3, pixel_type = 1, x_sampling = 1, y_sampling = 1},
	}
	return chans[:]
}

// Returns a human-readable description of what was written, or ok = false with
// the reason. The caller decides how to report it — the CLI prints, the GUI
// puts it in the log.
write_gpu_frame :: proc(
	frame: ^lc.GPU_Frame,
	path: string,
	enable_aovs: bool,
	exr_compress: bool,
) -> (
	message: string,
	ok: bool,
) {
	if frame.pixels == nil || frame.width <= 0 || frame.height <= 0 {
		return "nothing to write (empty frame)", false
	}

	// The GPU beauty buffer is bottom-row-first; stb compensates on write and
	// the EXR writer expects the same orientation.
	stbi.flip_vertically_on_write(true)

	if strings.has_suffix(strings.to_lower(path, context.temp_allocator), ".exr") {
		img: EXR_Image
		exr_image_init(&img, frame.width, frame.height)
		img.compression = EXR_COMPRESSION_ZIP if exr_compress else EXR_COMPRESSION_NONE
		defer exr_destroy(&img)

		// Beauty is always the first layer; enabled AOVs follow.
		exr_add_layer(&img, "", rgba_channels(), frame.beauty_linear)

		if enable_aovs {
			for aov in AOV_LAYERS {
				layer, has := frame.aov_results[aov.mode]
				if !has {
					continue
				}
				exr_add_layer(&img, aov.name, rgba_channels(), layer[:])
			}
		}

		cpath := strings.clone_to_cstring(path, context.temp_allocator)
		if !exr_write_file(&img, string(cpath)) {
			return "failed to write EXR", false
		}
		return fmt.aprintf("wrote %s (EXR, %d layers)", path, len(img.layers)), true
	}

	cpath := strings.clone_to_cstring(path, context.temp_allocator)
	if stbi.write_png(
		   cpath,
		   c.int(frame.width),
		   c.int(frame.height),
		   3,
		   raw_data(frame.pixels),
		   c.int(frame.width * 3),
	   ) == 0 {
		return "failed to write PNG", false
	}
	return fmt.aprintf("wrote %s", path), true
}
