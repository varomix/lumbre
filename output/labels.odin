package output

// Writing a frame of ground truth to a multi-layer EXR.
//
// Separate from `write_gpu_frame` because labels are not a picture. Nothing
// here is tonemapped, gamma-encoded or clamped: an instance id written through
// an sRGB curve is not an id, and a depth clipped to [0, 1] is not a
// measurement. The beauty path and this one share the EXR container and
// nothing else.
//
// The input is plain CPU arrays rather than the renderer's own type, which is
// what keeps `output` free of a dependency on `realtime` — the CLI, the GUI
// and the tests all hand it the same four slices.

import "core:fmt"
import "core:strings"

// One frame of labels, top-row-first, as the GPU hands them back.
Label_Frame :: struct {
	width:    i32,
	height:   i32,
	instance: []u32,
	semantic: []u32,
	depth:    []f32,
	normal:   [][4]f32,
}

// A single-channel layer. EXR names channels within a layer, and a mask has
// only one thing to say, so "Y" — the luminance channel name — is what
// viewers expect to find and display.
@(private)
single_channel :: proc(pixel_type: u8) -> []EXR_Channel {
	@(static) chans: [1]EXR_Channel
	chans[0] = EXR_Channel {
		name = "Y",
		component = 0,
		pixel_type = pixel_type,
		x_sampling = 1,
		y_sampling = 1,
	}
	return chans[:]
}

// XYZ rather than RGB: this is a direction, and naming it as colour invites a
// viewer to apply a colour transform to it.
@(private)
normal_channels :: proc() -> []EXR_Channel {
	@(static) chans := [3]EXR_Channel {
		{name = "X", component = 0, pixel_type = 2, x_sampling = 1, y_sampling = 1},
		{name = "Y", component = 1, pixel_type = 2, x_sampling = 1, y_sampling = 1},
		{name = "Z", component = 2, pixel_type = 2, x_sampling = 1, y_sampling = 1},
	}
	return chans[:]
}

// Every label channel travels through the EXR writer's [4]f32 pixel slot; the
// channel's `pixel_type` decides how it lands in the file. Ids go in as their
// exact integer value and come out as UINT.
@(private)
pack_u32 :: proc(src: []u32, width, height: i32) -> [][4]f32 {
	out := make([][4]f32, len(src), context.temp_allocator)
	flip_into(out, width, height, proc(dst: ^[4]f32, src: []u32, i: int) {
		dst^ = {f32(src[i]), 0, 0, 0}
	}, src)
	return out
}

@(private)
pack_f32 :: proc(src: []f32, width, height: i32) -> [][4]f32 {
	out := make([][4]f32, len(src), context.temp_allocator)
	flip_into(out, width, height, proc(dst: ^[4]f32, src: []f32, i: int) {
		dst^ = {src[i], 0, 0, 0}
	}, src)
	return out
}

@(private)
pack_vec :: proc(src: [][4]f32, width, height: i32) -> [][4]f32 {
	out := make([][4]f32, len(src), context.temp_allocator)
	flip_into(out, width, height, proc(dst: ^[4]f32, src: [][4]f32, i: int) {
		dst^ = src[i]
	}, src)
	return out
}

// The EXR writer reads its pixel arrays bottom-row-first, matching the path
// tracer's GPU beauty buffer. Label targets come off the rasterizer the other
// way up, so the flip happens once, here, rather than in every packer.
@(private)
flip_into :: proc(
	out: [][4]f32,
	width, height: i32,
	set: proc(dst: ^[4]f32, src: $S, i: int),
	src: S,
) {
	w := int(width)
	for y in 0 ..< int(height) {
		src_row := y * w
		dst_row := (int(height) - 1 - y) * w
		for x in 0 ..< w {
			set(&out[dst_row + x], src, src_row + x)
		}
	}
}

// Writes instance, semantic, depth and normal as four layers of one EXR.
//
// Returns a description of what was written, or ok = false with the reason —
// the same contract as `write_gpu_frame`, so a caller can report either the
// same way.
write_label_exr :: proc(
	frame: Label_Frame,
	path: string,
	exr_compress: bool,
) -> (
	message: string,
	ok: bool,
) {
	texels := int(frame.width) * int(frame.height)
	if texels <= 0 {
		return "nothing to write (empty label frame)", false
	}
	if len(frame.instance) != texels || len(frame.semantic) != texels ||
	   len(frame.depth) != texels || len(frame.normal) != texels {
		return "label channels disagree about the frame size", false
	}

	img: EXR_Image
	exr_image_init(&img, frame.width, frame.height)
	img.compression = EXR_COMPRESSION_ZIP if exr_compress else EXR_COMPRESSION_NONE
	defer exr_destroy(&img)

	// Ids as UINT so nothing downstream can interpolate them; depth as full
	// float because half would quantise metres at any real scene scale.
	exr_add_layer(&img, "instance", single_channel(0), pack_u32(frame.instance, frame.width, frame.height))
	exr_add_layer(&img, "semantic", single_channel(0), pack_u32(frame.semantic, frame.width, frame.height))
	exr_add_layer(&img, "depth", single_channel(2), pack_f32(frame.depth, frame.width, frame.height))
	exr_add_layer(&img, "normal", normal_channels(), pack_vec(frame.normal, frame.width, frame.height))

	cpath := strings.clone_to_cstring(path, context.temp_allocator)
	if !exr_write_file(&img, string(cpath)) {
		return "failed to write label EXR", false
	}
	return fmt.aprintf("wrote %s (labels, %d layers)", path, len(img.layers)), true
}
