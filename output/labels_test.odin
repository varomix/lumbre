package output

// Checks the label EXR: ids survive as exact integers, and the image comes out
// the right way up.
//
//   odin test output
//
// Orientation is the failure worth guarding. The EXR writer takes its pixel
// arrays bottom-row-first, because that is how the path tracer's GPU buffer
// arrives, while the rasterizer hands its label targets back top-row-first. A
// vertical flip that goes missing produces a file that looks entirely
// plausible — right ids, right counts, right boxes — and labels every object
// with the mask of whatever sits mirrored across the frame. Nothing but a
// pixel-level check catches it.

import "core:os"
import "core:path/filepath"
import "core:testing"

import lc "../core"

@(private = "file")
read_u32 :: proc(data: []u8, at: int) -> u32 {
	return(
		u32(data[at]) | u32(data[at + 1]) << 8 | u32(data[at + 2]) << 16 |
		u32(data[at + 3]) << 24 \
	)
}

@(private = "file")
read_i32 :: proc(data: []u8, at: int) -> i32 {
	return i32(read_u32(data, at))
}

// Walks an uncompressed scanline EXR to the pixel block for row `y`, returning
// the raw interleaved channel data for that row.
@(private = "file")
scanline_data :: proc(data: []u8, y, height: int) -> ([]u8, bool) {
	// Skip magic and version, then the header attributes, which end at a lone
	// null byte where an attribute name would start.
	at := 8
	for at < len(data) {
		if data[at] == 0 {
			at += 1
			break
		}
		// name, type, then a 4-byte size and that many bytes of value.
		for data[at] != 0 { at += 1 }
		at += 1
		for data[at] != 0 { at += 1 }
		at += 1
		size := int(read_i32(data, at))
		at += 4 + size
	}
	if at + height * 8 > len(data) {
		return nil, false
	}

	// The offset table is one u64 per scanline block; NONE compression puts
	// one row in each.
	offset := int(read_u32(data, at + y * 8))
	if offset + 8 > len(data) {
		return nil, false
	}
	row := int(read_i32(data, offset))
	size := int(read_i32(data, offset + 4))
	if row != y || offset + 8 + size > len(data) {
		return nil, false
	}
	return data[offset + 8:][:size], true
}

@(test)
test_label_exr_ids_and_orientation :: proc(t: ^testing.T) {
	// A 2×2 frame, top-row-first, with a distinct id in every texel so any
	// transposition or flip changes the answer.
	frame := lc.Label_Frame {
		width    = 2,
		height   = 2,
		instance = {10, 20, 30, 40},
		semantic = {1, 2, 3, 4},
		depth    = {1.5, 2.5, 3.5, 4.5},
		normal   = {{0, 1, 0, 1}, {0, 1, 0, 1}, {0, 1, 0, 1}, {0, 1, 0, 1}},
	}

	tmp, _ := os.temp_dir(context.temp_allocator)
	path, _ := filepath.join({tmp, "lumbre_labels_test.exr"}, context.temp_allocator)
	message, ok := write_label_exr(frame, path, false)
	defer delete(message)
	testing.expect(t, ok, "write_label_exr failed")
	defer os.remove(path)

	data, read_err := os.read_entire_file(path, context.temp_allocator)
	read_ok := read_err == nil
	testing.expect(t, read_ok, "could not read back the label EXR")
	if !read_ok {
		return
	}

	// Channels are sorted by full name: depth.Y, instance.Y, normal.X, .Y, .Z,
	// semantic.Y. Every one is 4 bytes wide here, so instance starts one
	// channel-row in.
	row_bytes :: 2 * 4
	instance_at :: row_bytes

	top, top_ok := scanline_data(data, 0, 2)
	testing.expect(t, top_ok, "could not locate scanline 0")
	bottom, bottom_ok := scanline_data(data, 1, 2)
	testing.expect(t, bottom_ok, "could not locate scanline 1")
	if !top_ok || !bottom_ok {
		return
	}

	// Scanline 0 is the top of the image, which is where ids 10 and 20 began.
	testing.expect_value(t, read_u32(top, instance_at), 10)
	testing.expect_value(t, read_u32(top, instance_at + 4), 20)
	testing.expect_value(t, read_u32(bottom, instance_at), 30)
	testing.expect_value(t, read_u32(bottom, instance_at + 4), 40)
}
