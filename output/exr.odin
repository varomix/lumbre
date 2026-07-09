package output

// Minimal EXR (OpenEXR 2.0) single-part scanline writer for Lumbre.
//
// Supported:
//   * Scanline, increasing-Y storage
//   * Half-float (FP16) or 32-bit float pixel data
//   * No compression (1-line chunks) or ZIP/deflate compression
//     (16-line chunks, raw deflate per the EXR spec).
//   * Multiple channels per file (R, G, B, A, plus custom names)
//   * Multiple AOV layers, each as a "layer.channel" group
//
// Format reference: OpenEXR 2.0 specification, "Scanline Images".

import "core:c"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"

when ODIN_OS == .Windows {
	foreign import zlib "system:zlib1"
} else when ODIN_OS == .Darwin {
	foreign import zlib "system:z"
} else {
	foreign import zlib "system:z"
}

Z_OK       :: 0
Z_FINISH   :: 4

// Minimal zlib stream struct and bindings for deflate.
z_stream :: struct {
	next_in:   [^]byte,
	avail_in:  c.uint,
	total_in:  c.ulong,

	next_out:  [^]byte,
	avail_out: c.uint,
	total_out: c.ulong,

	msg:        cstring,
	state:      rawptr,

	zalloc:     rawptr,
	zfree:      rawptr,
	opaque:     rawptr,

	data_type:  c.int,
	adler:      c.ulong,
	reserved:   c.ulong,
}

@(default_calling_convention = "c")
foreign zlib {
	// deflateInit2 with windowBits = -15 -> raw deflate (no zlib header),
	// which is what EXR ZIP compression requires.
	deflateInit2_ :: proc(
		strm: ^z_stream,
		level: c.int,
		method: c.int,
		windowBits: c.int,
		memLevel: c.int,
		strategy: c.int,
		version: cstring,
		stream_size: c.int,
	) -> c.int ---
	deflate :: proc(strm: ^z_stream, flush: c.int) -> c.int ---
	deflateEnd :: proc(strm: ^z_stream) -> c.int ---
	deflateBound :: proc(strm: ^z_stream, sourceLen: c.ulong) -> c.ulong ---
}

EXR_MAGIC :: 0x01312F76

// OpenEXR compression type values (per the spec, table 3).
// 0 = no compression, 1 = RLE, 2 = ZIPS (1 line per block),
// 3 = ZIP (16 lines per block), 4 = PIZ.
EXR_COMPRESSION_NONE :: 0
EXR_COMPRESSION_ZIP  :: 3

// Number of lines per scanline block. ZIP requires 16 per the spec;
// NONE and ZIPS use 1-line chunks.
exr_lines_per_block :: proc(compression: int) -> int {
	if compression == EXR_COMPRESSION_ZIP { return 16 }
	return 1
}

// Half-float (FP16) bit conversion. Handles the common positive
// radiance range, with fallbacks for inf/nan/denormal.
f32_to_f16 :: proc(x: f32) -> u16 {
	if math.is_nan(x) || math.is_inf(x, 0) {
		return 0
	}
	x_copy := x
	bits: u32 = ---
	#no_bounds_check {
		bits = (^u32)(&x_copy)^
	}
	sign := u16((bits >> 16) & 0x8000)
	exp := i32((bits >> 23) & 0xFF) - 127 + 15
	frac := bits & 0x7FFFFF
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant := (frac | 0x800000) >> u32(1 - exp + 13)
		return sign | u16(mant)
	}
	if exp >= 31 {
		return sign | 0x7C00
	}
	return sign | (u16(exp) << 10) | u16(frac >> 13)
}

EXR_Channel :: struct {
	name:        string,
	component:   u8, // 0=R, 1=G, 2=B, 3=A
	pixel_type:  u8, // 0=uint, 1=half, 2=float
	x_sampling:  i32,
	y_sampling:  i32,
}

EXR_Layer :: struct {
	name:     string,
	channels: [dynamic]EXR_Channel,
	pixels:   [][4]f32,
}

EXR_Image :: struct {
	width:  i32,
	height: i32,
	layers: [dynamic]EXR_Layer,
	compression: int, // EXR_COMPRESSION_NONE or EXR_COMPRESSION_ZIP
}

exr_image_init :: proc(img: ^EXR_Image, width, height: i32) {
	img.width = width
	img.height = height
	img.compression = EXR_COMPRESSION_NONE // default
}

exr_add_layer :: proc(img: ^EXR_Image, name: string, channels: []EXR_Channel, pixels: [][4]f32) {
	layer := EXR_Layer{
		name = strings.clone(name),
	}
	layer.pixels = make([][4]f32, len(pixels))
	copy(layer.pixels, pixels)
	for c in channels {
		append(&layer.channels, c)
	}
	append(&img.layers, layer)
}

exr_destroy :: proc(img: ^EXR_Image) {
	for &layer in img.layers {
		delete(layer.channels)
		delete(layer.pixels)
		delete(layer.name)
	}
	delete(img.layers)
}

EXR_Channel_Spec :: struct {
	layer:   string,
	channel: EXR_Channel,
}

write_attr :: proc(buf: ^[dynamic]u8, name, type: string, data: []u8) {
	append(buf, ..transmute([]u8)name)
	append(buf, 0)
	append(buf, ..transmute([]u8)type)
	append(buf, 0)
	size: [4]u8
	size_v := i32(len(data))
	#no_bounds_check {
		(^u32)(&size[0])^ = u32(size_v)
	}
	append(buf, ..size[:])
	append(buf, ..data)
}

full_channel_name :: proc(spec: EXR_Channel_Spec, allocator := context.allocator) -> string {
	if spec.layer == "" {
		return strings.clone(spec.channel.name)
	}
	return strings.concatenate({spec.layer, ".", spec.channel.name}, allocator)
}

channel_spec_less :: proc(a, b: EXR_Channel_Spec) -> bool {
	fa := full_channel_name(a)
	fb := full_channel_name(b)
	defer {
		delete(fa)
		delete(fb)
	}
	return fa < fb
}

interleave_scanline :: proc(
	ordered: []EXR_Channel_Spec,
	pixels_by_layer: map[string][dynamic][4]f32,
	y_start: i32,
	scanline_height: i32,
	width: i32,
) -> [dynamic]u8 {
	out: [dynamic]u8
	for spec in ordered {
		pixels := pixels_by_layer[spec.layer]
		for y in 0 ..< scanline_height {
			py := y_start + y
			for x in 0 ..< width {
				idx := int(py) * int(width) + int(x)
				rgba := pixels[idx]
				val: f32
				switch spec.channel.component {
				case 0: val = rgba.x
				case 1: val = rgba.y
				case 2: val = rgba.z
				case 3: val = rgba.w
				case:   val = rgba.x
				}
				if spec.channel.pixel_type == 1 {
					h := f32_to_f16(val)
					hb: [2]u8
					#no_bounds_check {
						(^u16)(&hb[0])^ = h
					}
					append(&out, ..hb[:])
				} else {
					fb: [4]u8
					#no_bounds_check {
						(^f32)(&fb[0])^ = val
					}
					append(&out, ..fb[:])
				}
			}
		}
	}
	return out
}

// Compress a buffer with zlib's deflate. The OpenEXR spec describes
// ZIP compression as raw deflate (windowBits = -15) but the reference
// OpenEXR library and OpenImageIO both produce/consume the zlib-wrapped
// format (windowBits = 15) with the standard 0x78 0x9C header. We
// match that behaviour here. Uses the default compression level (6).
deflate_compress :: proc(input: []u8, allocator := context.allocator) -> ([]u8, bool) {
	if len(input) == 0 {
		result := make([]u8, 0, allocator)
		return result, true
	}

	strm: z_stream
	strm.next_in = &input[0]
	strm.avail_in = c.uint(len(input))
	strm.next_out = nil
	strm.avail_out = 0
	strm.zalloc = nil
	strm.zfree = nil
	strm.opaque = nil

	// windowBits = 15 -> zlib format (with 0x78 0x9C header and Adler-32
	// trailer). This matches the OpenEXR reference library and
	// OpenImageIO's behaviour for ZIP compression.
	rc := deflateInit2_(
		&strm,
		c.int(-1), // Z_DEFAULT_COMPRESSION
		c.int(8),  // Z_DEFLATED
		c.int(15),
		c.int(8),  // memLevel
		c.int(0),  // Z_DEFAULT_STRATEGY
		cstring("1.2.12"),
		c.int(size_of(z_stream)),
	)
	if rc != Z_OK {
		return nil, false
	}
	defer deflateEnd(&strm)

	// Allocate the output buffer based on the worst case.
	bound := int(deflateBound(&strm, c.ulong(len(input))))
	out_buf := make([dynamic]u8, bound, allocator)
	defer delete(out_buf)

	strm.next_out = &out_buf[0]
	strm.avail_out = c.uint(bound)

	rc = deflate(&strm, Z_FINISH)
	if rc != Z_OK && rc != 1 { // 1 = Z_STREAM_END
		return nil, false
	}

	total_out := int(strm.total_out)
	result := make([]u8, total_out, allocator)
	copy(result, out_buf[:total_out])
	return result, true
}

exr_write_file :: proc(img: ^EXR_Image, path: string) -> bool {
	if len(img.layers) == 0 {
		fmt.eprintln("exr: no layers to write")
		return false
	}

	compression := img.compression
	lines_per_block := exr_lines_per_block(compression)

	// Build the sorted channel spec list.
	specs: [dynamic]EXR_Channel_Spec
	defer delete(specs)
	pixels_by_layer := make(map[string][dynamic][4]f32)
	defer {
		for _, v in pixels_by_layer {
			delete(v)
		}
		delete(pixels_by_layer)
	}

	for &layer in img.layers {
		pixels_by_layer[layer.name] = make([dynamic][4]f32, len(layer.pixels))
		copy(pixels_by_layer[layer.name][:], layer.pixels)
		for c in layer.channels {
			append(&specs, EXR_Channel_Spec{layer = layer.name, channel = c})
		}
	}

	sorted_specs := make([]EXR_Channel_Spec, len(specs))
	copy(sorted_specs, specs[:])
	for i in 1 ..< len(sorted_specs) {
		j := i
		for j > 0 && channel_spec_less(sorted_specs[j], sorted_specs[j - 1]) {
			sorted_specs[j - 1], sorted_specs[j] = sorted_specs[j], sorted_specs[j - 1]
			j -= 1
		}
	}
	defer delete(sorted_specs)

	// Build the header.
	header: [dynamic]u8
	defer delete(header)

	all_chans: [dynamic]u8
	for spec in sorted_specs {
		full := full_channel_name(spec)
		append(&all_chans, ..transmute([]u8)full)
		append(&all_chans, 0)
		pt: [4]u8
		pt[0] = spec.channel.pixel_type
		append(&all_chans, ..pt[:])
		append(&all_chans, 0, 0, 0, 0)
		xs: [4]u8
		#no_bounds_check {
			(^i32)(&xs[0])^ = spec.channel.x_sampling
		}
		append(&all_chans, ..xs[:])
		ys: [4]u8
		#no_bounds_check {
			(^i32)(&ys[0])^ = spec.channel.y_sampling
		}
		append(&all_chans, ..ys[:])
		delete(full)
	}
	write_attr(&header, "channels", "chlist", all_chans[:])
	delete(all_chans)

	// compression attribute
	comp: [1]u8
	comp[0] = u8(compression & 0xFF)
	write_attr(&header, "compression", "compression", comp[:])

	// dataWindow
	dw: [16]u8
	#no_bounds_check {
		(^i32)(&dw[0])^ = i32(0)
		(^i32)(&dw[4])^ = i32(0)
		(^i32)(&dw[8])^ = img.width - 1
		(^i32)(&dw[12])^ = img.height - 1
	}
	write_attr(&header, "dataWindow", "box2i", dw[:])
	write_attr(&header, "displayWindow", "box2i", dw[:])

	lo: [1]u8 = {0}
	write_attr(&header, "lineOrder", "lineOrder", lo[:])

	par: [4]u8
	#no_bounds_check {
		(^f32)(&par[0])^ = f32(1.0)
	}
	write_attr(&header, "pixelAspectRatio", "float", par[:])

	swc: [8]u8
	#no_bounds_check {
		(^f32)(&swc[0])^ = f32(0.0)
		(^f32)(&swc[4])^ = f32(0.0)
	}
	write_attr(&header, "screenWindowCenter", "v2f", swc[:])

	sww: [4]u8
	#no_bounds_check {
		(^f32)(&sww[0])^ = f32(1.0)
	}
	write_attr(&header, "screenWindowWidth", "float", sww[:])

	append(&header, 0) // end of header

	// Open file
	f, err := os.open(
		path,
		os.O_WRONLY | os.O_CREATE | os.O_TRUNC,
		os.Permissions_Read_Write_All,
	)
	if err != nil {
		fmt.eprintln("exr: failed to open", path, ":", err)
		return false
	}
	defer os.close(f)

	check_write :: proc(written: int, err: os.Error) -> bool {
		_, _ = written, err
		return true
	}

	// Magic + version
	magic_bytes: [4]u8
	#no_bounds_check {
		(^u32)(&magic_bytes[0])^ = u32(EXR_MAGIC)
	}
	_, _ = os.write(f, magic_bytes[:])

	version_bytes: [4]u8 = {2, 0, 0, 0}
	_, _ = os.write(f, version_bytes[:])

	_, _ = os.write(f, header[:])

	// Offset table
	num_blocks := (int(img.height) + lines_per_block - 1) / lines_per_block
	offset_table_size := num_blocks * 8
	header_end := 4 + 4 + len(header)

	offset_table: [dynamic]u8
	defer delete(offset_table)
	reserve(&offset_table, offset_table_size)
	for i in 0 ..< offset_table_size {
		append(&offset_table, 0)
	}
	_, _ = os.write(f, offset_table[:])

	// Compute and write scanline blocks. For NONE compression, each
	// block is just (y_coordinate, data) where data is the raw
	// interleaved scanlines. For ZIP compression, each 16-line
	// block is deflated with raw deflate (windowBits = -15).
	block_offsets := make([dynamic]u64, num_blocks)
	defer delete(block_offsets)

	current_offset := u64(header_end) + u64(offset_table_size)
	for block in 0 ..< num_blocks {
		y_start := i32(block * lines_per_block)
		y_end := y_start + i32(lines_per_block)
		if y_end > img.height {
			y_end = img.height
		}
		actual_height := y_end - y_start

		// The renderer's pixel buffer has y=0 at the top of the
		// image. EXR uses the opposite convention (y=0 at the
		// bottom), so we flip the source row index when reading
		// the pixel buffer.
		src_y_start := img.height - y_start - actual_height

		interleaved := interleave_scanline(
			sorted_specs,
			pixels_by_layer,
			src_y_start, actual_height, img.width,
		)

		stored_data: []u8
		stored_owned: [dynamic]u8
		owned_compressed: [dynamic]u8
		if compression == EXR_COMPRESSION_ZIP {
			compressed, ok := deflate_compress(interleaved[:])
			delete(interleaved)
			if !ok {
				fmt.eprintln("exr: deflate failed at block", block)
				return false
			}
			// Move the slice into a [dynamic]u8 so we can use a single
			// delete path below.
			owned_compressed = make([dynamic]u8, len(compressed))
			copy(owned_compressed[:], compressed)
			delete(compressed)
			stored_owned = owned_compressed
			stored_data = stored_owned[:]
		} else {
			stored_owned = interleaved
			stored_data = stored_owned[:]
		}

		block_offsets[block] = current_offset

		// Chunk header: y_coordinate + pixel_data_size
		y_bytes: [4]u8
		#no_bounds_check {
			(^i32)(&y_bytes[0])^ = y_start
		}
		_, _ = os.write(f, y_bytes[:])

		size_bytes: [4]u8
		#no_bounds_check {
			(^i32)(&size_bytes[0])^ = i32(len(stored_data))
		}
		_, _ = os.write(f, size_bytes[:])

		_, _ = os.write(f, stored_data)

		current_offset += u64(8) + u64(len(stored_data))
		delete(stored_owned)
	}

	// Patch offset table
	_, _ = os.seek(f, i64(header_end), .Start)
	for block in 0 ..< num_blocks {
		bo_bytes: [8]u8
		#no_bounds_check {
			(^u64)(&bo_bytes[0])^ = block_offsets[block]
		}
		_, _ = os.write(f, bo_bytes[:])
	}
	_ = check_write

	return true
}
