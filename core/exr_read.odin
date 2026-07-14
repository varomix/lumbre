package lumbre_core

import "core:c"
import "core:fmt"
import "core:os"

// Minimal OpenEXR scanline reader for HDRI environments. stb_image cannot read
// EXR, so DomeLights that point at a .exr fell through to a black environment.
// This covers the formats real-world HDRIs ship in: uncompressed, ZIP, and ZIPS
// scanline images with HALF or FLOAT RGB(A) channels. Tiled, multipart, deep,
// and the wavelet/lossy codecs (PIZ, PXR24, B44, DWA) are rejected with a clear
// message rather than decoded wrong.
//
// The ZIP path reverses the exact pipeline the EXR writer applies
// (output/exr.odin): zlib inflate, then un-predict (prefix sum), then
// un-reorder (interleave the even/odd byte halves).

when ODIN_OS == .Windows {
	foreign import zlib "system:zlib1"
} else {
	foreign import zlib "system:z"
}

@(default_calling_convention = "c")
foreign zlib {
	// One-shot zlib inflate. Returns Z_OK (0) on success.
	uncompress :: proc(dest: [^]u8, dest_len: ^c.ulong, source: [^]u8, source_len: c.ulong) -> c.int ---
}

EXR_COMPRESSION_NONE :: 0
EXR_COMPRESSION_RLE  :: 1
EXR_COMPRESSION_ZIPS :: 2
EXR_COMPRESSION_ZIP  :: 3

// Half-float (FP16) -> FP32. Mirrors the inverse of output/exr.odin's f32_to_f16
// and handles subnormals, inf, and NaN.
f16_to_f32 :: proc(h: u16) -> f32 {
	sign := u32(h >> 15) & 1
	exp := u32(h >> 10) & 0x1f
	mant := u32(h) & 0x3ff
	f: u32
	if exp == 0 {
		if mant == 0 {
			f = sign << 31
		} else {
			e := u32(127 - 15 + 1)
			m := mant
			for (m & 0x400) == 0 {
				m <<= 1
				e -= 1
			}
			m &= 0x3ff
			f = (sign << 31) | (e << 23) | (m << 13)
		}
	} else if exp == 0x1f {
		f = (sign << 31) | (0xff << 23) | (mant << 13)
	} else {
		f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)
	}
	return transmute(f32)f
}

@(private = "file")
Exr_Channel :: struct {
	name:       string,
	pixel_type: i32, // 0=uint, 1=half, 2=float
	bytes:      int, // per-sample size
	component:  int, // 0=R,1=G,2=B,3=A, -1=other (skipped)
}

@(private = "file")
Exr_Cursor :: struct {
	data: []u8,
	pos:  int,
}

@(private = "file")
read_u32le :: proc(cur: ^Exr_Cursor) -> (u32, bool) {
	if cur.pos + 4 > len(cur.data) { return 0, false }
	b := cur.data[cur.pos:]
	v := u32(b[0]) | u32(b[1]) << 8 | u32(b[2]) << 16 | u32(b[3]) << 24
	cur.pos += 4
	return v, true
}

@(private = "file")
read_i32le :: proc(cur: ^Exr_Cursor) -> (i32, bool) {
	v, ok := read_u32le(cur)
	return i32(v), ok
}

@(private = "file")
read_u64le :: proc(cur: ^Exr_Cursor) -> (u64, bool) {
	lo, ok1 := read_u32le(cur)
	hi, ok2 := read_u32le(cur)
	return u64(lo) | u64(hi) << 32, ok1 && ok2
}

// Reads a null-terminated string starting at the cursor.
@(private = "file")
read_cstr :: proc(cur: ^Exr_Cursor) -> (string, bool) {
	start := cur.pos
	for cur.pos < len(cur.data) {
		if cur.data[cur.pos] == 0 {
			s := string(cur.data[start:cur.pos])
			cur.pos += 1
			return s, true
		}
		cur.pos += 1
	}
	return "", false
}

// Loads an EXR file as linear RGB, top row first (row 0 = dataWindow y-min),
// matching stb's HDR layout so it drops straight into `load_environment`.
load_exr_rgb :: proc(path: string, allocator := context.allocator) -> (pixels: []f32, width: int, height: int, ok: bool) {
	raw, read_err := os.read_entire_file_from_path(path, allocator)
	if read_err != nil {
		fmt.eprintln("Failed to read EXR:", path, read_err)
		return nil, 0, 0, false
	}
	defer delete(raw, allocator)

	cur := Exr_Cursor{data = raw, pos = 0}

	magic, ok0 := read_u32le(&cur)
	if !ok0 || magic != 0x01312f76 {
		fmt.eprintln("Not an EXR file:", path)
		return nil, 0, 0, false
	}
	version, ok1 := read_u32le(&cur)
	if !ok1 {
		return nil, 0, 0, false
	}
	// Reject anything but a single-part scanline image.
	if version & 0x200 != 0 || version & 0x800 != 0 || version & 0x1000 != 0 {
		fmt.eprintln("Unsupported EXR (tiled/multipart/deep):", path)
		return nil, 0, 0, false
	}

	channels: [dynamic]Exr_Channel
	defer delete(channels)
	compression := EXR_COMPRESSION_NONE
	have_compression := false
	xmin, ymin, xmax, ymax: i32
	have_window := false

	// Header: a sequence of attributes terminated by an empty name.
	for {
		name, okn := read_cstr(&cur)
		if !okn {
			return nil, 0, 0, false
		}
		if name == "" {
			break // end of header
		}
		type, okt := read_cstr(&cur)
		if !okt {
			return nil, 0, 0, false
		}
		size, oks := read_i32le(&cur)
		if !oks || size < 0 || cur.pos + int(size) > len(cur.data) {
			return nil, 0, 0, false
		}
		attr_start := cur.pos
		switch name {
		case "channels":
			sub := Exr_Cursor{data = cur.data[:attr_start + int(size)], pos = attr_start}
			for {
				if sub.pos >= len(sub.data) { break }
				if sub.data[sub.pos] == 0 { break } // channel list terminator
				cname, okc := read_cstr(&sub)
				if !okc { break }
				ptype, _ := read_i32le(&sub)
				sub.pos += 4 // pLinear(1) + reserved(3)
				sub.pos += 8 // xSampling + ySampling
				comp := -1
				switch cname {
				case "R": comp = 0
				case "G": comp = 1
				case "B": comp = 2
				case "A": comp = 3
				}
				bytes := 2 if ptype == 1 else 4
				append(&channels, Exr_Channel{name = cname, pixel_type = ptype, bytes = bytes, component = comp})
			}
		case "compression":
			if size >= 1 {
				compression = int(cur.data[attr_start])
				have_compression = true
			}
		case "dataWindow":
			w := Exr_Cursor{data = cur.data, pos = attr_start}
			xmin, _ = read_i32le(&w)
			ymin, _ = read_i32le(&w)
			xmax, _ = read_i32le(&w)
			ymax, _ = read_i32le(&w)
			have_window = true
		}
		cur.pos = attr_start + int(size)
	}

	if !have_window || !have_compression || len(channels) == 0 {
		fmt.eprintln("Malformed EXR header:", path)
		return nil, 0, 0, false
	}
	if compression != EXR_COMPRESSION_NONE && compression != EXR_COMPRESSION_ZIP && compression != EXR_COMPRESSION_ZIPS {
		fmt.eprintfln("Unsupported EXR compression %d (only none/zip/zips): %s", compression, path)
		return nil, 0, 0, false
	}

	w := int(xmax - xmin + 1)
	h := int(ymax - ymin + 1)
	if w <= 0 || h <= 0 || w > 1 << 20 || h > 1 << 20 {
		return nil, 0, 0, false
	}

	// Storage order is the channel order in the header (EXR requires it sorted).
	bytes_per_line := 0
	for ch in channels {
		bytes_per_line += w * ch.bytes
	}
	lines_per_block := 16 if compression == EXR_COMPRESSION_ZIP else 1
	num_blocks := (h + lines_per_block - 1) / lines_per_block

	// Offset table: one u64 per block. We seek by these so line order and any
	// gaps are handled the same way the reference reader does.
	offsets := make([]u64, num_blocks, allocator)
	defer delete(offsets, allocator)
	for i in 0 ..< num_blocks {
		offsets[i], _ = read_u64le(&cur)
	}

	out := make([]f32, w * h * 3, allocator)
	scratch := make([]u8, lines_per_block * bytes_per_line, allocator)
	defer delete(scratch, allocator)

	for i in 0 ..< num_blocks {
		bc := Exr_Cursor{data = raw, pos = int(offsets[i])}
		if bc.pos < 0 || bc.pos >= len(raw) {
			delete(out, allocator)
			return nil, 0, 0, false
		}
		y0, oky := read_i32le(&bc)
		data_size, okd := read_i32le(&bc)
		if !oky || !okd || data_size < 0 || bc.pos + int(data_size) > len(raw) {
			delete(out, allocator)
			return nil, 0, 0, false
		}
		lines := lines_per_block
		if int(y0-ymin) + lines > h {
			lines = h - int(y0 - ymin)
		}
		uncompressed_size := lines * bytes_per_line
		src := raw[bc.pos:bc.pos + int(data_size)]

		block: []u8
		if compression == EXR_COMPRESSION_NONE || int(data_size) >= uncompressed_size {
			// Uncompressed, or a block the compressor stored raw because ZIP did
			// not shrink it. Plain channel data, no predictor/reorder.
			block = src
		} else {
			dst_len := c.ulong(uncompressed_size)
			rc := uncompress(raw_data(scratch), &dst_len, raw_data(src), c.ulong(len(src)))
			if rc != 0 || int(dst_len) != uncompressed_size {
				delete(out, allocator)
				fmt.eprintln("EXR inflate failed:", path)
				return nil, 0, 0, false
			}
			exr_unpredict(scratch[:uncompressed_size])
			exr_uninterleave(scratch[:uncompressed_size], allocator)
			block = scratch[:uncompressed_size]
		}

		// De-interleave: within the block, rows top-to-bottom, and for each row
		// the channels in header order, each holding `w` samples.
		off := 0
		for r in 0 ..< lines {
			row := int(y0-ymin) + r
			for ch in channels {
				if ch.component >= 0 && ch.component < 3 {
					dst_base := (row * w) * 3 + ch.component
					if ch.pixel_type == 1 {
						for x in 0 ..< w {
							b := block[off + x * 2:]
							hv := u16(b[0]) | u16(b[1]) << 8
							out[dst_base + x * 3] = f16_to_f32(hv)
						}
					} else {
						for x in 0 ..< w {
							b := block[off + x * 4:]
							bits := u32(b[0]) | u32(b[1]) << 8 | u32(b[2]) << 16 | u32(b[3]) << 24
							out[dst_base + x * 3] = transmute(f32)bits
						}
					}
				}
				off += w * ch.bytes
			}
		}
	}

	return out, w, h, true
}

// Reverse the OpenEXR ZIP byte predictor (prefix sum of byte deltas). Inverse of
// output/exr.odin's zip_predict.
@(private = "file")
exr_unpredict :: proc(data: []u8) {
	for i in 1 ..< len(data) {
		d := int(data[i - 1]) + int(data[i]) - 128
		data[i] = u8(d & 0xff)
	}
}

// Reverse the OpenEXR ZIP byte reorder: the first half holds the even byte
// positions and the second half the odd ones; interleave them back in place.
// Inverse of output/exr.odin's zip_reorder.
@(private = "file")
exr_uninterleave :: proc(data: []u8, allocator := context.allocator) {
	n := len(data)
	if n <= 1 { return }
	tmp := make([]u8, n, allocator)
	defer delete(tmp, allocator)
	half := (n + 1) / 2
	t1 := 0
	t2 := half
	s := 0
	for s < n {
		tmp[s] = data[t1]
		t1 += 1
		s += 1
		if s < n {
			tmp[s] = data[t2]
			t2 += 1
			s += 1
		}
	}
	copy(data, tmp)
}
