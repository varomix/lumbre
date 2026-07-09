package main

import "core:c"
import "core:fmt"
import m "core:math"
import "core:os"
import "core:strings"
import stbi "vendor:stb/image"

// IEC 61966-2-1 sRGB electro-optical transfer function.
srgb_to_linear :: proc(c: f64) -> f64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return m.pow((c + 0.055) / 1.055, 2.4)
}

// Decode table for the 256 possible 8-bit sRGB channel values. Built once so
// bilinear sampling never pays for a `pow` per texel fetch.
@(private = "file")
srgb_lut: [256]f64
@(private = "file")
srgb_lut_ready: bool

@(private = "file")
init_srgb_lut :: proc() {
	if srgb_lut_ready {
		return
	}
	for i in 0 ..< 256 {
		srgb_lut[i] = srgb_to_linear(f64(i) / 255.0)
	}
	srgb_lut_ready = true
}

// Load an image file (PNG, JPEG, etc.) and return an RGBA8 TextureMap.
// `base_dir` is the directory used to resolve relative paths. Pass ""
// to use the current working directory. `srgb` marks the RGB channels as
// sRGB-encoded so sampling decodes them to linear.
load_texture :: proc(path: string, base_dir: string, srgb := true, allocator := context.allocator) -> (TextureMap, bool) {
	if len(path) == 0 {
		return TextureMap{}, false
	}

	// Resolve the full path
	resolved := path
	if base_dir != "" && !is_absolute_path(path) {
		resolved = strings.concatenate({base_dir, path}, allocator)
		defer if resolved != path { delete(resolved, allocator) }
	}

	width, height, channels: c.int
	stbi.set_flip_vertically_on_load(c.int(1))
	pixels := stbi.load(strings.clone_to_cstring(resolved, allocator), &width, &height, &channels, 4)
	if pixels == nil {
		fmt.eprintln("Failed to load texture:", resolved)
		return TextureMap{}, false
	}
	defer stbi.image_free(pixels)

	tex := TextureMap{
		width    = i32(width),
		height   = i32(height),
		pixels   = make([]u8, int(width) * int(height) * 4, allocator),
		has_data = true,
		srgb     = srgb,
	}
	copy(tex.pixels, ([^]u8)(pixels)[:int(width) * int(height) * 4])
	return tex, true
}

// Load an RGBA8 texture from an in-memory byte buffer (PNG, JPEG, etc.).
// Used by the glTF loader to read buffer-embedded images from a .glb.
// `srgb` marks the RGB channels as sRGB-encoded (true for color maps).
load_texture_from_memory :: proc(data: []u8, srgb := true, allocator := context.allocator) -> (TextureMap, bool) {
	if len(data) == 0 {
		return TextureMap{}, false
	}
	width, height, channels: c.int
	stbi.set_flip_vertically_on_load(c.int(1))
	pixels := stbi.load_from_memory(raw_data(data), c.int(len(data)), &width, &height, &channels, 4)
	if pixels == nil {
		return TextureMap{}, false
	}
	defer stbi.image_free(pixels)

	tex := TextureMap{
		width    = i32(width),
		height   = i32(height),
		pixels   = make([]u8, int(width) * int(height) * 4, allocator),
		has_data = true,
		srgb     = srgb,
	}
	copy(tex.pixels, ([^]u8)(pixels)[:int(width) * int(height) * 4])
	return tex, true
}

is_absolute_path :: proc(path: string) -> bool {
	if len(path) == 0 {
		return false
	}
	if path[0] == '/' {
		return true
	}
	// Windows-style "C:\" or "C:/"
	if len(path) >= 3 && path[0] >= 'A' && path[0] <= 'Z' && path[1] == ':' && (path[2] == '/' || path[2] == '\\') {
		return true
	}
	return false
}

// Generate an 8x8 procedural checkerboard texture, used for testing
// the texture pipeline without external image files. The texture has
// 8 black/white squares.
make_checker_texture :: proc(width, height, cells: i32, allocator := context.allocator) -> TextureMap {
	tex := TextureMap{
		width    = width,
		height   = height,
		pixels   = make([]u8, int(width) * int(height) * 4, allocator),
		has_data = true,
	}
	for y in 0 ..< height {
		for x in 0 ..< width {
			cx := x * cells / width
			cy := y * cells / height
			cell := (cx + cy) & 1
			r: u8 = 255 if cell == 0 else 32
			g: u8 = 200 if cell == 0 else 32
			b: u8 = 100 if cell == 0 else 200
			off := (int(y) * int(width) + int(x)) * 4
			tex.pixels[off + 0] = r
			tex.pixels[off + 1] = g
			tex.pixels[off + 2] = b
			tex.pixels[off + 3] = 255
		}
	}
	return tex
}

// Sample an RGBA8 texture with bilinear filtering and wrap-mode repeat.
// Returns a color in [0, 1]^3 (alpha is included in `out`).
sample_texture_bilinear :: proc(tex: ^TextureMap, u, v: f64) -> (out: Color, alpha: f64) {
	if !tex.has_data || tex.width <= 0 || tex.height <= 0 {
		return Color{1.0, 1.0, 1.0}, 1.0
	}
	init_srgb_lut()
	w := f64(tex.width)
	h := f64(tex.height)
	// Wrap to [0, 1)
	uu := u - m_floor(u)
	vv := v - m_floor(v)
	// Convert to texel space
	px := uu * w - 0.5
	py := vv * h - 0.5
	x0 := i32(m_floor(px))
	y0 := i32(m_floor(py))
	x1 := x0 + 1
	y1 := y0 + 1
	fx := px - f64(x0)
	fy := py - f64(y0)

	wrap_x :: proc(idx, w: i32) -> i32 {
		out := idx % w
		if out < 0 { out += w }
		return out
	}
	wrap_y :: proc(idx, h: i32) -> i32 {
		out := idx % h
		if out < 0 { out += h }
		return out
	}

	xi0 := wrap_x(x0, tex.width)
	xi1 := wrap_x(x1, tex.width)
	yi0 := wrap_y(y0, tex.height)
	yi1 := wrap_y(y1, tex.height)

	// Decode to linear *before* filtering: averaging sRGB-encoded values
	// darkens edges. Alpha is always linear and is never decoded.
	pixel_at :: proc(tex: ^TextureMap, x, y: i32) -> [4]f64 {
		off := (int(y) * int(tex.width) + int(x)) * 4
		if tex.srgb {
			return [4]f64 {
				srgb_lut[tex.pixels[off + 0]],
				srgb_lut[tex.pixels[off + 1]],
				srgb_lut[tex.pixels[off + 2]],
				f64(tex.pixels[off + 3]) / 255.0,
			}
		}
		return [4]f64 {
			f64(tex.pixels[off + 0]) / 255.0,
			f64(tex.pixels[off + 1]) / 255.0,
			f64(tex.pixels[off + 2]) / 255.0,
			f64(tex.pixels[off + 3]) / 255.0,
		}
	}

	p00 := pixel_at(tex, xi0, yi0)
	p10 := pixel_at(tex, xi1, yi0)
	p01 := pixel_at(tex, xi0, yi1)
	p11 := pixel_at(tex, xi1, yi1)

	a := (1.0 - fx) * (1.0 - fy)
	b := fx * (1.0 - fy)
	c := (1.0 - fx) * fy
	d := fx * fy

	r := p00[0] * a + p10[0] * b + p01[0] * c + p11[0] * d
	g := p00[1] * a + p10[1] * b + p01[1] * c + p11[1] * d
	bb := p00[2] * a + p10[2] * b + p01[2] * c + p11[2] * d
	a8 := p00[3] * a + p10[3] * b + p01[3] * c + p11[3] * d
	return Color{r, g, bb}, a8
}

m_floor :: proc(x: f64) -> f64 {
	if x < 0.0 {
		return -m_ceil(-x)
	}
	return f64(i64(x))
}

m_ceil :: proc(x: f64) -> f64 {
	xi := i64(x)
	if f64(xi) == x {
		return f64(xi)
	}
	if x > 0 {
		return f64(xi + 1)
	}
	return f64(xi)
}
