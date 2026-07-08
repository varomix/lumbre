package main

import m "core:math/linalg/glsl"

write_color :: proc(pixel_color: Color, samples_per_pixel: i32, pixels: []u8, pixel_index: int) {
	scale := 1.0 / f64(samples_per_pixel)
	r := m.sqrt(scale * pixel_color.x)
	g := m.sqrt(scale * pixel_color.y)
	b := m.sqrt(scale * pixel_color.z)

	r = m.clamp(r, 0.0, 0.999)
	g = m.clamp(g, 0.0, 0.999)
	b = m.clamp(b, 0.0, 0.999)

	ir := i32(256.0 * r)
	ig := i32(256.0 * g)
	ib := i32(256.0 * b)

	pixels[pixel_index + 0] = u8(ir)
	pixels[pixel_index + 1] = u8(ig)
	pixels[pixel_index + 2] = u8(ib)
}
