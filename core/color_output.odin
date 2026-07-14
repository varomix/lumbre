package lumbre_core

import m "core:math/linalg/glsl"

// IEC 61966-2-1 sRGB opto-electronic transfer function: linear -> sRGB.
// Inverse of `srgb_to_linear` in texture.odin. Used when encoding linear
// radiance into an 8-bit LDR buffer (PNG). EXR output stays linear.
linear_to_srgb :: proc(c: f64) -> f64 {
	if c <= 0.0031308 {
		return 12.92 * c
	}
	return 1.055 * m.pow(c, 1.0 / 2.4) - 0.055
}

write_color :: proc(pixel_color: Color, samples_per_pixel: i32, pixels: []u8, pixel_index: int) {
	scale := 1.0 / f64(samples_per_pixel)
	// Reinhard tonemap (matches the GPU path's `reinhard_tonemap`) so HDR
	// scenes -- emissive lights, HDRI domes -- compress highlights instead of
	// clamping them to flat white. Maps [0, inf) -> [0, 1). Negatives (from
	// fireflies) are floored first: the OETF is undefined for them.
	lr := max(scale * pixel_color.x, 0.0)
	lg := max(scale * pixel_color.y, 0.0)
	lb := max(scale * pixel_color.z, 0.0)
	r := linear_to_srgb(lr / (1.0 + lr))
	g := linear_to_srgb(lg / (1.0 + lg))
	b := linear_to_srgb(lb / (1.0 + lb))

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
