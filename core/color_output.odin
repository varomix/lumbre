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
	// Display encode: no tonemap. Average the samples, clamp linear radiance to
	// [0, 1] (HDR highlights hard-clip to white, matching a plain sRGB display
	// transform), then apply the sRGB OETF. A Reinhard tonemap here desaturated
	// highlights and flattened contrast; the raw linear -> sRGB path keeps
	// saturation and matches reference renderers. Negatives (fireflies) floor to
	// 0: the OETF is undefined for them.
	lr := m.clamp(scale * pixel_color.x, 0.0, 1.0)
	lg := m.clamp(scale * pixel_color.y, 0.0, 1.0)
	lb := m.clamp(scale * pixel_color.z, 0.0, 1.0)
	r := linear_to_srgb(lr)
	g := linear_to_srgb(lg)
	b := linear_to_srgb(lb)

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

// The 8-bit sRGB byte for linear `x`: linear_to_srgb, rounded to the nearest
// of the 256 levels.
//
// Computed without the pow. A float's high bits say which narrow bucket it
// lies in; each bucket's byte is tabulated, and buckets are narrower than the
// gap between two levels, so one comparison against the next level's
// threshold settles the rest. The viewport encodes every pixel of every batch
// -- two million pows at 1080p, which took longer than rendering the batch.
//
// It rounds where the encode it replaced truncated: that darkened every pixel
// by half a level on average against the rasterizer's display, whose UNORM
// write rounds, and turned pure white into 254.
srgb8_encode :: #force_inline proc(x: f32) -> u8 #no_bounds_check {
	if !(x > SRGB8_MIN) { // also NaN and negatives
		return 0
	}
	if x >= 1 {
		return 255
	}
	b := srgb8_bucket_bytes[(transmute(u32)x - SRGB8_MIN_BITS) >> SRGB8_BUCKET_SHIFT]
	if x >= srgb8_thresholds[b] { // b < 255 here, since x < 1
		b += 1
	}
	return b
}

// The byte srgb8_encode must produce, computed the slow way.
srgb8_reference :: proc "contextless" (x: f32) -> u8 {
	c := clamp(f64(x), 0.0, 1.0)
	v := c <= 0.0031308 ? 12.92 * c : 1.055 * m.pow(c, 1.0 / 2.4) - 0.055
	return u8(clamp(v * 255.0 + 0.5, 0.0, 255.0))
}

// Everything at or below this encodes to 0 (it is under half a level).
@(private = "file")
SRGB8_MIN :: f32(1.0 / 65536.0)
@(private = "file")
SRGB8_MIN_BITS :: u32(0x37800000) // bits of SRGB8_MIN, 2^-16
// Buckets keep a float's exponent and top 12 mantissa bits: 1/4096 of an
// octave wide, far finer than a level anywhere in [SRGB8_MIN, 1).
@(private = "file")
SRGB8_BUCKET_SHIFT :: 11
@(private = "file")
SRGB8_BUCKETS :: (0x3f800000 - SRGB8_MIN_BITS) >> SRGB8_BUCKET_SHIFT

// srgb8_thresholds[k] is the smallest f32 whose byte is k + 1.
@(private = "file")
srgb8_thresholds: [256]f32
// The byte at the low edge of each bucket.
@(private = "file")
srgb8_bucket_bytes: [SRGB8_BUCKETS]u8

@(init, private = "file")
srgb8_build_tables :: proc "contextless" () {
	for k in 0 ..< 255 {
		// Search the bit patterns of non-negative floats up to 1, where
		// ordering by bits is ordering by value. 1.0 itself encodes to 255,
		// so every level is reached within the range.
		lo, hi := u32(0), transmute(u32)f32(1.0)
		for lo < hi {
			mid := lo + (hi - lo) / 2
			if int(srgb8_reference(transmute(f32)mid)) >= k + 1 {
				hi = mid
			} else {
				lo = mid + 1
			}
		}
		srgb8_thresholds[k] = transmute(f32)lo
	}
	srgb8_thresholds[255] = max(f32)
	for i in 0 ..< SRGB8_BUCKETS {
		srgb8_bucket_bytes[i] = srgb8_reference(transmute(f32)(SRGB8_MIN_BITS + u32(i) << SRGB8_BUCKET_SHIFT))
	}
}
