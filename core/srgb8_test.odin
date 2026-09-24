package lumbre_core

import "core:math/rand"
import "core:testing"

// srgb8_encode must match the slow reference byte for byte.
@(test)
test_srgb8_encode_matches_reference :: proc(t: ^testing.T) {
	mismatches := 0
	check :: proc(x: f32, mismatches: ^int) {
		if srgb8_encode(x) != srgb8_reference(x) {
			mismatches^ += 1
		}
	}
	// Dense near every level, where an off-by-one would show.
	for k in 0 ..< 256 {
		for probe in 0 ..< 4000 {
			check(f32(k) / 255.0 * f32(probe) / 3999.0, &mismatches)
		}
	}
	// A wide random sweep, including values outside [0, 1].
	rng := rand.create(11)
	context.random_generator = rand.default_random_generator(&rng)
	for _ in 0 ..< 2_000_000 {
		check(rand.float32_range(-0.5, 1.5), &mismatches)
	}
	// Every float just below and above 1.
	for bits in u32(0x3f7f0000) ..< u32(0x3f800100) {
		check(transmute(f32)bits, &mismatches)
	}
	testing.expectf(t, mismatches == 0, "%d values encode differently", mismatches)
	testing.expect_value(t, srgb8_encode(1), 255)
	testing.expect_value(t, srgb8_encode(0), 0)
}
