package lumbre_realtime

// Checks the cascade fitting.
//
//   odin test realtime
//
// This is the part of shadow mapping that is worth testing on the CPU: the
// projections either contain the frustum slice they were built for or they do
// not, and that is a numeric question. Acne and peter-panning are tuning; a
// cascade that does not cover its own slice is a bug, and on screen it looks
// like "shadows stop partway across the floor", which is easy to mistake for a
// range setting.

import "core:math"
import "core:testing"

import lc "../core"

@(private = "file")
test_camera :: proc() -> Camera_Frame {
	cam := lc.make_camera(
		lookfrom = {6, 4, 9},
		lookat = {0, 0, 0},
		vup = {0, 1, 0},
		vfov = 45.0,
		aspect_ratio = 16.0 / 9.0,
		aperture = 0.0,
		focus_dist = 11.0,
	)
	return camera_frame(cam)
}

// Projects a world point with a cascade and reports whether it lands inside
// that cascade's clip volume.
@(private = "file")
inside :: proc(c: Cascade, p: [3]f32) -> bool {
	clip := c.view_proj * [4]f32{p.x, p.y, p.z, 1}
	if clip.w <= 0 {
		return false
	}
	ndc := [3]f32{clip.x / clip.w, clip.y / clip.w, clip.z / clip.w}
	return ndc.x >= -1 && ndc.x <= 1 &&
	       ndc.y >= -1 && ndc.y <= 1 &&
	       ndc.z >= 0 && ndc.z <= 1
}

@(test)
test_cascades_cover_their_frustum_slices :: proc(t: ^testing.T) {
	f := test_camera()
	light_dir := normalize3({-0.4, -1, -0.3}) // a sun, travelling downward
	cascades := shadow_build_cascades(f, light_dir, {-10, -1, -10}, {10, 6, 10})

	testing.expect(t, cascades.enabled, "cascades must be enabled with a sun and real bounds")

	near := f.focus * NEAR_SCALE
	prev := near
	tan_half := math.tan(f.vfov * 0.5)

	for cascade, ci in cascades.slices {
		// Every corner of this slice must project inside this cascade. If it
		// does not, geometry there samples a shadow map that never saw it.
		for d in ([2]f32{prev, cascade.split_far}) {
			half_h := d * tan_half
			half_w := half_h * f.aspect
			centre := f.eye + f.forward * d
			for sy in ([2]f32{-1, 1}) {
				for sx in ([2]f32{-1, 1}) {
					corner := centre + f.right * (half_w * sx) + f.up * (half_h * sy)
					testing.expectf(
						t, inside(cascade, corner),
						"cascade %d does not contain a corner of its own slice (d=%v)", ci, d,
					)
				}
			}
		}
		prev = cascade.split_far
	}
}

@(test)
test_splits_increase_and_reach_the_scene :: proc(t: ^testing.T) {
	f := test_camera()
	bounds_min := [3]f32{-10, -1, -10}
	bounds_max := [3]f32{10, 6, 10}
	cascades := shadow_build_cascades(f, {0, -1, 0}, bounds_min, bounds_max)

	prev: f32 = 0
	for c, i in cascades.slices {
		testing.expectf(t, c.split_far > prev, "split %d (%v) must exceed the previous (%v)", i, c.split_far, prev)
		prev = c.split_far
	}

	// The last split must reach the far side of the scene, or shadows simply
	// stop partway. It must NOT reach the camera's far plane, which sits a
	// thousand focus distances out and would waste the whole depth range.
	furthest: f32 = 0
	for i in 0 ..< 8 {
		corner := [3]f32 {
			(i & 1) != 0 ? bounds_max.x : bounds_min.x,
			(i & 2) != 0 ? bounds_max.y : bounds_min.y,
			(i & 4) != 0 ? bounds_max.z : bounds_min.z,
		}
		d := corner - f.eye
		furthest = max(furthest, d.x * f.forward.x + d.y * f.forward.y + d.z * f.forward.z)
	}
	last := cascades.slices[CASCADE_COUNT - 1].split_far
	testing.expectf(t, math.abs(last - furthest) < 1e-3, "last split %v, want the scene's far extent %v", last, furthest)
	testing.expectf(t, last < f.focus * FAR_SCALE * 0.01, "last split %v must be far short of the camera far plane", last)
}

@(test)
test_cascades_are_progressively_larger :: proc(t: ^testing.T) {
	f := test_camera()
	cascades := shadow_build_cascades(f, normalize3({0, -1, -0.2}), {-10, -1, -10}, {10, 6, 10})

	// Each cascade covers more world per texel than the one before it — that is
	// the entire point of cascading. The x scale is 1/radius, but it is folded
	// into the light basis by the view multiply, so recover it as the LENGTH of
	// the first row rather than reading one element (which also carries the
	// light's orientation and is signed).
	prev_scale := max(f32)
	for c, i in cascades.slices {
		scale := math.sqrt(
			c.view_proj[0, 0] * c.view_proj[0, 0] +
			c.view_proj[0, 1] * c.view_proj[0, 1] +
			c.view_proj[0, 2] * c.view_proj[0, 2],
		)
		testing.expectf(
			t, scale < prev_scale,
			"cascade %d scale %v is not smaller than the previous %v", i, scale, prev_scale,
		)
		prev_scale = scale
	}
}

@(test)
test_no_sun_means_no_cascades :: proc(t: ^testing.T) {
	lights := []lc.Light {
		{kind = .Point, position = {0, 2, 0}, intensity = {1, 1, 1}},
		{kind = .Quad, position = {0, 3, 0}, u = {1, 0, 0}, v = {0, 0, 1}, intensity = {1, 1, 1}},
	}
	_, found := sun_light(lights)
	testing.expect(t, !found, "a scene with no distant light has no sun")

	with_sun := []lc.Light {
		{kind = .Point, intensity = {1, 1, 1}},
		{kind = .Distant, direction = {0, -2, 0}, intensity = {1, 1, 1}},
	}
	dir, ok := sun_light(with_sun)
	testing.expect(t, ok, "the distant light is the sun")
	// Returned normalized, since the cascade basis assumes a unit direction.
	length := math.sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z)
	testing.expectf(t, math.abs(length - 1) < 1e-6, "sun direction length %v, want 1", length)
}
