package lumbre_realtime

// Verifies that the rasterizer frames a scene exactly as the path tracer does.
//
// This is the whole reason `camera.odin` derives its matrices from
// `core.Camera` instead of from the orbit controls: every later step in this
// phase is checked by putting the two modes side by side, and that comparison
// says nothing at all if the two disagree about where the camera is pointing.
//
//   odin test realtime
//
// The check is a round trip. `core.get_ray(s, t)` is the ray the path tracer
// sends through image-plane coordinate (s, t), with (0, 0) the bottom-left
// corner. Take a point on that ray, project it with the matrices, and it must
// land at NDC (2s-1, 2t-1) — same pixel, from a completely different
// derivation.

import "core:math"
import "core:testing"

import m "core:math/linalg/glsl"

import lc "../core"

@(private = "file")
project :: proc(vp: matrix[4, 4]f32, p: [3]f32) -> (ndc: [3]f32, w: f32) {
	clip := vp * [4]f32{p.x, p.y, p.z, 1}
	return {clip.x / clip.w, clip.y / clip.w, clip.z / clip.w}, clip.w
}

@(test)
test_camera_framing_matches_path_tracer :: proc(t: ^testing.T) {
	// A deliberately awkward camera: off-axis, non-square aspect, and a focus
	// distance that is not 1, so a derivation that quietly assumes any of those
	// fails here.
	cam := lc.make_camera(
		lookfrom = {3.1, 2.4, -5.7},
		lookat = {0.3, 0.9, 0.2},
		vup = {0, 1, 0},
		vfov = 47.0,
		aspect_ratio = 16.0 / 9.0,
		aperture = 0.0, // depth of field would randomize the ray origin
		focus_dist = 6.25,
	)

	u := camera_uniforms(cam)
	// Aperture is zero above, so the ray origin never uses this.
	rng := lc.Rng{state = 1}

	// Corners and centre, plus a few interior points.
	samples := [][2]f64{
		{0, 0}, {1, 0}, {0, 1}, {1, 1},
		{0.5, 0.5}, {0.25, 0.75}, {0.83, 0.17},
	}

	for st in samples {
		s, tt := st[0], st[1]
		ray := lc.get_ray(cam, s, tt, &rng)

		// Anywhere along the ray projects to the same pixel; pick a point well
		// in front of the camera so the result is not near-degenerate.
		p := ray.origin + ray.direction * 0.7
		ndc, w := project(u.view_proj, {f32(p.x), f32(p.y), f32(p.z)})

		testing.expect(t, w > 0, "point must be in front of the camera")

		want_x := f32(2.0 * s - 1.0)
		want_y := f32(2.0 * tt - 1.0)
		testing.expectf(
			t,
			math.abs(ndc.x - want_x) < 1e-4 && math.abs(ndc.y - want_y) < 1e-4,
			"(s=%v t=%v): got NDC (%v, %v), want (%v, %v)",
			s, tt, ndc.x, ndc.y, want_x, want_y,
		)
	}
}

@(test)
test_camera_depth_range :: proc(t: ^testing.T) {
	cam := lc.make_camera(
		lookfrom = {0, 0, 5},
		lookat = {0, 0, 0},
		vup = {0, 1, 0},
		vfov = 60.0,
		aspect_ratio = 1.0,
		aperture = 0.0,
		focus_dist = 5.0,
	)

	f := camera_frame(cam)
	u := camera_uniforms(cam)

	// SDL_GPU clip space puts depth in [0, 1], not OpenGL's [-1, 1], and the
	// projection is reversed-Z: the near plane must map to 1 and the far plane
	// to 0. Getting this wrong reads as a working image with a broken depth
	// test, which is exactly the kind of bug that survives a visual check.
	near_p := f.eye + f.forward * (f.focus * NEAR_SCALE)
	far_p := f.eye + f.forward * (f.focus * FAR_SCALE)

	near_ndc, _ := project(u.view_proj, near_p)
	far_ndc, _ := project(u.view_proj, far_p)

	testing.expectf(t, math.abs(near_ndc.z - 1.0) < 1e-3, "near plane -> %v, want 1", near_ndc.z)
	testing.expectf(t, math.abs(far_ndc.z - 0.0) < 1e-3, "far plane -> %v, want 0", far_ndc.z)
}

@(test)
test_camera_frame_recovers_inputs :: proc(t: ^testing.T) {
	VFOV :: 37.5
	ASPECT :: 4.0 / 3.0
	FOCUS :: 2.75

	cam := lc.make_camera(
		lookfrom = {-2, 1.5, 4},
		lookat = {1, 0.25, -1},
		vup = {0, 1, 0},
		vfov = VFOV,
		aspect_ratio = ASPECT,
		aperture = 0.0,
		focus_dist = FOCUS,
	)

	f := camera_frame(cam)

	testing.expectf(
		t,
		math.abs(f.vfov - f32(lc.degrees_to_radians(VFOV))) < 1e-5,
		"vfov %v, want %v", f.vfov, f32(lc.degrees_to_radians(VFOV)),
	)
	testing.expectf(t, math.abs(f.aspect - f32(ASPECT)) < 1e-5, "aspect %v", f.aspect)
	testing.expectf(t, math.abs(f.focus - f32(FOCUS)) < 1e-4, "focus %v", f.focus)

	// The basis must stay orthonormal and right-handed, or normals and
	// backface culling go wrong later in ways that are hard to attribute.
	want_right := m.cross(f.forward, f.up)
	testing.expectf(
		t,
		m.length(f.right - want_right) < 1e-5,
		"right %v is not forward x up %v", f.right, want_right,
	)
}
