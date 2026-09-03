package lumbre_realtime

// Checks the light conversion: which kinds survive, and the geometry each one
// hands the shader.
//
//   odin test realtime
//
// Worth testing because the errors here are silent. A quad whose centre is
// computed from the wrong corner still lights the scene, just from slightly the
// wrong place; a dropped kind simply goes dark; a sphere given the wrong area
// is only "a bit dim". None of that announces itself in a screenshot the way a
// broken shader does.

import "core:math"
import "core:testing"

import lc "../core"

@(test)
test_unsupported_kinds_are_dropped :: proc(t: ^testing.T) {
	lights := []lc.Light {
		{kind = .Point, intensity = {1, 1, 1}},
		{kind = .Mesh, intensity = {5, 5, 5}},   // emissive geometry, not analytic
		{kind = .Dome, intensity = {2, 2, 2}},   // the environment; IBL's job
		{kind = .Distant, direction = {0, -1, 0}, intensity = {1, 1, 1}},
	}

	out := make([dynamic]Light_GPU)
	defer delete(out)
	lights_convert(lights, &out)

	// Mesh and Dome must not become analytic lights. Passing them through would
	// double-count the Cornell box's light panel: once as emissive geometry in
	// the G-buffer and again as a lamp.
	testing.expectf(t, len(out) == 2, "got %d lights, want 2 (Mesh and Dome dropped)", len(out))
	testing.expect(t, i32(out[0].params.x) == i32(Light_Kind_GPU.Point), "first must be the point light")
	testing.expect(t, i32(out[1].params.x) == i32(Light_Kind_GPU.Distant), "second must be the distant light")
}

@(test)
test_quad_becomes_centre_and_normal :: proc(t: ^testing.T) {
	// A 2x3 quad in the XZ plane, with `position` at the corner as core stores
	// it. Its centre is the corner plus half of each edge.
	light := lc.Light {
		kind = .Quad,
		position = {1, 5, 2},
		u = {2, 0, 0},
		v = {0, 0, 3},
		intensity = {4, 4, 4},
		area = 6,
	}

	out := make([dynamic]Light_GPU)
	defer delete(out)
	lights_convert({light}, &out)

	testing.expectf(t, len(out) == 1, "got %d lights", len(out))
	q := out[0]

	// Centre, not the corner. Using the corner shifts every quad light by half
	// its diagonal, which reads as a plausible but wrong lighting direction.
	want_centre := [3]f32{2, 5, 3.5}
	testing.expectf(
		t,
		math.abs(q.position.x - want_centre.x) < 1e-5 &&
		math.abs(q.position.y - want_centre.y) < 1e-5 &&
		math.abs(q.position.z - want_centre.z) < 1e-5,
		"centre (%v %v %v), want %v", q.position.x, q.position.y, q.position.z, want_centre,
	)

	// Normal is u x v, normalized: (2,0,0) x (0,0,3) = (0,-6,0) -> (0,-1,0).
	testing.expectf(
		t,
		math.abs(q.direction.x) < 1e-5 &&
		math.abs(q.direction.y + 1) < 1e-5 &&
		math.abs(q.direction.z) < 1e-5,
		"normal (%v %v %v), want (0 -1 0)", q.direction.x, q.direction.y, q.direction.z,
	)

	// The shader multiplies radiance by this, so it must be the real area.
	testing.expectf(t, math.abs(q.emission.w - 6) < 1e-5, "area %v, want 6", q.emission.w)
}

@(test)
test_area_shapes_carry_their_area :: proc(t: ^testing.T) {
	RADIUS :: 2.0
	HEIGHT :: 5.0

	lights := []lc.Light {
		{kind = .Sphere, position = {0, 0, 0}, radius = RADIUS, intensity = {1, 1, 1}},
		{kind = .Disc, position = {0, 0, 0}, direction = {0, -1, 0}, radius = RADIUS, intensity = {1, 1, 1}},
		{
			kind = .Cylinder, position = {0, 0, 0}, direction = {0, 1, 0},
			radius = RADIUS, height = HEIGHT, intensity = {1, 1, 1},
		},
	}

	out := make([dynamic]Light_GPU)
	defer delete(out)
	lights_convert(lights, &out)
	testing.expectf(t, len(out) == 3, "got %d lights", len(out))

	// Sphere and disc both project a disc of area pi*r^2.
	want_disc := f32(math.PI * RADIUS * RADIUS)
	testing.expectf(t, math.abs(out[0].emission.w - want_disc) < 1e-3, "sphere area %v, want %v", out[0].emission.w, want_disc)
	testing.expectf(t, math.abs(out[1].emission.w - want_disc) < 1e-3, "disc area %v, want %v", out[1].emission.w, want_disc)

	// Cylinder: the side only, 2*pi*r*h.
	want_side := f32(2.0 * math.PI * RADIUS * HEIGHT)
	testing.expectf(t, math.abs(out[2].emission.w - want_side) < 1e-3, "cylinder area %v, want %v", out[2].emission.w, want_side)

	// And its representative point is the axis midpoint, not the base.
	testing.expectf(
		t, math.abs(out[2].position.y - f32(HEIGHT * 0.5)) < 1e-5,
		"cylinder centre y %v, want %v", out[2].position.y, HEIGHT * 0.5,
	)
}

@(test)
test_spot_keeps_its_cone :: proc(t: ^testing.T) {
	light := lc.Light {
		kind = .Spot,
		position = {0, 3, 0},
		direction = {0, -1, 0},
		intensity = {10, 10, 10},
		cos_inner = 0.9,
		cos_outer = 0.7,
	}

	out := make([dynamic]Light_GPU)
	defer delete(out)
	lights_convert({light}, &out)

	s := out[0]
	// Inner and outer must not be swapped: the shader's smoothstep runs from
	// outer to inner, and reversing them inverts the cone's soft edge.
	testing.expectf(t, math.abs(s.params.y - 0.9) < 1e-6, "cos_inner %v, want 0.9", s.params.y)
	testing.expectf(t, math.abs(s.params.z - 0.7) < 1e-6, "cos_outer %v, want 0.7", s.params.z)
	testing.expect(t, s.params.y > s.params.z, "inner cosine must exceed outer")
}
