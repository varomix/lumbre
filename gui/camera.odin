package main

// Viewport camera.
//
// The renderer's `Camera` is a precomputed frustum basis — origin plus the
// three vectors a ray is built from — which is the right shape for tracing and
// the wrong shape for navigating. So the GUI keeps an orbit model here and
// rebuilds a `Camera` from it whenever anything changes.
//
// Every change routes through `orbit_camera_apply`, which is also the single
// place that invalidates the IPR, so there is no way to move the camera and
// forget to restart accumulation.

import "core:math"

import lc "../core"
import m "core:math/linalg/glsl"

Orbit_Camera :: struct {
	target:   lc.Vec3,
	distance: f64,
	yaw:      f64, // radians, around +Y
	pitch:    f64, // radians, clamped away from the poles
	vfov:     f64, // degrees
	aperture: f64,
}

// Just short of straight up/down: at exactly +/- pi/2 the up vector and the
// view direction become parallel and the basis collapses.
PITCH_LIMIT :: 1.5533 // ~89 degrees

// Direction from the target towards the camera.
@(private = "file")
orbit_direction :: proc(c: ^Orbit_Camera) -> lc.Vec3 {
	cp := math.cos(c.pitch)
	return lc.Vec3{cp * math.sin(c.yaw), math.sin(c.pitch), cp * math.cos(c.yaw)}
}

orbit_camera_eye :: proc(c: ^Orbit_Camera) -> lc.Vec3 {
	return c.target + orbit_direction(c) * c.distance
}

// Builds a renderer Camera from the orbit model. Deliberately returns one
// rather than writing into a scene: the UI thread must be able to compute a
// camera without touching scene memory the render worker is using.
//
// `aspect` comes from the render resolution, not the panel: they differ while a
// resize is in flight, and using the panel's would stretch the image.
orbit_camera_build :: proc(c: ^Orbit_Camera, aspect: f64) -> lc.Camera {
	a := aspect
	if a <= 0 {
		a = 1
	}
	return lc.make_camera(
		orbit_camera_eye(c),
		c.target,
		lc.Vec3{0, 1, 0},
		c.vfov,
		a,
		c.aperture,
		max(c.distance, 1.0e-4), // focus on the orbit target
	)
}

// Frames the whole scene. The distance is chosen so the bounding sphere fits
// the *vertical* field of view, then widened when the viewport is narrower than
// it is tall, which is the case that otherwise crops the subject.
orbit_camera_frame_bounds :: proc(c: ^Orbit_Camera, lo, hi: lc.Vec3, aspect: f64) {
	centre := (lo + hi) * 0.5
	extent := hi - lo
	radius := m.length(extent) * 0.5
	if radius <= 0 {
		radius = 1
	}

	c.target = centre

	half_v := math.to_radians(c.vfov) * 0.5
	dist := radius / math.sin(half_v)
	if aspect > 0 && aspect < 1 {
		dist /= aspect
	}
	c.distance = dist * 1.15 // a little margin so nothing touches the frame edge
}

orbit_camera_frame_scene :: proc(c: ^Orbit_Camera, scene: ^lc.Scene, aspect: f64) -> bool {
	lo, hi, ok := lc.scene_world_bounds(scene)
	if !ok {
		return false
	}
	orbit_camera_frame_bounds(c, lo, hi, aspect)
	return true
}

// Recovers an orbit model from a camera the importer built, so opening a USD
// scene with an authored camera starts from that view rather than snapping to a
// default framing.
//
// `Camera` stores an orthonormal basis (u, v) plus the frustum vectors, from
// which the rest follows: w = u x v points from the subject back towards the
// eye, and the focus distance is the component of the frustum along it.
orbit_camera_from_scene :: proc(c: ^Orbit_Camera, scene: ^lc.Scene, aspect: f64) {
	cam := scene.camera

	w := m.cross(cam.u, cam.v)
	if m.length(w) < 1.0e-9 {
		orbit_camera_frame_scene(c, scene, aspect)
		return
	}
	w = m.normalize(w)

	// origin - lower_left - horizontal/2 - vertical/2 == focus_dist * w
	to_plane := cam.origin - cam.lower_left_corner - cam.horizontal * 0.5 - cam.vertical * 0.5
	focus := m.length(to_plane)
	if focus <= 1.0e-6 {
		focus = 1
	}

	// vertical == focus_dist * 2 * tan(vfov/2) * v
	half_height := m.length(cam.vertical) * 0.5 / focus
	c.vfov = math.to_degrees(2.0 * math.atan(half_height))
	if c.vfov <= 1.0 || c.vfov >= 179.0 {
		c.vfov = 40.0
	}

	c.distance = focus
	c.target = cam.origin - w * focus
	c.pitch = clamp(math.asin(w.y), -PITCH_LIMIT, PITCH_LIMIT)
	c.yaw = math.atan2(w.x, w.z)
	c.aperture = cam.lens_radius * 2.0
}

// ── navigation ───────────────────────────────────────────────────────────────

orbit_camera_tumble :: proc(c: ^Orbit_Camera, dx, dy: f64) {
	TUMBLE_SPEED :: 0.008
	c.yaw -= dx * TUMBLE_SPEED
	c.pitch = clamp(c.pitch + dy * TUMBLE_SPEED, -PITCH_LIMIT, PITCH_LIMIT)
}

// Pans in the camera's own screen plane, scaled by distance so the subject
// tracks the cursor at any zoom level.
orbit_camera_pan :: proc(c: ^Orbit_Camera, dx, dy: f64, viewport_height: f64) {
	if viewport_height <= 0 {
		return
	}
	dir := orbit_direction(c)
	right := m.normalize(m.cross(lc.Vec3{0, 1, 0}, dir))
	up := m.cross(dir, right)

	// World units per pixel at the target plane.
	scale := 2.0 * c.distance * math.tan(math.to_radians(c.vfov) * 0.5) / viewport_height
	c.target = c.target + right * (dx * scale) + up * (dy * scale)
}

// Multiplicative so each notch covers the same proportion of the remaining
// distance — linear dollying crawls when far out and overshoots when close in.
orbit_camera_dolly :: proc(c: ^Orbit_Camera, amount: f64) {
	c.distance = clamp(c.distance * math.pow(0.9, amount), 1.0e-3, 1.0e9)
}
