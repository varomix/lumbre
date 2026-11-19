package lumbre_realtime

import "core:os"
import "core:slice"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

@(test)
test_halton_points :: proc(t: ^testing.T) {
	testing.expect_value(t, halton(1, 2), 0.5)
	testing.expect_value(t, halton(2, 2), 0.25)
	testing.expect_value(t, halton(3, 2), 0.75)
	testing.expect_value(t, halton(1, 3), f32(1) / 3)
	testing.expect_value(t, halton(2, 3), f32(2) / 3)
	testing.expect_value(t, halton(3, 3), f32(1) / 9)
	// The jitter cycle covers the pixel without repeating a point.
	seen: [TAA_SAMPLES][2]f32
	for i in 0 ..< TAA_SAMPLES {
		p := [2]f32{halton(u32(i + 1), 2), halton(u32(i + 1), 3)}
		testing.expect(t, p.x > 0 && p.x < 1 && p.y > 0 && p.y < 1)
		for q in seen[:i] { testing.expect(t, q != p) }
		seen[i] = p
	}
}

// The jitter moves the image, never the depth: a jittered projection maps a
// point to the same depth and shifts x and y by exactly the jitter.
@(test)
test_jitter_shifts_ndc_only :: proc(t: ^testing.T) {
	cam := lc.make_camera({0, 0, 5}, {0, 0, 0}, {0, 1, 0}, 40, 1.5, 0, 5)
	f := camera_frame(cam)
	p := [4]f32{0.3, -0.2, 1, 1}
	jitter := [2]f32{0.01, -0.02}
	a := camera_view_proj(f) * p
	b := camera_view_proj(f, jitter) * p
	near :: proc(x, y: f32) -> bool { return abs(x - y) < 1e-6 }
	testing.expect(t, near(b.z / b.w, a.z / a.w))
	testing.expect(t, near(b.x / b.w, a.x / a.w + jitter.x))
	testing.expect(t, near(b.y / b.w, a.y / a.w + jitter.y))
}

when #config(GPU_TESTS, false) {

@(private = "file")
BLACK := [3]f32{0, 0, 0}

// A grey emissive triangle on black, its long edge diagonal across the frame.
@(private = "file")
make_edge_scene :: proc() -> lc.Scene {
	tris := make([]lc.Triangle, 1)
	n := lc.Vec3{0, 0, 1}
	tris[0] = lc.Triangle{v0 = {-3, -2, 0}, v1 = {3, -2, 0}, v2 = {-3, 2.3, 0}, n0 = n, n1 = n, n2 = n}
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 1)
	materials[0] = lc.Material{kind = .Emissive, emission = {0.5, 0.5, 0.5}, emission_strength = 1}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	// Black, rather than the default sky a scene without one gets.
	scene.environment = {width = 1, height = 1, has_data = true, pixels = BLACK[:], intensity = 1}
	return scene
}

@(private = "file")
open_renderer :: proc(t: ^testing.T) -> (^sdl.GPUDevice, Renderer) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	r, ok := renderer_create(gpu)
	if !ok { sdl.DestroyGPUDevice(gpu); testing.fail_now(t, "renderer creation") }
	return gpu, r
}

// Pixels strictly between the black background and the triangle's grey:
// partial coverage, which an aliased edge never shows.
@(private = "file")
count_partial :: proc(pixels: []u8) -> (n: int) {
	for i := 0; i < len(pixels); i += 4 {
		v := pixels[i]
		if v > 8 && v < 180 { n += 1 }
	}
	return
}

W :: 64
H :: 48

@(test)
test_gpu_taa_antialiases_a_still_deterministically :: proc(t: ^testing.T) {
	gpu, r := open_renderer(t)
	defer sdl.DestroyGPUDevice(gpu)
	defer renderer_destroy(&r)
	scene := make_edge_scene()
	defer destroy_test_scene(&scene)
	cam := lc.make_camera({0, 0, 6}, {0, 0, 0}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 6)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	r.settings.ao = false
	r.settings.ssr = false

	// No antialiasing at all: every pixel is either covered or not.
	r.settings.taa = false
	r.settings.fxaa = false
	aliased, ok := renderer_read_color(&r, cam, W, H, .Shaded)
	if !ok { testing.fail_now(t, "aliased render") }
	defer delete(aliased)
	testing.expect_value(t, count_partial(aliased), 0)

	r.settings.taa = true
	first, ok_a := renderer_read_color(&r, cam, W, H, .Shaded)
	if !ok_a { testing.fail_now(t, "TAA render") }
	defer delete(first)
	// The edge crosses about 70 pixels; most of them must now be partial.
	testing.expectf(t, count_partial(first) > 40, "%d partially covered pixels", count_partial(first))
	// Away from the edge the converged image is the aliased one: TAA must not
	// soften or brighten flat regions.
	interior := ((H / 2) * W + W / 4) * 4
	// Within one step: the history is half-float, and blending rounds.
	testing.expect(t, abs(i32(first[interior]) - i32(aliased[interior])) <= 1)
	testing.expect(t, abs(i32(first[4]) - i32(aliased[4])) <= 1)

	// A still depends on nothing drawn before it: a moved camera in between
	// leaves the next still byte-identical.
	other := lc.make_camera({1, 0.5, 6}, {0, 0, 0}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 6)
	_ = renderer_render(&r, other, W, H, .Shaded)
	second, ok_b := renderer_read_color(&r, cam, W, H, .Shaded)
	if !ok_b { testing.fail_now(t, "second TAA render") }
	defer delete(second)
	testing.expect(t, slice.equal(first, second), "TAA stills differ between runs")
}

// Labels come from their own unjittered pass, so TAA changes no label byte.
@(test)
test_gpu_taa_leaves_labels_alone :: proc(t: ^testing.T) {
	gpu, r := open_renderer(t)
	defer sdl.DestroyGPUDevice(gpu)
	defer renderer_destroy(&r)
	scene := make_edge_scene()
	defer destroy_test_scene(&scene)
	cam := lc.make_camera({0, 0, 6}, {0, 0, 0}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 6)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))

	capture :: proc(t: ^testing.T, r: ^Renderer, cam: lc.Camera) -> Label_Frame {
		c: Capture
		defer capture_destroy(r.gpu, &c)
		if !capture_begin(r, &c, cam, W, H, .Shaded, true) { testing.fail_now(t, "capture") }
		pixels, frame, ok := capture_finish(r.gpu, &c)
		if !ok { testing.fail_now(t, "capture finish") }
		delete(pixels)
		return frame
	}
	r.settings.taa = false
	plain := capture(t, &r, cam)
	defer label_frame_destroy(&plain)
	r.settings.taa = true
	temporal := capture(t, &r, cam)
	defer label_frame_destroy(&temporal)
	testing.expect(t, slice.equal(plain.instance, temporal.instance))
	testing.expect(t, slice.equal(plain.semantic, temporal.semantic))
	testing.expect(t, slice.equal(plain.depth, temporal.depth))
}

// After the camera moves, reprojected history must still line up: one frame
// after a pan is much closer to the converged still than a frame with no
// history.
// A history fetched from the wrong place would be clipped away -- or worse,
// smear -- and leave the edge aliased.
@(test)
test_gpu_taa_reprojects_after_a_pan :: proc(t: ^testing.T) {
	gpu, r := open_renderer(t)
	defer sdl.DestroyGPUDevice(gpu)
	defer renderer_destroy(&r)
	scene := make_edge_scene()
	defer destroy_test_scene(&scene)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	r.settings.ao = false
	r.settings.ssr = false
	a := lc.make_camera({0, 0, 6}, {0, 0, 0}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 6)
	// About five pixels right and three up.
	b := lc.make_camera({0.4, 0.25, 6}, {0.4, 0.25, 0}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 6)

	read :: proc(t: ^testing.T, r: ^Renderer) -> []u8 {
		cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
		transfer, ok := download_begin(r.gpu, cmd, r.color, W, H, 4)
		if !ok { testing.fail_now(t, "download") }
		defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		defer sdl.ReleaseGPUFence(r.gpu, fence)
		testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
		pixels := make([]u8, W * H * 4)
		testing.expect(t, map_copy(r.gpu, transfer, pixels))
		return pixels
	}
	difference :: proc(x, y: []u8) -> (sum: int) {
		for i := 0; i < len(x); i += 4 { sum += abs(int(x[i]) - int(y[i])) }
		return
	}

	still, ok := renderer_read_color(&r, b, W, H, .Shaded)
	if !ok { testing.fail_now(t, "still") }
	defer delete(still)

	// Converge at `a`, then one frame at `b`.
	if renderer_render_still(&r, a, W, H, .Shaded) == nil { testing.fail_now(t, "converge") }
	if renderer_render(&r, b, W, H, .Shaded) == nil { testing.fail_now(t, "pan") }
	panned := read(t, &r)
	defer delete(panned)

	// One frame at `b` with no history at all.
	renderer_reset_history(&r)
	if renderer_render(&r, b, W, H, .Shaded) == nil { testing.fail_now(t, "fresh") }
	fresh := read(t, &r)
	defer delete(fresh)

	// Measured on Metal: about 5x closer reprojecting correctly, 2.7x with the
	// history's y flipped, and worse than no history without reprojection.
	testing.expectf(t, f32(difference(panned, still)) * 3.5 < f32(difference(fresh, still)),
		"panned frame is %d from the still, a frame without history %d",
		difference(panned, still), difference(fresh, still))
}

@(test)
test_gpu_taa_history_and_profiling :: proc(t: ^testing.T) {
	gpu, r := open_renderer(t)
	defer sdl.DestroyGPUDevice(gpu)
	defer renderer_destroy(&r)
	scene := make_edge_scene()
	defer destroy_test_scene(&scene)
	cam := lc.make_camera({0, 0, 6}, {0, 0, 0}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 6)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))

	for _ in 0 ..< 3 { testing.expect(t, renderer_render(&r, cam, W, H, .Shaded) != nil) }
	testing.expect(t, r.taa_valid && renderer_taa_active(&r))
	testing.expect_value(t, r.taa_frame, 3)
	testing.expect_value(t, r.taa_still, 2)

	// A resize discards history; so does showing another view.
	testing.expect(t, renderer_render(&r, cam, W + 8, H, .Shaded) != nil)
	testing.expect_value(t, r.taa_frame, 1)
	testing.expect(t, renderer_render(&r, cam, W + 8, H, .Albedo) != nil)
	testing.expect(t, !r.taa_valid && !renderer_taa_active(&r))
	testing.expect_value(t, r.taa_jitter, [2]f32{})

	// Profiling: one frame, every pass that ran charged, none that did not.
	r.profile = true
	testing.expect(t, renderer_render(&r, cam, W, H, .Shaded) != nil)
	testing.expect_value(t, r.profiler.frames, 1)
	p := r.profiler.last_frame
	testing.expect(t, p[.Debug] == 0 && p[.FXAA] == 0 && p[.Labels] == 0)
	sum: f32
	for ms in p { testing.expect(t, ms >= 0); sum += ms }
	testing.expect(t, sum > 0)
}

}

when #config(GPU_TESTS, false) {

// An object that moves under a still camera: the history follows it by its
// velocity, where depth reprojection would read it from where it was.
@(test)
test_gpu_taa_follows_moving_objects :: proc(t: ^testing.T) {
	gpu, r := open_renderer(t)
	defer sdl.DestroyGPUDevice(gpu)
	defer renderer_destroy(&r)
	scene := make_edge_scene()
	defer destroy_test_scene(&scene)
	r.settings.ao = false
	r.settings.ssr = false
	cam := lc.make_camera({0, 0, 6}, {0, 0, 0}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 6)
	at_a := scene.nodes[0].local_transform
	at_b := at_a
	at_b[0, 3] += 0.4 // about five pixels right
	at_b[1, 3] += 0.25 // and three up

	read :: proc(t: ^testing.T, r: ^Renderer) -> []u8 {
		cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
		transfer, ok := download_begin(r.gpu, cmd, r.color, W, H, 4)
		if !ok { testing.fail_now(t, "download") }
		defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		defer sdl.ReleaseGPUFence(r.gpu, fence)
		testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
		pixels := make([]u8, W * H * 4)
		testing.expect(t, map_copy(r.gpu, transfer, pixels))
		return pixels
	}
	difference :: proc(x, y: []u8) -> (sum: int) {
		for i := 0; i < len(x); i += 4 { sum += abs(int(x[i]) - int(y[i])) }
		return
	}
	// Converge at A, move to B, draw one frame. `by_velocity` false forgets
	// where instances were, which leaves depth reprojection alone.
	moved_frame :: proc(t: ^testing.T, r: ^Renderer, scene: ^lc.Scene, cam: lc.Camera, a, b: matrix[4, 4]f32, key: u64, by_velocity: bool) -> []u8 {
		scene.nodes[0].local_transform = a
		testing.expect(t, renderer_set_scene(r, scene, key))
		if renderer_render_still(r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "converge") }
		scene.nodes[0].local_transform = b
		testing.expect(t, renderer_set_scene(r, scene, key + 1))
		if !by_velocity { r.taa_worlds_valid = false }
		if renderer_render(r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "moved") }
		testing.expect_value(t, r.taa_object_motion, by_velocity)
		return read(t, r)
	}

	scene.nodes[0].local_transform = at_b
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	still, ok := renderer_read_color(&r, cam, W, H, .Shaded)
	if !ok { testing.fail_now(t, "still") }
	defer delete(still)

	followed := moved_frame(t, &r, &scene, cam, at_a, at_b, 10, true)
	defer delete(followed)
	// The frames after, settling on the new pose, must not reproject again.
	if renderer_render(&r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "settle") }
	testing.expect(t, !r.taa_object_motion, "settling frames reproject by a stale velocity")

	by_depth := moved_frame(t, &r, &scene, cam, at_a, at_b, 20, false)
	defer delete(by_depth)

	// Measured on Metal: 5x closer by velocity.
	testing.expectf(t, f32(difference(followed, still)) * 3.5 < f32(difference(by_depth, still)),
		"by velocity %d from the still, by depth %d", difference(followed, still), difference(by_depth, still))
}

}
