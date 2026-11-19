package lumbre_realtime

// The motion pass against projected displacement.
//
//   odin test realtime -define:GPU_TESTS=true -define:ODIN_TEST_THREADS=1
//
// One unit quad drawn by two nodes, so both are one instanced draw and the
// previous transform has to be found by instance id, not by draw.

import "core:os"
import "core:strings"
import "core:testing"

import m "core:math/linalg/glsl"

import lc "../core"
import sdl "vendor:sdl3"

when #config(GPU_TESTS, false) {

MOTION_W :: 128
MOTION_H :: 64

@(private = "file")
quad_scene :: proc() -> lc.Scene {
	corner :: proc(x, y: f64) -> lc.Vec3 { return {x, y, 0} }
	tris := make([]lc.Triangle, 2)
	n := lc.Vec3{0, 0, 1}
	tris[0] = {v0 = corner(-1, -1), v1 = corner(1, -1), v2 = corner(1, 1), n0 = n, n1 = n, n2 = n}
	tris[1] = {v0 = corner(-1, -1), v1 = corner(1, 1), v2 = corner(-1, 1), n0 = n, n1 = n, n2 = n}
	meshes := make([]lc.Mesh, 1)
	meshes[0] = {triangles = tris, transform = m.mat4(1)}
	nodes := make([]lc.SceneNode, 2)
	nodes[0] = lc.make_node(m.mat4Translate({-2, 0, 0}), 0, -1, -1)
	nodes[1] = lc.make_node(m.mat4Translate({2, 0, 0}), 0, -1, -1)
	materials := make([]lc.Material, 1)
	materials[0] = {kind = .Principled, albedo = {0.5, 0.5, 0.5}, roughness = 0.5}
	return lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
}

@(private = "file")
free_quad_scene :: proc(s: ^lc.Scene) {
	delete(s.meshes[0].triangles)
	delete(s.meshes)
	delete(s.nodes)
	delete(s.materials)
}

@(private = "file")
motion_camera :: proc(eye: lc.Vec3) -> lc.Camera {
	return lc.make_camera(eye, eye - {0, 0, 10}, {0, 1, 0}, 40, f64(MOTION_W) / f64(MOTION_H), 0, 10)
}

// Where a world point lands, in pixels (x right, y down), through the same
// unjittered projection the pass uses.
@(private = "file")
project_px :: proc(cam: lc.Camera, p: [3]f32) -> [2]f32 {
	clip := camera_view_proj(camera_frame(cam)) * [4]f32{p.x, p.y, p.z, 1}
	ndc := clip.xy / clip.w
	return {(ndc.x * 0.5 + 0.5) * MOTION_W, (0.5 - ndc.y * 0.5) * MOTION_H}
}

@(private = "file")
velocity_at :: proc(v: [][2]f32, px: [2]f32) -> [2]f32 {
	x := clamp(int(px.x), 0, MOTION_W - 1)
	y := clamp(int(px.y), 0, MOTION_H - 1)
	return v[y * MOTION_W + x]
}

@(private = "file")
near2 :: proc(a, b: [2]f32) -> bool {
	return abs(a.x - b.x) < 1e-3 && abs(a.y - b.y) < 1e-3
}

@(test)
test_gpu_motion_vectors :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	scene := quad_scene()
	defer free_quad_scene(&scene)
	cam := motion_camera({0, 0, 10})
	left, right := [3]f32{-2, 0, 0}, [3]f32{2, 0, 0}

	// Nothing moved: every pixel, covered or not, is exactly zero.
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	still, still_ok := motion_read(&r, cam, cam, MOTION_W, MOTION_H)
	if !still_ok { testing.fail_now(t, "motion readback") }
	worst: [2]f32
	nonzero := 0
	for v in still { if v != {0, 0} { nonzero += 1; if abs(v.x) + abs(v.y) > abs(worst.x) + abs(worst.y) { worst = v } } }
	testing.expectf(t, nonzero == 0, "%d nonzero, worst %v", nonzero, worst)
	delete(still)

	// The left node came from 0.5 further left and 0.25 lower; the right one
	// is static. Same draw, so only the id tells them apart.
	scene.nodes[0].prev_local_transform = m.mat4Translate({-2.5, -0.25, 0})
	scene.nodes[0].has_prev_local = true
	testing.expect(t, renderer_set_scene(&r, &scene, 2))
	testing.expect_value(t, r.geometry_uploads, 1) // instances only
	moved, moved_ok := motion_read(&r, cam, cam, MOTION_W, MOTION_H)
	if !moved_ok { testing.fail_now(t, "motion readback") }
	defer delete(moved)
	want := project_px(cam, left) - project_px(cam, {-2.5, -0.25, 0})
	got := velocity_at(moved, project_px(cam, left))
	testing.expectf(t, near2(got, want), "moving quad: got %v, want %v", got, want)
	// Right and up in the world is right and up the image: +x, -y.
	testing.expect(t, got.x > 1 && got.y < -0.1)
	testing.expect_value(t, velocity_at(moved, project_px(cam, right)), [2]f32{0, 0})
	testing.expect_value(t, velocity_at(moved, {MOTION_W / 2, 2}), [2]f32{0, 0}) // background

	// Camera motion alone: the static quad slides the other way on screen.
	prev_cam := motion_camera({0.5, 0, 10})
	panned, panned_ok := motion_read(&r, cam, prev_cam, MOTION_W, MOTION_H)
	if !panned_ok { testing.fail_now(t, "motion readback") }
	defer delete(panned)
	want = project_px(cam, right) - project_px(prev_cam, right)
	got = velocity_at(panned, project_px(cam, right))
	testing.expectf(t, near2(got, want), "panned camera: got %v, want %v", got, want)
	testing.expect(t, got.x > 1)
}

}
