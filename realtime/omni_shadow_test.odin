package lumbre_realtime

import "core:math"
import "core:os"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

// Point and sphere lights take omni slots in scene order up to the limit,
// independently of the spot lights' layers.
@(test)
test_omni_shadow_slots :: proc(t: ^testing.T) {
	lights := make([dynamic]lc.Light, context.temp_allocator)
	for i in 0 ..< MAX_OMNI_SHADOWS + 1 {
		append(&lights, lc.Light{kind = .Point, position = {f64(i), 2, 0}, intensity = {1, 1, 1}})
	}
	append(&lights, lc.Light{kind = .Spot, position = {0, 3, 0}, direction = {0, -1, 0}, intensity = {1, 1, 1}, cos_inner = 0.9, cos_outer = 0.8})
	out: [dynamic]Light_GPU
	defer delete(out)
	lights_convert(lights[:], &out)
	for i in 0 ..< MAX_OMNI_SHADOWS { testing.expect_value(t, out[i].params.w, f32(i)) }
	testing.expect_value(t, out[MAX_OMNI_SHADOWS].params.w, -1)
	testing.expect_value(t, out[MAX_OMNI_SHADOWS + 1].params.w, 0) // the spot's own layer
	testing.expect_value(t, omni_shadows_build(out[:], {-5, 0, -5}, {5, 4, 5}).count, MAX_OMNI_SHADOWS)
}

// Each face sees the points along its own axis at the centre of its map, and
// together the six cover every direction: whichever face the shader would
// pick for a point projects it inside that face.
@(test)
test_omni_faces_cover_the_sphere :: proc(t: ^testing.T) {
	out: [dynamic]Light_GPU
	defer delete(out)
	lights_convert({lc.Light{kind = .Point, position = {1, 2, 3}, intensity = {1, 1, 1}}}, &out)
	shadows := omni_shadows_build(out[:], {-5, 0, -5}, {5, 4, 5})
	testing.expect_value(t, shadows.count, 1)
	eye := [3]f32{1, 2, 3}
	project :: proc(m: matrix[4, 4]f32, p: [3]f32) -> [4]f32 {
		c := m * [4]f32{p.x, p.y, p.z, 1}
		return {c.x / c.w, c.y / c.w, c.z / c.w, c.w}
	}
	// Mirrors omni_face in shading.slang.
	pick :: proc(v: [3]f32) -> int {
		a := [3]f32{abs(v.x), abs(v.y), abs(v.z)}
		if a.x >= a.y && a.x >= a.z { return v.x > 0 ? 0 : 1 }
		if a.y >= a.z { return v.y > 0 ? 2 : 3 }
		return v.z > 0 ? 4 : 5
	}
	faces := OMNI_FACES
	for dir, face in faces {
		p := project(shadows.view_proj[face], eye + dir * 2)
		testing.expect(t, abs(p.x) < 1e-5 && abs(p.y) < 1e-5 && p.z > 0 && p.z < 1 && p.w > 0)
		testing.expect_value(t, pick(dir), face)
	}
	// A spread of directions, corners and edges included.
	for i in 0 ..< 200 {
		u := (f32(i) + 0.5) / 200
		theta := math.acos(1 - 2 * u)
		phi := f32(i) * 2.3999632
		d := [3]f32{math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi)}
		for corner in ([][3]f32{d, {1, 1, 1}, {-1, 1, -1}, {1, -1, 0}}) {
			face := pick(corner)
			p := project(shadows.view_proj[face], eye + corner * 1.5)
			testing.expectf(t, abs(p.x) <= 1 && abs(p.y) <= 1 && p.w > 0, "%v on face %d lands at %v", corner, face, p)
		}
	}
}

when #config(GPU_TESTS, false) {

// A point light 2 above a floor, a block between them: the floor under the
// block is dark, the floor beside it lit, and the six faces are drawn once.
@(test)
test_gpu_point_light_casts_shadows :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	quad :: proc(tris: []lc.Triangle, y, half: f64) {
		n := lc.Vec3{0, 1, 0}
		a, b, c, d := lc.Point3{-half, y, half}, lc.Point3{half, y, half}, lc.Point3{half, y, -half}, lc.Point3{-half, y, -half}
		tris[0] = lc.Triangle{v0 = a, v1 = b, v2 = c, n0 = n, n1 = n, n2 = n}
		tris[1] = lc.Triangle{v0 = a, v1 = c, v2 = d, n0 = n, n1 = n, n2 = n}
	}
	floor_tris := make([]lc.Triangle, 2)
	quad(floor_tris, 0, 4)
	block_tris := make([]lc.Triangle, 2)
	quad(block_tris, 1, 0.5)
	meshes := make([]lc.Mesh, 2)
	meshes[0] = lc.Mesh{triangles = floor_tris, transform = 1}
	meshes[1] = lc.Mesh{triangles = block_tris, transform = 1}
	nodes := make([]lc.SceneNode, 2)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	nodes[1] = lc.make_node(1, 1, -1, -1)
	materials := make([]lc.Material, 1)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.8, 0.8, 0.8}, roughness = 1}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)
	black := []f32{0, 0, 0}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = black, intensity = 1}
	lights := []lc.Light{{kind = .Point, position = {0, 2, 0}, intensity = {20, 20, 20}}}
	scene.lights = lights

	W :: 64
	H :: 64
	cam := lc.make_camera({0, 6, 0.01}, {0, 0, 0}, {0, 0, -1}, 60, 1, 0, 6)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	r.settings.ao = false
	r.settings.ssr = false
	r.settings.taa = false

	floor_red :: proc(t: ^testing.T, r: ^Renderer, cam: lc.Camera, px, py: int) -> f32 {
		if renderer_render(r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "render") }
		cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
		transfer, downloaded := download_begin(r.gpu, cmd, r.hdr_color, W, H, 8)
		if !downloaded { testing.fail_now(t, "HDR download") }
		defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		defer sdl.ReleaseGPUFence(r.gpu, fence)
		testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
		pixels := make([][4]u16, W * H)
		defer delete(pixels)
		testing.expect(t, map_copy(r.gpu, transfer, pixels))
		return f16_to_f32(pixels[py * W + px][0])
	}

	// As in the spot test: from the eye the block hides the floor out to 0.6,
	// from the light its shadow reaches 1.0. Sampled along x and along z, so
	// the -Y face is checked in two directions, and diagonally outward at
	// 2.5, 2.5, where the floor is lit through the edge between faces.
	to_px :: proc(x: f64) -> int { return W / 2 + int(x / (2 * 6 * math.tan(f64(30 * math.RAD_PER_DEG))) * W) }
	shadow_x := floor_red(t, &r, cam, to_px(0.8), H / 2)
	shadow_z := floor_red(t, &r, cam, W / 2, to_px(0.8))
	lit := floor_red(t, &r, cam, to_px(1.4), H / 2)
	diagonal := floor_red(t, &r, cam, to_px(2.5), to_px(2.5))
	testing.expectf(t, lit > 0.05, "lit floor is %v", lit)
	testing.expectf(t, shadow_x < lit * 0.1 && shadow_z < lit * 0.1, "shadowed floor is %v and %v, lit %v", shadow_x, shadow_z, lit)
	testing.expectf(t, diagonal > 0, "the floor between faces is %v", diagonal)
	testing.expect_value(t, r.omni_layers.draws, 6)

	// Nothing moved: no face is drawn again.
	_ = floor_red(t, &r, cam, 0, 0)
	testing.expect_value(t, r.omni_layers.draws, 6)
}

}
