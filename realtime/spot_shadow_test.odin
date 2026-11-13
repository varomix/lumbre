package lumbre_realtime

import "core:math"
import "core:os"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

@(private = "file")
spot :: proc(position, direction: lc.Vec3, half_angle_deg: f64) -> lc.Light {
	c := math.cos(half_angle_deg * math.RAD_PER_DEG)
	return lc.Light {
		kind = .Spot, position = position, direction = direction,
		intensity = {200, 200, 200}, cos_inner = c, cos_outer = c,
	}
}

// Spots take layers in scene order up to the limit; nothing else takes one.
@(test)
test_spot_shadow_layers :: proc(t: ^testing.T) {
	lights := make([dynamic]lc.Light, context.temp_allocator)
	append(&lights, lc.Light{kind = .Point, position = {0, 1, 0}, intensity = {1, 1, 1}})
	for i in 0 ..< MAX_SPOT_SHADOWS + 2 {
		append(&lights, spot({f64(i), 3, 0}, {0, -1, 0}, 30))
	}
	append(&lights, lc.Light{kind = .Distant, direction = {0, -1, 0}, intensity = {1, 1, 1}})
	out: [dynamic]Light_GPU
	defer delete(out)
	lights_convert(lights[:], &out)
	testing.expect_value(t, out[0].params.w, -1)
	for i in 0 ..< MAX_SPOT_SHADOWS {
		testing.expect_value(t, out[1 + i].params.w, f32(i))
	}
	testing.expect_value(t, out[1 + MAX_SPOT_SHADOWS].params.w, -1)
	testing.expect_value(t, out[len(out) - 1].params.w, -1)

	shadows := spot_shadows_build(out[:], {-5, 0, -5}, {5, 3, 5})
	testing.expect_value(t, shadows.count, MAX_SPOT_SHADOWS)
}

// A point straight down the axis lands mid-map, nearer points at smaller
// depth; a point outside the cone falls outside the map, one behind the light
// behind the projection.
@(test)
test_spot_shadow_projection :: proc(t: ^testing.T) {
	out: [dynamic]Light_GPU
	defer delete(out)
	lights_convert({spot({0, 4, 0}, {0, -1, 0}, 30)}, &out)
	shadows := spot_shadows_build(out[:], {-5, 0, -5}, {5, 4, 5})
	testing.expect_value(t, shadows.count, 1)
	m := shadows.view_proj[0]
	project :: proc(m: matrix[4, 4]f32, p: [3]f32) -> [4]f32 {
		c := m * [4]f32{p.x, p.y, p.z, 1}
		return {c.x / c.w, c.y / c.w, c.z / c.w, c.w}
	}
	floor := project(m, {0, 0, 0})
	mid := project(m, {0, 2, 0})
	testing.expect(t, abs(floor.x) < 1e-5 && abs(floor.y) < 1e-5)
	testing.expect(t, mid.z > 0 && mid.z < floor.z && floor.z < 1)
	// 30 degrees from the axis at 4 below the light: tan(30) * 4 = 2.31 across.
	inside := project(m, {2.2, 0, 0})
	outside := project(m, {3, 0, 0})
	testing.expect(t, abs(inside.x) < 1 && abs(outside.x) > 1)
	testing.expect(t, project(m, {0, 6, 0}).w < 0)
	// The texel at the floor covers 2 * 4 * tan(half) / resolution.
	testing.expect(t, shadows.texel_scale[0] * 4 > 2 * 4 * math.tan(f32(30 * math.RAD_PER_DEG)) / SPOT_SHADOW_RESOLUTION)
}

when #config(GPU_TESTS, false) {

// A spot straight down onto a floor with a raised square under it: the floor
// beneath the square is dark, the floor beside it lit, and moving the camera
// draws no spot layer while moving the square redraws one.
@(test)
test_gpu_spot_light_casts_shadows :: proc(t: ^testing.T) {
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
	lights := []lc.Light{spot({0, 2, 0}, {0, -1, 0}, 40)}
	scene.lights = lights

	W :: 64
	H :: 64
	// Looking straight down from beside the light.
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

	// From the eye 6 up, the block (top at 1, half-width 0.5) hides the floor
	// out to x = 0.5 * 6 / 5 = 0.6. From the light 2 up, its shadow reaches
	// x = 0.5 * 2 / 1 = 1.0. So the floor at x = 0.8 is visible and shadowed,
	// and at 1.4 visible and lit, inside the cone's 2 * tan(40) = 1.68.
	// On screen: a 60 degree fov over 64 px spans 6.93 at the floor.
	to_px :: proc(x: f64) -> int { return W / 2 + int(x / (2 * 6 * math.tan(f64(30 * math.RAD_PER_DEG))) * W) }
	shadowed := floor_red(t, &r, cam, to_px(0.8), H / 2)
	lit := floor_red(t, &r, cam, to_px(1.4), H / 2)
	testing.expectf(t, lit > 0.05, "lit floor is %v", lit)
	testing.expectf(t, shadowed < lit * 0.1, "shadowed floor is %v, lit %v", shadowed, lit)
	testing.expect_value(t, r.spot_layer_draws, 1)

	// The camera moves: no spot layer is redrawn.
	moved := lc.make_camera({0.3, 6, 0.01}, {0.3, 0, 0}, {0, 0, -1}, 60, 1, 0, 6)
	_ = floor_red(t, &r, moved, 0, 0)
	testing.expect_value(t, r.spot_layer_draws, 1)

	// The block moves away: its layer is redrawn and the floor is lit again.
	scene.nodes[1].local_transform[0, 3] += 3
	testing.expect(t, renderer_set_scene(&r, &scene, 2))
	relit := floor_red(t, &r, cam, to_px(0.8), H / 2)
	testing.expect_value(t, r.spot_layer_draws, 2)
	testing.expectf(t, relit > lit * 0.5, "floor is %v once the block has moved", relit)
}

}
