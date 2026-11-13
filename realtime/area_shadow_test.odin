package lumbre_realtime

import "core:os"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

when #config(GPU_TESTS, false) {

// A disc light of radius R = 0.5, 2 up, facing down, over a floor, and a
// half-plane occluder at height 1 covering x < 0. From a floor point at x, the
// occluder's edge, straight above x = 0, hides the part of the disc beyond
// x (1 - 2 / 1) = -x: half of it at x = 0, all of it from x = -0.5 back, none
// from x = +0.5 on. Visibility is the floor with the occluder over the floor
// without it, which cancels the light's own falloff.
//
// A point light's hard shadow -- what every area light had before -- steps
// from 0 to 1 at x = 0 and fails the penumbra checks.
@(test)
test_gpu_area_light_penumbra :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	up := lc.Vec3{0, 1, 0}
	quad :: proc(tris: []lc.Triangle, a, b, c, d: lc.Point3, n: lc.Vec3) {
		tris[0] = lc.Triangle{v0 = a, v1 = b, v2 = c, n0 = n, n1 = n, n2 = n}
		tris[1] = lc.Triangle{v0 = a, v1 = c, v2 = d, n0 = n, n1 = n, n2 = n}
	}
	tris := make([]lc.Triangle, 4)
	quad(tris[0:2], {-4, 0, 4}, {4, 0, 4}, {4, 0, -4}, {-4, 0, -4}, up)
	quad(tris[2:4], {-4, 1, 4}, {0, 1, 4}, {0, 1, -4}, {-4, 1, -4}, up)
	meshes := make([]lc.Mesh, 2)
	meshes[0] = lc.Mesh{triangles = tris[0:2], transform = 1, borrowed_triangles = true}
	meshes[1] = lc.Mesh{triangles = tris[2:4], transform = 1, borrowed_triangles = true}
	defer delete(tris)
	nodes := make([]lc.SceneNode, 2)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	nodes[1] = lc.make_node(1, 1, -1, -1)
	materials := make([]lc.Material, 1)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.5, 0.5, 0.5}, roughness = 1, specular = 0}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)
	black := []f32{0, 0, 0}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = black, intensity = 1}
	lights := []lc.Light{lc.make_disc_light({0, 2, 0}, {0, -1, 0}, 0.5, {4, 4, 4})}
	scene.lights = lights

	// Straight down from under the occluder's height, so the camera sees the
	// floor through nothing: the occluder is a one-sided pane above it, and
	// below 1 the camera is under it.
	W :: 129
	H :: 33
	cam := lc.make_camera({0, 0.9, 0.001}, {0, 0, 0}, {0, 0, -1}, 90, f64(W) / f64(H), 0, 0.9)
	r.settings.ao = false
	r.settings.ssr = false
	r.settings.taa = false
	r.settings.fxaa = false

	floor_row :: proc(t: ^testing.T, r: ^Renderer, cam: lc.Camera, out: []f32) {
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
		for x in 0 ..< W { out[x] = f16_to_f32(pixels[(H / 2) * W + x][0]) }
	}

	// Unoccluded first: the occluder's node points at no mesh.
	nodes[1].mesh_idx = -1
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	open, shaded: [W]f32
	floor_row(t, &r, cam, open[:])
	nodes[1].mesh_idx = 1
	testing.expect(t, renderer_set_scene(&r, &scene, 2))
	floor_row(t, &r, cam, shaded[:])

	// Floor x of pixel column px: a 90-degree VERTICAL field from 0.9 up,
	// so the row spans 0.9 * tan(45 deg) * W / H either side.
	visibility_at :: proc(open, shaded: []f32, x: f32) -> f32 {
		half_width := f32(0.9) * f32(W) / f32(H)
		px := int((x / half_width * 0.5 + 0.5) * f32(W))
		px = clamp(px, 0, W - 1)
		return shaded[px] / max(open[px], 1e-6)
	}
	v := proc(x: f32, open, shaded: []f32) -> f32 { return visibility_at(open, shaded, x) }
	testing.expectf(t, v(-0.7, open[:], shaded[:]) < 0.1, "deep in the umbra: %v", v(-0.7, open[:], shaded[:]))
	testing.expectf(t, v(0.7, open[:], shaded[:]) > 0.9, "past the penumbra: %v", v(0.7, open[:], shaded[:]))
	mid := v(0, open[:], shaded[:])
	testing.expectf(t, mid > 0.3 && mid < 0.7, "under the edge, half the disc is hidden: %v", mid)
	inner := v(-0.25, open[:], shaded[:])
	outer := v(0.25, open[:], shaded[:])
	testing.expectf(t, inner > 0.03 && inner < mid, "inside the penumbra: %v (edge %v)", inner, mid)
	testing.expectf(t, outer > mid && outer < 0.97, "outside the penumbra: %v (edge %v)", outer, mid)
}

}
