package lumbre_realtime

import "core:os"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

when #config(GPU_TESTS, false) {

// A floor of albedo 0.5 under a hand-made two-probe grid, in a black world with
// no lights, so the floor shows nothing but the probes' diffuse term:
// albedo * irradiance (kd is 1: no specular reflectance, roughness 1).
//
//   - Both probes open, irradiance 0.4: the floor is 0.2.
//   - The grid's revision no longer matches: back to the black environment.
//   - `gi_probes` off: the same.
//   - The lower probe, under the floor, sees the floor's underside half a
//     cell away and carries irradiance 1. Only the Chebyshev test keeps it
//     out: its facing weight alone would still pull the floor to about 0.24.
@(test)
test_gpu_probes_light_the_floor :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	tris := make([]lc.Triangle, 2)
	n := lc.Vec3{0, 1, 0}
	tris[0] = lc.Triangle{v0 = {-4, 0, 4}, v1 = {4, 0, 4}, v2 = {4, 0, -4}, n0 = n, n1 = n, n2 = n}
	tris[1] = lc.Triangle{v0 = {-4, 0, 4}, v1 = {4, 0, -4}, v2 = {-4, 0, -4}, n0 = n, n1 = n, n2 = n}
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 1)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.5, 0.5, 0.5}, roughness = 1, specular = 0}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)
	black := []f32{0, 0, 0}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = black, intensity = 1}

	// One column of two probes, at y = -0.5 and y = 0.5.
	grid := lc.Probe_Grid {
		revision     = lc.probe_scene_revision(&scene),
		dims         = {1, 2, 1},
		bounds_min   = {-4, -1, -4},
		bounds_max   = {4, 1, 4},
		max_distance = 2,
		irradiance   = make([][3]f32, 2 * lc.PROBE_TILE_TEXELS),
		depth        = make([][2]f32, 2 * lc.PROBE_TILE_TEXELS),
	}
	defer lc.probe_grid_destroy(&grid)
	fill :: proc(g: ^lc.Probe_Grid, probe: int, irradiance: f32, hit: f32) {
		for i in probe * lc.PROBE_TILE_TEXELS ..< (probe + 1) * lc.PROBE_TILE_TEXELS {
			g.irradiance[i] = irradiance
			d := hit / g.max_distance
			g.depth[i] = {d, d * d}
		}
	}

	W :: 33
	H :: 33
	cam := lc.make_camera({0, 3, 0.01}, {0, 0, 0}, {0, 0, -1}, 30, 1, 0, 3)
	r.settings.ao = false
	r.settings.ssr = false
	r.settings.taa = false

	centre_red :: proc(t: ^testing.T, r: ^Renderer, cam: lc.Camera) -> f32 {
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
		return f16_to_f32(pixels[(H / 2) * W + W / 2][0])
	}

	key := u64(1)
	testing.expect(t, renderer_set_scene(&r, &scene, key))

	// Open on both sides: the lower probe's lookups pass through nothing.
	fill(&grid, 0, 0.4, 2)
	fill(&grid, 1, 0.4, 2)
	testing.expect(t, renderer_set_probes(&r, &grid, &scene))
	testing.expect(t, r.probes.active)
	got := centre_red(t, &r, cam)
	testing.expectf(t, abs(got - 0.2) < 0.2 * 0.03, "constant probes: floor %v, want 0.2", got)

	r.settings.gi_probes = false
	got = centre_red(t, &r, cam)
	testing.expectf(t, got < 1e-3, "probes off: floor %v, want 0", got)
	r.settings.gi_probes = true

	// Below the floor, the lower probe hits its underside 0.5 away.
	fill(&grid, 0, 1.0, 0.5)
	testing.expect(t, renderer_set_probes(&r, &grid, &scene))
	got = centre_red(t, &r, cam)
	testing.expectf(t, abs(got - 0.2) < 0.2 * 0.03, "occluded probe: floor %v, want 0.2", got)

	// Stale: baked for some other scene.
	grid.revision += 1
	testing.expect(t, renderer_set_probes(&r, &grid, &scene))
	testing.expect(t, !r.probes.active)
	got = centre_red(t, &r, cam)
	testing.expectf(t, got < 1e-3, "stale probes: floor %v, want 0", got)

	// The revision follows the scene: a new key with the scene unchanged
	// re-checks and stays stale; restoring the revision brings them back.
	grid.revision -= 1
	testing.expect(t, renderer_set_probes(&r, &grid, &scene))
	key += 1
	testing.expect(t, renderer_set_scene(&r, &scene, key))
	testing.expect(t, r.probes.active)
	materials[0].albedo = {0.25, 0.25, 0.25}
	key += 1
	testing.expect(t, renderer_set_scene(&r, &scene, key))
	testing.expect(t, !r.probes.active)
}

}
