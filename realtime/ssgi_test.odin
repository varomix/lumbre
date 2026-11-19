package lumbre_realtime

import "core:os"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

when #config(GPU_TESTS, false) {

// A floor of albedo 0.5 under an even sky of radiance 1, with a wall standing
// on it that is black and emits radiance `wall` (its whole light, the same
// from every side). The camera looks down at the floor in front of the wall.
@(private = "file")
Ssgi_Scene :: struct {
	scene: lc.Scene,
	cam:   lc.Camera,
}

@(private = "file")
SSGI_W :: 256
@(private = "file")
SSGI_H :: 256

@(private = "file")
ssgi_scene :: proc(wall: [3]f64) -> Ssgi_Scene {
	quad :: proc(tris: []lc.Triangle, a, b, c, d: lc.Point3, n: lc.Vec3, mat: i32) {
		tris[0] = lc.Triangle{v0 = a, v1 = b, v2 = c, n0 = n, n1 = n, n2 = n, mat_idx = mat}
		tris[1] = lc.Triangle{v0 = a, v1 = c, v2 = d, n0 = n, n1 = n, n2 = n, mat_idx = mat}
	}
	tris := make([]lc.Triangle, 4)
	quad(tris[0:2], {-4, 0, 4}, {4, 0, 4}, {4, 0, -4}, {-4, 0, -4}, {0, 1, 0}, 0)
	quad(tris[2:4], {-4, 0, -1}, {4, 0, -1}, {4, 8, -1}, {-4, 8, -1}, {0, 0, 1}, 1)
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 2)
	// No specular reflectance: kd is exactly 1 and the floor's ambient is
	// albedo * irradiance, nothing else to separate out.
	materials[0] = lc.Material{kind = .Principled, albedo = {0.5, 0.5, 0.5}, roughness = 1, specular = 0}
	materials[1] = lc.Material{kind = .Emissive, emission = wall, emission_strength = 1, roughness = 1, specular = 0}
	sky := make([]f32, 3)
	sky[0], sky[1], sky[2] = 1, 1, 1
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = sky, intensity = 1}
	return {scene = scene, cam = lc.make_camera({0, 3, 1.5}, {0, 0, -0.5}, {0, 1, 0}, 50, 1, 0, 3.5)}
}

@(private = "file")
ssgi_scene_destroy :: proc(s: ^Ssgi_Scene) {
	delete(s.scene.environment.pixels)
	destroy_test_scene(&s.scene)
}

// The shaded HDR image, before display.
@(private = "file")
ssgi_render :: proc(t: ^testing.T, r: ^Renderer, s: ^Ssgi_Scene, key: u64) -> [][4]f32 {
	testing.expect(t, renderer_set_scene(r, &s.scene, key))
	if renderer_render(r, s.cam, SSGI_W, SSGI_H, .Shaded) == nil { testing.fail_now(t, "render") }
	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	transfer, downloaded := download_begin(r.gpu, cmd, r.hdr_color, SSGI_W, SSGI_H, 8)
	if !downloaded { testing.fail_now(t, "HDR download") }
	defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	defer sdl.ReleaseGPUFence(r.gpu, fence)
	testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
	raw := make([][4]u16, SSGI_W * SSGI_H)
	defer delete(raw)
	testing.expect(t, map_copy(r.gpu, transfer, raw))
	pixels := make([][4]f32, SSGI_W * SSGI_H)
	for p, i in raw {
		pixels[i] = {f16_to_f32(p[0]), f16_to_f32(p[1]), f16_to_f32(p[2]), f16_to_f32(p[3])}
	}
	return pixels
}

// The floor's colour at (0, 0, z), averaged over the 3x3 pixels around it.
@(private = "file")
ssgi_floor_at :: proc(pixels: [][4]f32, cam: lc.Camera, z: f32) -> [3]f32 {
	view_proj := camera_view_proj(camera_frame(cam))
	clip := view_proj * [4]f32{0, 0, z, 1}
	x := int((clip.x / clip.w * 0.5 + 0.5) * SSGI_W)
	y := int((0.5 - clip.y / clip.w * 0.5) * SSGI_H)
	sum: [3]f32
	for dy in -1 ..= 1 {
		for dx in -1 ..= 1 {
			p := pixels[(y + dy) * SSGI_W + x + dx]
			sum += p.rgb
		}
	}
	return sum / 9
}

@(private = "file")
ssgi_renderer :: proc(t: ^testing.T) -> (gpu: ^sdl.GPUDevice, r: Renderer) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu = sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	ok: bool
	r, ok = renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	r.settings.taa = false
	r.settings.ssr = false
	return
}

// SSGI swaps what the sky sends from the directions the screen covers for
// what the screen shows there. A wall exactly as bright as the sky changes
// nothing, so the floor at its foot must read as the open floor does:
// albedo * 1. GTAO alone darkens it (by about a third here), as if the wall
// were black; with SSGI the darkening is undone and the wall's light put
// back. Fails if the screen's light is not added (the foot stays dark), if
// the sky's share is not taken out (the foot is too bright), or if GTAO's
// darkening is left in.
@(test)
test_gpu_ssgi_wall_as_bright_as_the_sky_changes_nothing :: proc(t: ^testing.T) {
	gpu, r := ssgi_renderer(t)
	defer sdl.DestroyGPUDevice(gpu)
	defer renderer_destroy(&r)
	s := ssgi_scene({1, 1, 1})
	defer ssgi_scene_destroy(&s)

	// GTAO's radius is 0.12 here and SSGI's 0.8: the foot is inside both,
	// the open floor outside both.
	foot_z: f32 = -1 + 0.06
	open_z: f32 = -1 + 1.5

	r.settings.ssgi = false
	off := ssgi_render(t, &r, &s, 1)
	defer delete(off)
	r.settings.ssgi = true
	on := ssgi_render(t, &r, &s, 1)
	defer delete(on)

	open_off := ssgi_floor_at(off, s.cam, open_z)
	foot_off := ssgi_floor_at(off, s.cam, foot_z)
	open_on := ssgi_floor_at(on, s.cam, open_z)
	foot_on := ssgi_floor_at(on, s.cam, foot_z)
	testing.expectf(t, abs(open_on.g - 0.5) < 0.01, "open floor is %v, want 0.5", open_on.g)
	// Only by undoing GTAO's slight darkening of a flat floor (about 1%,
	// see "GTAO" in docs/realtime-stabilization.md).
	testing.expectf(t, abs(open_on.g - open_off.g) < 0.01, "SSGI changed the open floor: %v to %v", open_off.g, open_on.g)
	testing.expectf(t, foot_off.g < 0.45, "without SSGI GTAO darkens the foot: %v", foot_off.g)
	testing.expectf(t, abs(foot_on.g - 0.5) < 0.015, "with SSGI the foot of a sky-bright wall is %v, want 0.5", foot_on.g)
}

// A wall emitting red 2, green 0 beside the same floor. At its foot the wall
// fills close to half the cosine-weighted hemisphere (exactly half for an
// unbounded wall; less here, as SSGI looks only 0.8 away and the wall's far
// parts are left to the sky). So the floor there gets about
// 0.5 * (0.5 * 1 + 0.5 * 2) = 0.75 red and 0.5 * 0.5 * 1 = 0.25 green, where
// the open floor has 0.5 of each. Fails if the wall's colour is not gathered
// or the sky it hides not taken out.
@(test)
test_gpu_ssgi_bleeds_a_walls_colour_onto_the_floor :: proc(t: ^testing.T) {
	gpu, r := ssgi_renderer(t)
	defer sdl.DestroyGPUDevice(gpu)
	defer renderer_destroy(&r)
	s := ssgi_scene({2, 0, 0})
	defer ssgi_scene_destroy(&s)

	on := ssgi_render(t, &r, &s, 1)
	defer delete(on)
	foot := ssgi_floor_at(on, s.cam, -1 + 0.06)
	open := ssgi_floor_at(on, s.cam, -1 + 1.5)
	testing.expectf(t, abs(open.r - 0.5) < 0.01 && abs(open.g - 0.5) < 0.01, "open floor is %v, want 0.5 grey", open)
	testing.expectf(t, foot.r > 0.68 && foot.r < 0.77, "red at the foot is %v, want about 0.75", foot.r)
	testing.expectf(t, foot.g > 0.24 && foot.g < 0.31, "green at the foot is %v, want about 0.25", foot.g)
}

}
