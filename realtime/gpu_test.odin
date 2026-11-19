package lumbre_realtime

import "core:math"
import "core:os"
import "core:fmt"
import "base:runtime"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

// Explicitly enabled so regular unit tests remain usable without a GPU.
// odin test realtime -define:GPU_TESTS=true -define:ODIN_TEST_THREADS=1
when #config(GPU_TESTS, false) {
@(init)
gpu_test_init :: proc "contextless" () {
	context = runtime.default_context()
	if !sdl.Init({.VIDEO}) { fmt.eprintln("GPU test SDL init:", sdl.GetError()) }
}
@(fini)
gpu_test_shutdown :: proc "contextless" () { sdl.Quit() }

@(test)
test_gpu_hdr_and_resource_reuse :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)
	for &mat in scene.materials { mat.kind = .Emissive; mat.emission = {4, 2, 1}; mat.emission_strength = 1 }
	cam := lc.make_camera({1, 12, 1}, {1, 0, 1}, {0, 0, -1}, 50, 1, 0, 12)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	vertices := r.scene.vertices
	environment := r.env.irradiance
	// Fresh scene revision, no geometry change: all expensive work is reused.
	testing.expect(t, renderer_set_scene(&r, &scene, 2))
	testing.expect(t, r.scene.vertices == vertices && r.env.irradiance == environment)
	testing.expect_value(t, r.geometry_uploads, 1)
	testing.expect_value(t, r.environment_builds, 1)
	scene.materials[0].emission.x = 8
	testing.expect(t, renderer_set_scene(&r, &scene, 3))
	testing.expect(t, r.scene.vertices == vertices)
	testing.expect_value(t, r.scene.batches[0].material.emission.x, 8)
	if renderer_render(&r, cam, 32, 32, .Shaded) == nil { testing.fail_now(t, "HDR render") }
	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	transfer, downloaded := download_begin(gpu, cmd, r.hdr_color, 32, 32, 8)
	if !downloaded { testing.fail_now(t, "HDR download") }
	defer sdl.ReleaseGPUTransferBuffer(gpu, transfer)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	if fence == nil { testing.fail_now(t, "HDR submit") }
	defer sdl.ReleaseGPUFence(gpu, fence)
	testing.expect(t, bool(sdl.WaitForGPUFences(gpu, true, &fence, 1)))
	pixels := make([][4]u16, 32 * 32)
	defer delete(pixels)
	testing.expect(t, map_copy(gpu, transfer, pixels))
	peak: f32
	for pixel in pixels { peak = max(peak, f16_to_f32(pixel[0])) }
	testing.expect(t, peak >= 4, "linear lighting must preserve values above one")
	// All debug bindings must be valid, even before label targets exist.
	for view in Debug_View {
		testing.expect(t, renderer_render(&r, cam, 32, 32, view) != nil)
	}
	// A changed transform rewrites instances only: no geometry, no environment.
	scene.nodes[0].local_transform[0, 3] += 1
	testing.expect(t, renderer_set_scene(&r, &scene, 4))
	testing.expect(t, r.scene.vertices == vertices)
	testing.expect_value(t, r.geometry_uploads, 1)
	testing.expect_value(t, r.instance_uploads, 1)
	testing.expect_value(t, r.environment_builds, 1)
	if renderer_render(&r, cam, 32, 32, .Shaded) == nil { testing.fail_now(t, "render after moving an instance") }
	// A CPU re-import owns new pixel arrays, but equal image content must not
	// upload another GPU texture.
	pixels_a := make([]u8, 4)
	pixels_b := make([]u8, 4)
	defer delete(pixels_a)
	defer delete(pixels_b)
	for &v in pixels_a { v = 255 }
	copy(pixels_b, pixels_a)
	scene.materials[0].albedo_tex = {width = 1, height = 1, has_data = true, srgb = true, pixels = pixels_a}
	testing.expect(t, renderer_set_scene(&r, &scene, 5))
	texture := r.scene.batches[0].albedo
	scene.materials[0].albedo_tex.pixels = pixels_b
	testing.expect(t, renderer_set_scene(&r, &scene, 6))
	testing.expect(t, r.scene.batches[0].albedo == texture)
	testing.expect_value(t, len(r.texture_cache), 1)
	pixels_b[0] = 16
	testing.expect(t, renderer_set_scene(&r, &scene, 7))
	testing.expect(t, r.scene.batches[0].albedo != texture)
	testing.expect_value(t, len(r.texture_cache), 1)
	testing.expect_value(t, r.geometry_uploads, 1)
	// Rotation and intensity do not participate in environment convolution.
	env_pixels := []f32{1, 1, 1}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = env_pixels, intensity = 1}
	testing.expect(t, renderer_set_scene(&r, &scene, 8))
	env_texture := r.env.irradiance
	scene.environment.rotation = 0.75
	scene.environment.intensity = 2
	testing.expect(t, renderer_set_scene(&r, &scene, 9))
	testing.expect_value(t, r.environment_builds, 2)
	testing.expect(t, r.env.irradiance == env_texture)
	testing.expect_value(t, r.env.intensity, 2)
}

@(test)
test_gpu_shadow_layers_redraw_only_when_stale :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)
	lights := make([]lc.Light, 1)
	defer delete(lights)
	lights[0] = lc.Light{kind = .Distant, direction = {-0.3, -1, -0.2}, intensity = {3, 3, 3}}
	scene.lights = lights
	cam := lc.make_camera({1, 12, 1}, {1, 0, 1}, {0, 0, -1}, 50, 1, 0, 12)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))

	if renderer_render(&r, cam, 32, 32, .Shaded) == nil { testing.fail_now(t, "first render") }
	testing.expect_value(t, r.shadow_layer_draws, u64(CASCADE_COUNT))
	// A look change redraws the frame, but nothing the light sees moved.
	r.settings.exposure = 1
	if renderer_render(&r, cam, 32, 32, .Shaded) == nil { testing.fail_now(t, "look change") }
	testing.expect_value(t, r.shadow_layer_draws, u64(CASCADE_COUNT))
	// Moving a caster invalidates every layer.
	scene.nodes[0].local_transform[0, 3] += 1
	testing.expect(t, renderer_set_scene(&r, &scene, 2))
	if renderer_render(&r, cam, 32, 32, .Shaded) == nil { testing.fail_now(t, "moved caster") }
	testing.expect_value(t, r.shadow_layer_draws, u64(2 * CASCADE_COUNT))
}

// A polished floor in front of a glowing wall: with reflections on, the floor
// must show the wall. The size is odd both ways, so the depth pyramid's
// absorbed last row and column are on the path being tested.
@(test)
test_gpu_reflections_show_the_scene :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	quad :: proc(tris: []lc.Triangle, a, b, c, d: lc.Point3, n: lc.Vec3, mat: i32) {
		tris[0] = lc.Triangle{v0 = a, v1 = b, v2 = c, n0 = n, n1 = n, n2 = n, mat_idx = mat}
		tris[1] = lc.Triangle{v0 = a, v1 = c, v2 = d, n0 = n, n1 = n, n2 = n, mat_idx = mat}
	}
	tris := make([]lc.Triangle, 4)
	quad(tris[0:2], {-4, 0, 4}, {4, 0, 4}, {4, 0, -4}, {-4, 0, -4}, {0, 1, 0}, 0)
	quad(tris[2:4], {-4, 0, -4}, {4, 0, -4}, {4, 4, -4}, {-4, 4, -4}, {0, 0, 1}, 1)
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 2)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.05, 0.05, 0.05}, roughness = 0.05}
	materials[1] = lc.Material{kind = .Emissive, emission = {4, 2, 1}, emission_strength = 1}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)

	W :: 67
	H :: 45
	cam := lc.make_camera({0, 2, 6}, {0, 0.5, -2}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 8)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	testing.expect_value(t, r.settings.ssr, true)

	// Red at a floor pixel just in front of the wall, from the linear HDR
	// target. The centre column on purpose: its rays run straight up the
	// screen, crossing no cell boundary sideways, which the march once
	// mistook for crossing one infinitely far behind it.
	floor_red :: proc(t: ^testing.T, r: ^Renderer, cam: lc.Camera) -> f32 {
		if renderer_render(r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "render") }
		cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
		transfer, downloaded := download_begin(r.gpu, cmd, r.hdr_color, W, H, 8)
		if !downloaded { testing.fail_now(t, "HDR download") }
		defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		if fence == nil { testing.fail_now(t, "HDR submit") }
		defer sdl.ReleaseGPUFence(r.gpu, fence)
		testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
		pixels := make([][4]u16, W * H)
		defer delete(pixels)
		testing.expect(t, map_copy(r.gpu, transfer, pixels))
		return f16_to_f32(pixels[(H * 2 / 3) * W + W / 2][0])
	}

	reflecting := floor_red(t, &r, cam)
	testing.expect_value(t, r.hiz_levels, 7) // 67 wide: 67, 33, 16, 8, 4, 2, 1
	r.settings.ssr = false
	plain := floor_red(t, &r, cam)
	testing.expectf(t, reflecting > plain + 0.1, "floor is %v with reflections, %v without", reflecting, plain)
}

// A floor in front of a narrow glowing strip, at three roughnesses. Glossy
// reflections must:
//   - still show at roughness 0.6, past where the single mirror ray used to
//     fade out (0.45);
//   - spread sideways: beside the strip's mirror image a glossy floor picks up
//     its light and a near-mirror does not;
//   - render the same twice, so a still is reproducible.
@(test)
test_gpu_glossy_reflections_spread :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	quad :: proc(tris: []lc.Triangle, a, b, c, d: lc.Point3, n: lc.Vec3, mat: i32) {
		tris[0] = lc.Triangle{v0 = a, v1 = b, v2 = c, n0 = n, n1 = n, n2 = n, mat_idx = mat}
		tris[1] = lc.Triangle{v0 = a, v1 = c, v2 = d, n0 = n, n1 = n, n2 = n, mat_idx = mat}
	}
	tris := make([]lc.Triangle, 4)
	quad(tris[0:2], {-4, 0, 4}, {4, 0, 4}, {4, 0, -4}, {-4, 0, -4}, {0, 1, 0}, 0)
	quad(tris[2:4], {-0.3, 0, -2}, {0.3, 0, -2}, {0.3, 3, -2}, {-0.3, 3, -2}, {0, 0, 1}, 1)
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 2)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.8, 0.8, 0.8}, metallic = 1, roughness = 0.02}
	materials[1] = lc.Material{kind = .Emissive, emission = {4, 4, 4}, emission_strength = 1}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)
	black := []f32{0, 0, 0}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = black, intensity = 1}

	W :: 81
	H :: 61
	cam := lc.make_camera({0, 1.5, 3}, {0, 0.3, -2}, {0, 1, 0}, 50, f64(W) / f64(H), 0, 5)
	r.settings.ao = false
	r.settings.taa = false

	frame :: proc(t: ^testing.T, r: ^Renderer, cam: lc.Camera) -> [][4]f32 {
		if renderer_render(r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "render") }
		cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
		transfer, downloaded := download_begin(r.gpu, cmd, r.hdr_color, W, H, 8)
		if !downloaded { testing.fail_now(t, "HDR download") }
		defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		defer sdl.ReleaseGPUFence(r.gpu, fence)
		testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
		raw := make([][4]u16, W * H)
		defer delete(raw)
		testing.expect(t, map_copy(r.gpu, transfer, raw))
		out := make([][4]f32, W * H, context.temp_allocator)
		for p, i in raw { out[i] = {f16_to_f32(p[0]), f16_to_f32(p[1]), f16_to_f32(p[2]), 1} }
		return out
	}
	// A floor row a little below the strip's foot: the strip's mirror image
	// at the centre column, and a column beside it.
	ROW :: H * 3 / 4
	BESIDE :: W / 2 + 8

	set_roughness :: proc(t: ^testing.T, r: ^Renderer, scene: ^lc.Scene, roughness: f64, key: u64) {
		scene.materials[0].roughness = roughness
		testing.expect(t, renderer_set_scene(r, scene, key))
	}

	set_roughness(t, &r, &scene, 0.02, 1)
	mirror := frame(t, &r, cam)
	set_roughness(t, &r, &scene, 0.3, 2)
	glossy := frame(t, &r, cam)
	again := frame(t, &r, cam)
	set_roughness(t, &r, &scene, 0.6, 3)
	rough := frame(t, &r, cam)
	r.settings.ssr = false
	rough_plain := frame(t, &r, cam)

	testing.expectf(t, mirror[ROW * W + W / 2].x > 0.5, "mirror shows the strip: %v", mirror[ROW * W + W / 2].x)
	testing.expectf(t, mirror[ROW * W + BESIDE].x < 0.02, "mirror beside the strip: %v", mirror[ROW * W + BESIDE].x)
	testing.expectf(t, glossy[ROW * W + BESIDE].x > 0.05, "glossy beside the strip: %v", glossy[ROW * W + BESIDE].x)
	same := true
	for p, i in glossy { if p != again[i] { same = false; break } }
	testing.expect(t, same, "a still renders the same twice")
	// Nearer the strip's foot: a lobe this wide spreads a narrow strip thin.
	NEAR :: (ROW - 6) * W + W / 2
	testing.expectf(t, rough[NEAR].x > rough_plain[NEAR].x + 0.05,
		"roughness 0.6 floor is %v with reflections, %v without", rough[NEAR].x, rough_plain[NEAR].x)
}

// Fog between the eye and a glowing wall, under a uniform white sky and no
// lights: the wall must dim by exactly the transmittance of the air in front
// of it, and the air must add the sky's share. A level ray crosses fog of one
// density, so the answer has a closed form.
@(test)
test_gpu_fog_matches_beer_lambert :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	// The floor and wall of the reflection test: 8 wide and deep, 4 high.
	quad :: proc(tris: []lc.Triangle, a, b, c, d: lc.Point3, n: lc.Vec3, mat: i32) {
		tris[0] = lc.Triangle{v0 = a, v1 = b, v2 = c, n0 = n, n1 = n, n2 = n, mat_idx = mat}
		tris[1] = lc.Triangle{v0 = a, v1 = c, v2 = d, n0 = n, n1 = n, n2 = n, mat_idx = mat}
	}
	tris := make([]lc.Triangle, 4)
	quad(tris[0:2], {-4, 0, 4}, {4, 0, 4}, {4, 0, -4}, {-4, 0, -4}, {0, 1, 0}, 0)
	quad(tris[2:4], {-4, 0, -4}, {4, 0, -4}, {4, 4, -4}, {-4, 4, -4}, {0, 0, 1}, 1)
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 2)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.05, 0.05, 0.05}, roughness = 0.9}
	materials[1] = lc.Material{kind = .Emissive, emission = {4, 2, 1}, emission_strength = 1}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)
	env_pixels := []f32{1, 1, 1}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = env_pixels, intensity = 1}

	W :: 65
	H :: 45
	// Level, at height 2, looking at the wall 10 away.
	cam := lc.make_camera({0, 2, 6}, {0, 2, -4}, {0, 1, 0}, 40, f64(W) / f64(H), 0, 10)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	testing.expect_value(t, r.settings.fog, false)

	centre_red :: proc(t: ^testing.T, r: ^Renderer, cam: lc.Camera) -> f32 {
		if renderer_render(r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "render") }
		cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
		transfer, downloaded := download_begin(r.gpu, cmd, r.hdr_color, W, H, 8)
		if !downloaded { testing.fail_now(t, "HDR download") }
		defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		if fence == nil { testing.fail_now(t, "HDR submit") }
		defer sdl.ReleaseGPUFence(r.gpu, fence)
		testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
		pixels := make([][4]u16, W * H)
		defer delete(pixels)
		testing.expect(t, map_copy(r.gpu, transfer, pixels))
		return f16_to_f32(pixels[(H / 2) * W + W / 2][0])
	}

	clear := centre_red(t, &r, cam)
	testing.expectf(t, abs(clear - 4) < 0.01, "the wall is %v without fog, want its emission, 4", clear)

	r.settings.fog = true
	r.settings.fog_density = 1
	r.settings.fog_height = 0.5
	// Extinction 1/8 at the floor (the scene is 8 across), thinning by e
	// every 0.5 * 4 = 2 up; at height 2, over 10 units.
	transmittance := math.exp(f32(-(1.0 / 8.0) * math.exp(f32(-1)) * 10))
	want := clear * transmittance + 1 * (1 - transmittance)
	foggy := centre_red(t, &r, cam)
	testing.expectf(t, abs(foggy - want) < 0.03, "the wall is %v through fog, want %v", foggy, want)

	// No density is no fog, not a pass that happens to do nothing.
	r.settings.fog_density = 0
	testing.expect_value(t, centre_red(t, &r, cam), clear)
}

}
