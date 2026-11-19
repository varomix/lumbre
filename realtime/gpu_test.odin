package lumbre_realtime

import "core:math"
import "core:math/linalg"
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

// GTAO is ambient visibility itself, so it has bounds it must keep. A tall
// wall is a half-space: from a floor point at its foot it can hide at most
// half the cosine-weighted sky, so visibility never drops below 0.5. Moving
// away, the wall drops out of the radius and visibility rises, to exactly 1
// on the open floor. A wall point counts fully within 0.4 of the radius, so a
// tenth of a radius out the horizon toward it is about 75 degrees and the
// answer about 0.63-0.65, not 0.5.
@(test)
test_gpu_gtao_sees_half_the_sky_at_a_wall :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	quad :: proc(tris: []lc.Triangle, a, b, c, d: lc.Point3, n: lc.Vec3) {
		tris[0] = lc.Triangle{v0 = a, v1 = b, v2 = c, n0 = n, n1 = n, n2 = n}
		tris[1] = lc.Triangle{v0 = a, v1 = c, v2 = d, n0 = n, n1 = n, n2 = n}
	}
	tris := make([]lc.Triangle, 4)
	quad(tris[0:2], {-4, 0, 4}, {4, 0, 4}, {4, 0, -4}, {-4, 0, -4}, {0, 1, 0})
	quad(tris[2:4], {-4, 0, -1}, {4, 0, -1}, {4, 8, -1}, {-4, 8, -1}, {0, 0, 1})
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 1)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.5, 0.5, 0.5}, roughness = 1}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)

	W :: 96
	H :: 96
	cam := lc.make_camera({0, 3, 1.5}, {0, 0, -0.5}, {0, 1, 0}, 50, 1, 0, 3.5)
	r.settings.taa = false
	// The scene's largest extent is 8, so the radius is 8 * AO_RADIUS_FRACTION
	// times this: about 1.2.
	r.settings.ao_radius = 10
	radius := f64(8 * AO_RADIUS_FRACTION * r.settings.ao_radius)
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	if renderer_render(&r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "render") }

	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	transfer, downloaded := download_begin(r.gpu, cmd, r.ao, W, H, 1)
	if !downloaded { testing.fail_now(t, "AO download") }
	defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	defer sdl.ReleaseGPUFence(r.gpu, fence)
	testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
	ao := make([]u8, W * H)
	defer delete(ao)
	testing.expect(t, map_copy(r.gpu, transfer, ao))

	// The AO at a floor point, averaged over the 3x3 pixels around where it
	// lands on screen.
	view_proj := camera_view_proj(camera_frame(cam))
	at :: proc(ao: []u8, view_proj: matrix[4, 4]f32, p: [3]f32) -> f32 {
		clip := view_proj * [4]f32{p.x, p.y, p.z, 1}
		x := int((clip.x / clip.w * 0.5 + 0.5) * W)
		y := int((0.5 - clip.y / clip.w * 0.5) * H)
		sum: f32
		for dy in -1 ..= 1 {
			for dx in -1 ..= 1 {
				sum += f32(ao[(y + dy) * W + x + dx]) / 255
			}
		}
		return sum / 9
	}
	foot := at(ao, view_proj, {0, 0, f32(-1 + 0.1 * radius)})
	half := at(ao, view_proj, {0, 0, f32(-1 + 0.5 * radius)})
	open := at(ao, view_proj, {0, 0, f32(-1 + 1.5 * radius)})
	testing.expectf(t, foot >= 0.48 && foot < 0.72, "visibility a tenth of a radius from the wall is %v, want 0.5 to 0.7", foot)
	testing.expectf(t, half > foot + 0.05, "visibility rises away from the wall: %v at 0.1 radius, %v at 0.5", foot, half)
	testing.expectf(t, open > 0.95, "visibility on the open floor is %v, want 1", open)
}

// Fog beside a quad light, in black surroundings, single scattering, g = 0:
// each step in the air scatters the light's radiance times the solid angle the
// panel fills from there, over 4 pi, dimmed by the fog between. Checked against that integral along the
// ray, with the solid angle summed from a fine grid over the panel -- not the
// polygon formula the shader uses -- for rays straight through the panel and
// for rays passing 0.7 beside it. The old point approximation, radiance times
// area cos / d^2, gave three times the integral through the panel; beside it,
// it was already within 5%.
@(test)
test_gpu_fog_lit_by_an_area_light :: proc(t: ^testing.T) {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	defer sdl.DestroyGPUDevice(gpu)
	r, ok := renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	defer renderer_destroy(&r)

	// A floor far below sets the bounds; the camera looks level, so its
	// rays see nothing but fog.
	tris := make([]lc.Triangle, 2)
	n := lc.Vec3{0, 1, 0}
	tris[0] = lc.Triangle{v0 = {-4, -4, 4}, v1 = {4, -4, 4}, v2 = {4, -4, -4}, n0 = n, n1 = n, n2 = n}
	tris[1] = lc.Triangle{v0 = {-4, -4, 4}, v1 = {4, -4, -4}, v2 = {-4, -4, -4}, n0 = n, n1 = n, n2 = n}
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 1)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.5, 0.5, 0.5}, roughness = 1}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)
	black := []f32{0, 0, 0}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = black, intensity = 1}
	// A 1 x 1 panel facing +z, centred at (0, 0, -2); radiance 4.
	corner := lc.Point3{-0.5, -0.5, -2}
	u := lc.Vec3{1, 0, 0}
	v := lc.Vec3{0, 1, 0}
	lights := []lc.Light{lc.make_area_light(corner, u, v, {4, 4, 4})}
	scene.lights = lights

	W :: 64
	H :: 64
	eye := lc.Point3{0, 0, 2}
	cam := lc.make_camera(eye, {0, 0, -2}, {0, 1, 0}, 60, 1, 0, 4)
	r.settings.fog = true
	r.settings.fog_density = 0.8
	r.settings.fog_height = 1000 // uniform
	r.settings.fog_anisotropy = 0
	r.settings.fog_multiple = false
	r.settings.ao = false
	r.settings.ssr = false
	r.settings.taa = false
	testing.expect(t, renderer_set_scene(&r, &scene, 1))
	if renderer_render(&r, cam, W, H, .Shaded) == nil { testing.fail_now(t, "render") }

	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	transfer, downloaded := download_begin(r.gpu, cmd, r.fog, W, H, 8)
	if !downloaded { testing.fail_now(t, "fog download") }
	defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	defer sdl.ReleaseGPUFence(r.gpu, fence)
	testing.expect(t, bool(sdl.WaitForGPUFences(r.gpu, true, &fence, 1)))
	raw := make([][4]u16, W * H)
	defer delete(raw)
	testing.expect(t, map_copy(r.gpu, transfer, raw))

	// A 4x4 block, which the steps' 4x4 offset pattern covers once: its mean
	// scattering, and the same pixels' rays integrated finely. The march's
	// density and reach are the renderer's: density / largest extent, out to
	// twice the eye's distance to the far side of the bounds.
	block :: proc(raw: [][4]u16, cam: lc.Camera, r: ^Renderer, x0, y0: int) -> (got, want: f64) {
		for y in y0 ..< y0 + 4 {
			for x in x0 ..< x0 + 4 {
				got += f64(f16_to_f32(raw[y * W + x][0])) / 16
			}
		}
		sigma := f64(r.settings.fog_density) / 8
		extent := r.scene.bounds_max - r.scene.bounds_min
		centre := (r.scene.bounds_max + r.scene.bounds_min) * 0.5
		reach := 2 * (f64(linalg.length(centre - [3]f32{0, 0, 2})) + f64(linalg.length(extent)) * 0.5)
		inv := linalg.inverse(camera_view_proj(camera_frame(cam)))
		for y in y0 ..< y0 + 4 {
			for x in x0 ..< x0 + 4 {
				ndc := [4]f32{(f32(x) + 0.5) / W * 2 - 1, 1 - (f32(y) + 0.5) / H * 2, 0.5, 1}
				h := inv * ndc
				dir := linalg.normalize([3]f64{f64(h.x / h.w), f64(h.y / h.w), f64(h.z / h.w) - 2})
				STEPS :: 4000
				dt := reach / STEPS
				sum: f64
				for k in 0 ..< STEPS {
					tt := (f64(k) + 0.5) * dt
					p := [3]f64{0, 0, 2} + dir * tt
					// The panel's solid angle from p, from a 48 x 48 grid on it.
					G :: 48
					omega: f64
					for j in 0 ..< G {
						for i in 0 ..< G {
							q := [3]f64{-0.5 + (f64(i) + 0.5) / G, -0.5 + (f64(j) + 0.5) / G, -2}
							d := q - p
							d2 := linalg.dot(d, d)
							omega += abs(d.z) / (d2 * math.sqrt(d2)) / (G * G)
						}
					}
					// Dimmed on the way from the light, up to the escape
					// length: the falloff (huge here) capped at the scene, 8.
					dimmed := math.exp(-sigma * min(linalg.length([3]f64{0, 0, -2} - p), 8))
					sum += math.exp(-sigma * tt) * sigma * dt * 4 * omega * dimmed / (4 * math.PI)
				}
				want += sum / 16
			}
		}
		return
	}
	for b in ([][3]int{{W / 2 - 2, H / 2 - 2, 0}, {W / 4 - 2, H / 2 - 2, 1}}) {
		got, want := block(raw[:], cam, &r, b[0], b[1])
		place := b[2] == 0 ? "through" : "beside"
		testing.expectf(t, abs(got - want) < want * 0.05, "fog %s the panel scatters %v, the integral says %v", place, got, want)
	}
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

	centre_rgb :: proc(t: ^testing.T, r: ^Renderer, cam: lc.Camera) -> [3]f32 {
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
		px := pixels[(H / 2) * W + W / 2]
		return {f16_to_f32(px[0]), f16_to_f32(px[1]), f16_to_f32(px[2])}
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

	// The same through fog of one density everywhere (a falloff far above
	// the scene), where every term has a closed form. sigma = 1/8, 10 to the
	// wall; light leaves the fog over the scene's size, 8, so the sky arrives
	// at each step dimmed by exp(-1).
	r.settings.fog_height = 1000
	t_u := math.exp(f32(-0.125 * 10))
	x := math.exp(f32(-1))
	//   - with multiple scattering, the haze is exactly the sky again, as it
	//     must be at albedo 1 under a uniform sky;
	uniform := centre_rgb(t, &r, cam)
	want_u := 4 * t_u + (1 - t_u)
	testing.expectf(t, abs(uniform.x - want_u) < 0.03, "uniform fog with multiple scattering: %v, want %v", uniform.x, want_u)
	//   - without, it is the dimmed sky, and darker;
	r.settings.fog_multiple = false
	single := centre_rgb(t, &r, cam)
	want_s := 4 * t_u + (1 - t_u) * x
	testing.expectf(t, abs(single.x - want_s) < 0.03, "uniform fog, single scattering: %v, want %v", single.x, want_s)
	r.settings.fog_multiple = true
	//   - coloured fog absorbs what it does not scatter: albedo 1 in red as
	//     white, 0 in blue leaves only the wall's 1 dimmed, and 0.5 in green
	//     gives a (1 - T) x / (1 - a (1 - x)) over the wall's 2 dimmed.
	r.settings.fog_colour = {1, 0.5, 0}
	tinted := centre_rgb(t, &r, cam)
	want_g := 2 * t_u + 0.5 * (1 - t_u) * x / (1 - 0.5 * (1 - x))
	testing.expectf(t, abs(tinted.x - uniform.x) < 0.01, "red through tinted fog is %v, want %v as in white", tinted.x, uniform.x)
	testing.expectf(t, abs(tinted.y - want_g) < 0.03, "green through tinted fog is %v, want %v", tinted.y, want_g)
	testing.expectf(t, abs(tinted.z - t_u) < 0.03, "blue through tinted fog is %v, want the wall's 1 dimmed to %v", tinted.z, t_u)
	r.settings.fog_colour = {1, 1, 1}

	// No density is no fog, not a pass that happens to do nothing.
	r.settings.fog_density = 0
	testing.expect_value(t, centre_red(t, &r, cam), clear)
}

}
