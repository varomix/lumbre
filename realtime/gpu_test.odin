package lumbre_realtime

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
	// A changed transform rebuilds geometry, but not the environment.
	scene.nodes[0].local_transform[0, 3] += 1
	testing.expect(t, renderer_set_scene(&r, &scene, 4))
	testing.expect_value(t, r.geometry_uploads, 2)
	testing.expect_value(t, r.environment_builds, 1)
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
	testing.expect_value(t, r.geometry_uploads, 2)
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

}
