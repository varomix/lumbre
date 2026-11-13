package lumbre_realtime

import "core:math"
import "core:os"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

when #config(GPU_TESTS, false) {

// A floor of albedo 0.5 under a quad, a disc and a sphere light, each straight
// overhead: the floor below the light reflects albedo * L * F, where F is the
// shape's form factor from that point, known in closed form. The old
// single-point approximation overshot the quad by about 58% and the disc by
// 44% here; for a sphere wholly above the horizon it was already exact, which
// the polygon standing in for the sphere must keep.
@(test)
test_gpu_area_lights_match_their_form_factors :: proc(t: ^testing.T) {
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
	// No specular reflectance, so F0 is zero and only the diffuse lobe -- the
	// form factor -- is measured, plus the Fresnel rise, small looking down.
	materials[0] = lc.Material{kind = .Principled, albedo = {0.5, 0.5, 0.5}, roughness = 1, specular = 0}
	scene := lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
	defer destroy_test_scene(&scene)
	black := []f32{0, 0, 0}
	scene.environment = {width = 1, height = 1, has_data = true, pixels = black, intensity = 1}

	W :: 33
	H :: 33
	// Straight down at the point under the light; lights are not geometry,
	// so the camera sees through them.
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

	Case :: struct {
		name:  string,
		light: lc.Light,
		want:  f32, // albedo * L * form factor
	}
	// Square, half-size 1, 1.5 overhead.
	x: f32 = 1.0 / 1.5
	square := (2 / math.PI) * 2 * (x / math.sqrt(1 + x * x)) * math.atan(x / math.sqrt(1 + x * x))
	cases := []Case {
		{"quad", lc.make_area_light({-1, 1.5, -1}, {2, 0, 0}, {0, 0, 2}, {2, 2, 2}), 0.5 * 2 * square},
		// Disc, radius 1, 1.5 overhead: R^2 / (h^2 + R^2).
		{"disc", lc.make_disc_light({0, 1.5, 0}, {0, -1, 0}, 1, {2, 2, 2}), 0.5 * 2 * (1.0 / (2.25 + 1))},
		// Sphere, radius 0.5, centre 2 overhead: (r / d)^2.
		{"sphere", lc.make_sphere_light({0, 2, 0}, 0.5, {4, 4, 4}), 0.5 * 4 * (0.25 / 4)},
	}
	for c, i in cases {
		lights := []lc.Light{c.light}
		scene.lights = lights
		testing.expect(t, renderer_set_scene(&r, &scene, u64(i + 1)))
		got := centre_red(t, &r, cam)
		testing.expectf(t, abs(got - c.want) < c.want * 0.03, "%s light: floor is %v, want %v", c.name, got, c.want)
	}
}

}
