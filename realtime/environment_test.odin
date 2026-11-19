package lumbre_realtime

import "core:math"
import "core:os"
import "core:strings"
import "core:testing"
import lc "../core"
import sdl "vendor:sdl3"

when #config(GPU_TESTS, false) {

// The split-sum maps must describe the same GGX as the analytic BRDF and the
// path tracer: alpha = roughness, exact uncorrelated Smith. Each check below
// is a CPU integral over the path tracer's own `ggx_D` and `ggx_G1`. With
// alpha = roughness^2 (the old mapping) the LUT's scale is off by up to 0.2
// and the prefiltered horizon by up to 0.14; with the Schlick-k geometry term
// the scale is off by up to 0.15.

@(private = "file")
env_test_renderer :: proc(t: ^testing.T, gpu: ^sdl.GPUDevice, scene: ^lc.Scene) -> (r: Renderer) {
	ok: bool
	r, ok = renderer_create(gpu)
	if !ok { testing.fail_now(t, "renderer creation") }
	if !renderer_set_scene(&r, scene, 1) { testing.fail_now(t, "set scene") }
	return
}

@(private = "file")
env_test_device :: proc(t: ^testing.T) -> ^sdl.GPUDevice {
	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	gpu := sdl.CreateGPUDevice({.MSL, .SPIRV}, true, preferred)
	if gpu == nil { testing.fail_now(t, "requested GPU backend unavailable") }
	return gpu
}

@(private = "file")
env_test_scene :: proc(env: lc.Environment) -> lc.Scene {
	tris := make([]lc.Triangle, 1)
	n := lc.Vec3{0, 1, 0}
	tris[0] = lc.Triangle{v0 = {-1, 0, 1}, v1 = {1, 0, 1}, v2 = {1, 0, -1}, n0 = n, n1 = n, n2 = n}
	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = 1}
	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.make_node(1, 0, -1, -1)
	materials := make([]lc.Material, 1)
	materials[0] = lc.Material{kind = .Principled, albedo = {0.5, 0.5, 0.5}, roughness = 0.5}
	return lc.Scene{meshes = meshes, nodes = nodes, materials = materials, environment = env}
}

@(private = "file")
download_f16 :: proc(t: ^testing.T, gpu: ^sdl.GPUDevice, tex: ^sdl.GPUTexture, w, h: i32, channels: int, mip: u32 = 0) -> []f32 {
	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	transfer, ok := download_begin(gpu, cmd, tex, w, h, u32(channels * 2), mip)
	if !ok { testing.fail_now(t, "download") }
	defer sdl.ReleaseGPUTransferBuffer(gpu, transfer)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	defer sdl.ReleaseGPUFence(gpu, fence)
	testing.expect(t, bool(sdl.WaitForGPUFences(gpu, true, &fence, 1)))
	raw := make([]u16, int(w * h) * channels, context.temp_allocator)
	testing.expect(t, map_copy(gpu, transfer, raw))
	out := make([]f32, len(raw), context.temp_allocator)
	for v, i in raw { out[i] = f16_to_f32(v) }
	return out
}

// Scale and bias of the environment BRDF at (n_dot_v, roughness), by
// integrating the path tracer's GGX over the hemisphere of light directions.
@(private = "file")
reference_env_brdf :: proc(n_dot_v, roughness: f64) -> (scale, bias: f64) {
	alpha_sq := lc.ggx_alpha_sq(roughness)
	v := [3]f64{math.sqrt(1 - n_dot_v * n_dot_v), 0, n_dot_v}
	THETA :: 1024
	PHI :: 512
	for i in 0 ..< THETA {
		theta := (f64(i) + 0.5) / THETA * math.PI / 2
		for j in 0 ..< PHI {
			phi := (f64(j) + 0.5) / PHI * 2 * math.PI
			l := [3]f64{math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)}
			hv := v + l
			h := hv / math.sqrt(hv.x * hv.x + hv.y * hv.y + hv.z * hv.z)
			v_dot_h := max(v.x * h.x + v.y * h.y + v.z * h.z, 0)
			f := lc.ggx_D(h.z, alpha_sq) * lc.ggx_G1(n_dot_v, alpha_sq) * lc.ggx_G1(l.z, alpha_sq) / (4 * n_dot_v)
			d_omega := math.sin(theta) * (math.PI / 2 / THETA) * (2 * math.PI / PHI)
			fc := math.pow(1 - v_dot_h, 5)
			scale += (1 - fc) * f * d_omega
			bias += fc * f * d_omega
		}
	}
	return
}

@(test)
test_gpu_brdf_lut_matches_the_path_tracers_ggx :: proc(t: ^testing.T) {
	gpu := env_test_device(t)
	defer sdl.DestroyGPUDevice(gpu)
	scene := env_test_scene({})
	defer destroy_test_scene(&scene)
	r := env_test_renderer(t, gpu, &scene)
	defer renderer_destroy(&r)

	lut := download_f16(t, gpu, r.brdf_lut, BRDF_LUT_SIZE, BRDF_LUT_SIZE, 2)
	// Texel (x, y) holds n_dot_v = (x + 0.5) / SIZE, roughness = (y + 0.5) / SIZE.
	for c in ([][2]int{{63, 38}, {63, 63}, {95, 95}, {31, 63}}) {
		n_dot_v := (f64(c[0]) + 0.5) / BRDF_LUT_SIZE
		roughness := (f64(c[1]) + 0.5) / BRDF_LUT_SIZE
		want_scale, want_bias := reference_env_brdf(n_dot_v, roughness)
		i := (c[1] * BRDF_LUT_SIZE + c[0]) * 2
		got_scale, got_bias := f64(lut[i]), f64(lut[i + 1])
		testing.expectf(
			t, abs(got_scale - want_scale) < 0.01 && abs(got_bias - want_bias) < 0.005,
			"LUT at n.v %.3f roughness %.3f is (%.4f, %.4f), want (%.4f, %.4f)",
			n_dot_v, roughness, got_scale, got_bias, want_scale, want_bias,
		)
	}
}

// A sky that is 1 above the horizon and 0 below. The prefiltered value for a
// direction just above the horizon is the share of the GGX lobe, as the
// prefilter weights it (half-vectors drawn from D(h) cos(h), reflected about
// v = n, weighted by n.l), that lands in the sky. A lobe of the wrong width
// moves it a long way.
@(private = "file")
reference_horizon :: proc(elevation, roughness: f64) -> f64 {
	alpha_sq := lc.ggx_alpha_sq(roughness)
	// Frame: n at `elevation` above the horizon, in the x-y plane.
	n := [3]f64{math.cos(elevation), math.sin(elevation), 0}
	tng := [3]f64{-math.sin(elevation), math.cos(elevation), 0}
	btg := [3]f64{0, 0, 1}
	num, den: f64
	THETA :: 1024
	PHI :: 512
	for i in 0 ..< THETA {
		th := (f64(i) + 0.5) / THETA * math.PI / 2
		for j in 0 ..< PHI {
			ph := (f64(j) + 0.5) / PHI * 2 * math.PI
			hl := [3]f64{math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)}
			h := tng * hl.x + btg * hl.y + n * hl.z
			// l = 2 (n.h) h - n, with v = n.
			l := 2 * hl.z * h - n
			n_dot_l := l.x * n.x + l.y * n.y + l.z * n.z
			if n_dot_l <= 0 { continue }
			w := lc.ggx_D(hl.z, alpha_sq) * hl.z * math.sin(th) * n_dot_l
			den += w
			if l.y > 0 { num += w }
		}
	}
	return num / den
}

@(test)
test_gpu_prefiltered_environment_has_the_path_tracers_lobe :: proc(t: ^testing.T) {
	gpu := env_test_device(t)
	defer sdl.DestroyGPUDevice(gpu)

	EW :: 512
	EH :: 256
	pixels := make([]f32, EW * EH * 3)
	defer delete(pixels)
	for y in 0 ..< EH {
		value: f32 = y < EH / 2 ? 1 : 0 // row 0 is the zenith
		for x in 0 ..< EW {
			for c in 0 ..< 3 { pixels[(y * EW + x) * 3 + c] = value }
		}
	}
	scene := env_test_scene({width = EW, height = EH, pixels = pixels, has_data = true, intensity = 1})
	defer destroy_test_scene(&scene)
	r := env_test_renderer(t, gpu, &scene)
	defer renderer_destroy(&r)

	// Level k holds roughness k / (SPECULAR_MIPS - 1); levels 2 and 3 are 0.4
	// and 0.6. Read a row a little above the horizon, and one a little below.
	for level in ([]u32{2, 3}) {
		w := i32(SPECULAR_WIDTH >> level)
		h := i32(SPECULAR_HEIGHT >> level)
		mip := download_f16(t, gpu, r.env.specular, w, h, 4, level)
		roughness := f64(level) / f64(SPECULAR_MIPS - 1)
		for row in ([]i32{h / 2 - 3, h / 2 + 2}) {
			theta := (f64(row) + 0.5) / f64(h) * math.PI
			elevation := math.PI / 2 - theta
			want := reference_horizon(elevation, roughness)
			got: f64
			for x in 0 ..< w { got += f64(mip[(row * w + x) * 4]) }
			got /= f64(w)
			testing.expectf(
				t, abs(got - want) < 0.03,
				"roughness %.1f, %.1f degrees from the horizon: prefiltered %.4f, want %.4f",
				roughness, math.to_degrees(elevation), got, want,
			)
		}
	}
}

}
