package lumbre_core

// In-memory result of a GPU render. `pixels` is the sRGB-encoded 8-bit beauty
// (RGB8) shared with the CPU path's `Render_Buffer`; `beauty_linear` and
// `aov_results` are kept for the CLI's EXR writer. The Houdini bridge only
// needs `pixels`.
GPU_Frame :: struct {
	width, height: i32,
	pixels:        []u8,                       // RGB8, sRGB-encoded beauty
	beauty_linear: [][4]f32,                   // linear RGBA beauty (EXR)
	aov_results:   map[int][dynamic][4]f32,    // linear AOV layers (EXR)
}

destroy_gpu_frame :: proc(f: ^GPU_Frame) {
	delete(f.pixels)
	delete(f.beauty_linear)
	for _, v in f.aov_results {
		delete(v)
	}
	delete(f.aov_results)
	f^ = {}
}

// GPU render straight to an in-memory beauty buffer, matching the CPU path's
// `render_cpu_to_buffer` shape. AOVs and EXR-only work are skipped; this is the
// interactive/viewport entry point.
render_gpu_to_buffer :: proc(
	scene: ^Scene,
	image_width, image_height: i32,
	samples_per_pixel, max_depth: i32,
	max_radiance: f64,
	debug_mode: i32 = 0,
	roughness_cutoff: f64 = 0.95,
	glossy_bias: f64 = 0.0,
	gi_cache_enabled: b32 = true,
	gi_cache_distance: f32 = 0.0,
	gi_cache_normal_angle: f32 = 0.5,
	photon_enabled: b32 = true,
	photon_count: i32 = 200000,
	photon_radius: f32 = 0.0,
	photon_bounces: i32 = 8,
	denoise_enabled: b32 = false,
) -> Render_Buffer {
	frame := gpu_render_frame(scene, image_width, image_height, samples_per_pixel, max_depth,
		max_radiance, debug_mode, roughness_cutoff, glossy_bias,
		gi_cache_enabled, gi_cache_distance, gi_cache_normal_angle,
		photon_enabled, photon_count, photon_radius, photon_bounces,
		false, false, denoise_enabled)
	defer destroy_gpu_frame(&frame)

	// The GPU beauty buffer is bottom-row-first (the CLI writer compensates with
	// stbi's flip-on-write). `render_cpu_to_buffer` is top-row-first, so flip the
	// GPU result to match: both `*_to_buffer` APIs return a top-first buffer, and
	// frontends (the Houdini bridge) can treat them identically.
	w := int(frame.width)
	h := int(frame.height)
	pixels := make([]u8, w * h * 3)
	for y in 0 ..< h {
		src := (h - 1 - y) * w * 3
		dst := y * w * 3
		copy(pixels[dst:dst + w * 3], frame.pixels[src:src + w * 3])
	}
	return Render_Buffer{width = frame.width, height = frame.height, pixels = pixels}
}

// Convenience wrapper that renders the core's scene with its settings on the
// GPU, mirroring `lumbre_core_render_cpu`.
lumbre_core_render_gpu :: proc(core: ^Lumbre_Core) -> Render_Buffer {
	s := core.settings
	return render_gpu_to_buffer(&core.scene, s.image_width, s.image_height,
		s.samples_per_pixel, s.max_depth, s.max_radiance, s.debug_mode,
		s.roughness_cutoff, s.glossy_bias, s.gi_cache_enabled, s.gi_cache_distance,
		s.gi_cache_normal_angle, s.photon_enabled, s.photon_count, s.photon_radius,
		s.photon_bounces, s.denoise_enabled)
}
