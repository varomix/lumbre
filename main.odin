package main

import "core:fmt"
import "core:os"
import "core:strings"
import imp "./importers"
import rt "./realtime"

USE_GPU :: true

apply_quality_preset :: proc(cfg: ^Render_Config, preset: string) -> bool {
	switch preset {
	case "draft":
		cfg.samples_per_pixel = 16
		cfg.max_depth = 8
		cfg.photon_count = 100000
		cfg.photon_bounces = 4
		cfg.gi_cache_distance = 0.0
		cfg.photon_radius = 0.0
		return true
	case "preview":
		cfg.samples_per_pixel = 50
		cfg.max_depth = 12
		cfg.photon_count = 250000
		cfg.photon_bounces = 6
		cfg.gi_cache_distance = 0.0
		cfg.photon_radius = 0.0
		return true
	case "final":
		cfg.samples_per_pixel = 200
		cfg.max_depth = 20
		cfg.photon_count = 500000
		cfg.photon_bounces = 8
		cfg.gi_cache_distance = 0.0
		cfg.photon_radius = 0.0
		return true
	}
	return false
}

print_help :: proc() {
	fmt.println("Usage: lumbre [options]")
	fmt.println("  --test                     Render the built-in random-spheres test scene")
	fmt.println("  --scene, -s <file>         Load scene (.obj/.gltf/.glb/.usd/.usda/.usdc/.usdz)")
	fmt.println("  --width, -w <int>          Image width (default 1024)")
	fmt.println("  --height, -h <int>         Image height (default 576)")
	fmt.println("  --spp <int>                Samples per pixel (default 50)")
	fmt.println("  --depth <int>              Max bounces (default 20)")
	fmt.println("  --max-radiance <float>     Firefly clamp (default 1000)")
	fmt.println("  --roughness-cutoff <float> Bias: treat rough Principled as diffuse (default 0.95)")
	fmt.println("  --glossy-bias <float>      Bias: damp Principled roughness toward mirror (default 0)")
	fmt.println("  --output, -o <file.png>    Output file (default render.png)")
	fmt.println("  --cpu                      Force CPU renderer")
	fmt.println("  --gpu                      Force GPU renderer")
	fmt.println("  --quality <preset>         Quality preset: draft, preview, final")
	fmt.println("  --gi-cache <0|1>           Irradiance cache on/off (default 1)")
	fmt.println("  --gi-dist <float>          Cache lookup distance (default auto; >0 overrides)")
	fmt.println("  --gi-angle <float>         Cache normal angle threshold (default 0.5)")
	fmt.println("  --photon-map <0|1>         Photon mapping on/off (default 1)")
	fmt.println("  --photon-count <int>       Photon count (default 200000)")
	fmt.println("  --photon-radius <float>    Photon search radius (default auto; >0 overrides)")
	fmt.println("  --photon-bounces <int>     Max photon bounces (default 8)")
	fmt.println("  --deterministic            Reproducible output: disables the irradiance cache and")
	fmt.println("                              photon map, which are the only sources of run-to-run")
	fmt.println("                              variation. Costs bounce-light quality.")
	fmt.println("  --raster                   Render with the realtime rasterizer instead of the path")
	fmt.println("                              tracer: one frame per stage camera, no window.")
	fmt.println("  --labels                   With --raster, also write the ground-truth label EXR")
	fmt.println("                              (instance, semantic, depth, normal) and a COCO file")
	fmt.println("  --raster-view <n>          Rasterizer channel: 0=shaded (default), 1=albedo,")
	fmt.println("                              2=normal, 3=roughness, 4=metallic, 5=emission,")
	fmt.println("                              6=depth, 7=instance, 8=semantic")
	fmt.println("  --exposure <stops>         With --raster, exposure before the view transform (default 0)")
	fmt.println("  --view-transform <name>    With --raster: standard (default, clamp + sRGB), neutral, agx")
	fmt.println("  --fxaa <0|1>               With --raster, FXAA on the shaded view (default 1)")
	fmt.println("  --ao <0|1>                 With --raster, screen-space ambient occlusion (default 1)")
	fmt.println("  --ao-radius <scale>        With --raster, multiplies the automatic AO radius (default 1)")
	fmt.println("  --aovs                     Write AOV layers (albedo, normal, depth, direct, indirect) to .exr output")
	fmt.println("  --denoise [0|1]            OpenImageDenoise HDR ray-tracing denoiser (default off)")
	fmt.println("                              Set LUMBRE_OIDN_LIBRARY to an OIDN dylib path if needed")
	fmt.println("  --hdri <file.hdr>          Equirectangular HDRI environment (dome light)")
	fmt.println("  --hdri-rotation <deg>      Rotate the HDRI about the Y axis (default 0)")
	fmt.println("  --hdri-intensity <float>   Scale the HDRI radiance (default 1)")
	fmt.println("  --usd-camera <prim-name>   Use this Camera prim from a USD scene (default: first found)")
	fmt.println("  --subdiv-level <n>         Subdivision level for USD subdiv meshes (default 2, 0 = off)")
	fmt.println("  --sun-dir <x,y,z>          Add a directional/sun light (direction it travels)")
	fmt.println("  --sun-intensity <f|r,g,b>  Sun radiance, scalar or RGB (default 1)")
	fmt.println("  --sun-angle <deg>          Sun angular diameter for soft shadows (default 0.53)")
	fmt.println("  --zip                      Enable ZIP compression for .exr output (default: uncompressed)")
	fmt.println("  --frame-range N            Render a single frame N")
	fmt.println("  --frame-range N-M          Render a sequence of frames N..M (inclusive)")
	fmt.println("  --debug <mode>             Debug: 1=albedo, 2=normal, 3=depth, 4=primitive id, 5=direct, 6=light count, 7=direct candidates, 8=shadow visibility, 9=indirect, 10=GI cache hits, 11=photon contribution, 12=GI cache samples, 13=GI cache confidence, 14=UV, 15=albedo texture, 16=roughness/metallic")
	fmt.println("  --script <file.py> [-- args]  Run a Python script with no window. It authors")
	fmt.println("                              stages with pxr and renders them through `lumbre`;")
	fmt.println("                              anything after `--` is passed to it as sys.argv")
	fmt.println("  --help                     Show this help")
}

main :: proc() {
	// Default config
	cfg := Render_Config{
		image_width       = 1024,
		image_height      = 576,
		samples_per_pixel = 50,
		max_depth         = 20,
		max_radiance      = 1000.0,
		roughness_cutoff  = 0.95,
		glossy_bias       = 0.0,
		file_output       = "render.png",
		use_gpu           = USE_GPU,
		gi_cache_enabled  = true,
		gi_cache_distance = 0.0,
		gi_cache_normal_angle = 0.5,
		photon_enabled    = true,
		photon_count      = 200000,
		photon_radius     = 0.0,
		photon_bounces    = 8,
		enable_aovs       = false,
		exr_compress      = false,
		denoise_enabled    = false,
		frame_start       = 0,
		frame_end         = 0,
		frame_padding     = 4,
		hdri_file         = "",
		hdri_rotation     = 0.0,
		hdri_intensity    = 1.0,
		sun_enabled       = false,
		sun_dir           = Vec3{-1.0, -1.0, -1.0},
		sun_color         = Color{1.0, 1.0, 1.0},
		sun_angle         = 0.53, // ~ the sun's angular diameter in degrees
		usd_subdiv_level  = 2,
	}

	// Simple CLI arg parsing
	raster := Raster_Options{settings = rt.DEFAULT_RENDER_SETTINGS}
	use_raster := false
	run_test := false
	script_path := ""
	script_args: []string
	args := os.args[1:]
	for i := 0; i < len(args); i += 1 {
		arg := args[i]
		switch arg {
		case "--test":
			run_test = true
		case "--script":
			if i + 1 < len(args) {
				script_path = args[i + 1]
				i += 1
			}
		case "--":
			// Everything after belongs to the script. A bare separator keeps
			// a script's own `--width` from being read as Lumbre's.
			script_args = args[i + 1:]
			i = len(args)
		case "--bsdf-energy-test":
			// Diagnostic: Monte-Carlo integrate the CPU Principled BSDF's
			// directional reflectance to check energy conservation (no gain,
			// no NaN). See bsdf_energy_test.odin / plans/PRINCIPLED_BSDF.md.
			run_bsdf_energy_test()
			return
		case "--scene", "-s":
			if i + 1 < len(args) {
				cfg.scene_file = cstring(strings.clone_to_cstring(args[i + 1]))
				i += 1
			}
		case "--width", "-w":
			if i + 1 < len(args) {
				cfg.image_width = i32(parse_int(args[i + 1]))
				i += 1
			}
		case "--height", "-h":
			if i + 1 < len(args) {
				cfg.image_height = i32(parse_int(args[i + 1]))
				i += 1
			}
		case "--spp":
			if i + 1 < len(args) {
				cfg.samples_per_pixel = i32(parse_int(args[i + 1]))
				i += 1
			}
		case "--depth":
			if i + 1 < len(args) {
				cfg.max_depth = i32(parse_int(args[i + 1]))
				i += 1
			}
		case "--max-radiance":
			if i + 1 < len(args) {
				cfg.max_radiance = parse_float(string(args[i + 1]))
				i += 1
			}
		case "--roughness-cutoff":
			if i + 1 < len(args) {
				cfg.roughness_cutoff = parse_float(string(args[i + 1]))
				i += 1
			}
		case "--glossy-bias":
			if i + 1 < len(args) {
				cfg.glossy_bias = parse_float(string(args[i + 1]))
				i += 1
			}
		case "--output", "-o":
			if i + 1 < len(args) {
				cfg.file_output = cstring(strings.clone_to_cstring(args[i + 1]))
				i += 1
			}
		case "--cpu":
			cfg.use_gpu = false
		case "--gpu":
			cfg.use_gpu = true
		case "--quality":
			if i + 1 < len(args) {
				if !apply_quality_preset(&cfg, string(args[i + 1])) {
					fmt.eprintln("Unknown quality preset:", args[i + 1])
					fmt.eprintln("Expected one of: draft, preview, final")
					return
				}
				i += 1
			}
		case "--gi-cache":
			if i + 1 < len(args) {
				cfg.gi_cache_enabled = args[i + 1] == "1" || args[i + 1] == "true"
				i += 1
			}
		case "--gi-dist":
			if i + 1 < len(args) {
				cfg.gi_cache_distance = f32(parse_float(args[i + 1]))
				i += 1
			}
		case "--gi-angle":
			if i + 1 < len(args) {
				cfg.gi_cache_normal_angle = f32(parse_float(args[i + 1]))
				i += 1
			}
		case "--photon-map":
			if i + 1 < len(args) {
				cfg.photon_enabled = args[i + 1] == "1" || args[i + 1] == "true"
				i += 1
			}
		case "--photon-count":
			if i + 1 < len(args) {
				cfg.photon_count = i32(parse_int(args[i + 1]))
				i += 1
			}
		case "--photon-radius":
			if i + 1 < len(args) {
				cfg.photon_radius = f32(parse_float(args[i + 1]))
				i += 1
			}
		case "--photon-bounces":
			if i + 1 < len(args) {
				cfg.photon_bounces = i32(parse_int(args[i + 1]))
				i += 1
			}
		case "--deterministic":
			// The path tracer's sampling is already reproducible: the seed is a
			// literal and nothing time-based reaches the GPU. What drifts is the
			// two biased-GI caches, both of which accumulate through atomics in
			// thread-arrival order -- the irradiance cache is even read in the
			// same dispatch that fills it, so a pixel sees a race-ordered cache.
			//
			// Measured on the cornell box: with these two off, two runs are
			// byte-identical; with them on, they are not. Every debug AOV except
			// mode 9 (indirect) is bit-exact either way, because the others
			// evaluate at the first hit and never reach the caches.
			//
			// Making the caches themselves deterministic means building them in
			// a separate dispatch from the one that reads them, and ordering
			// each bucket. That is a real change to the renderer, not a flag.
			cfg.gi_cache_enabled = false
			cfg.photon_enabled = false
		case "--aovs":
			cfg.enable_aovs = true
		case "--denoise":
			// Optional argument: only consume the next token when it is an
			// explicit boolean value. Otherwise `--denoise` is a bare switch and
			// must not swallow a following flag (e.g. `--denoise -o out.png`).
			if i + 1 < len(args) && is_bool_arg(args[i + 1]) {
				cfg.denoise_enabled = args[i + 1] == "1" || args[i + 1] == "true"
				i += 1
			} else {
				cfg.denoise_enabled = true
			}
		case "--hdri":
			if i + 1 < len(args) {
				cfg.hdri_file = cstring(strings.clone_to_cstring(args[i + 1]))
				i += 1
			}
		case "--hdri-rotation":
			if i + 1 < len(args) {
				cfg.hdri_rotation = parse_float(args[i + 1])
				i += 1
			}
		case "--hdri-intensity":
			if i + 1 < len(args) {
				cfg.hdri_intensity = parse_float(args[i + 1])
				i += 1
			}
		case "--usd-camera":
			if i + 1 < len(args) {
				cfg.usd_camera_name = cstring(strings.clone_to_cstring(args[i + 1]))
				i += 1
			}
		case "--subdiv-level":
			if i + 1 < len(args) {
				cfg.usd_subdiv_level = i32(parse_int(args[i + 1]))
				i += 1
			}
		case "--force-aniso":
			if i + 1 < len(args) {
				cfg.force_anisotropic = parse_float(args[i + 1])
				i += 1
			}
		case "--force-glass":
			if i + 1 < len(args) {
				cfg.force_spec_trans = parse_float(args[i + 1])
				i += 1
			}
		case "--sun-dir":
			if i + 1 < len(args) {
				cfg.sun_dir = parse_vec3(args[i + 1])
				cfg.sun_enabled = true
				i += 1
			}
		case "--sun-intensity":
			if i + 1 < len(args) {
				cfg.sun_color = parse_color(args[i + 1])
				cfg.sun_enabled = true
				i += 1
			}
		case "--sun-angle":
			if i + 1 < len(args) {
				cfg.sun_angle = parse_float(args[i + 1])
				i += 1
			}
		case "--zip":
			cfg.exr_compress = true
		case "--frame-range":
			if i + 1 < len(args) {
				// Expect either "N" or "N-M".
				spec := args[i + 1]
				idx := 0
				// Parse the first integer.
				for idx < len(spec) && spec[idx] >= '0' && spec[idx] <= '9' {
					idx += 1
				}
				start := parse_int(spec[:idx])
				end := start
				if idx < len(spec) && spec[idx] == '-' {
					idx += 1
					end = parse_int(spec[idx:])
				}
				if end < start {
					fmt.eprintln("--frame-range: end < start")
					return
				}
				cfg.frame_start = i32(start)
				cfg.frame_end = i32(end)
				i += 1
			}
		case "--raster":
			use_raster = true
		case "--labels":
			raster.labels = true
			// Labels are only produced by the rasterizer's label pass, so
			// asking for them is asking for --raster. Requiring both would be
			// a spelling test, not a choice.
			use_raster = true
		case "--raster-view":
			if i + 1 < len(args) {
				raster.view = rt.Debug_View(parse_int(args[i + 1]))
				i += 1
			}
		case "--exposure":
			if i + 1 < len(args) {
				raster.settings.exposure = f32(parse_float(args[i + 1]))
				i += 1
			}
		case "--fxaa":
			if i + 1 < len(args) {
				raster.settings.fxaa = parse_int(args[i + 1]) != 0
				i += 1
			}
		case "--ao":
			if i + 1 < len(args) {
				raster.settings.ao = parse_int(args[i + 1]) != 0
				i += 1
			}
		case "--ao-radius":
			if i + 1 < len(args) {
				raster.settings.ao_radius = f32(parse_float(args[i + 1]))
				i += 1
			}
		case "--view-transform":
			if i + 1 < len(args) {
				switch args[i + 1] {
				case "standard": raster.settings.view_transform = .Standard
				case "neutral":  raster.settings.view_transform = .Neutral
				case "agx":      raster.settings.view_transform = .AgX
				case:
					fmt.eprintln("Unknown view transform:", args[i + 1], "(standard, neutral, agx)")
					os.exit(1)
				}
				i += 1
			}
		case "--help":
			print_help()
			return
		case:
			// An unrecognized switch used to be ignored in silence, which
			// makes an old binary indistinguishable from a broken feature:
			// `--raster` on a build that predates it renders the path-traced
			// image and says nothing. Bare words are still allowed through,
			// since they are the values of the switches above.
			if strings.has_prefix(arg, "-") {
				fmt.eprintln("Unknown option:", arg)
				fmt.eprintln("Run `lumbre --help` for the full list.")
				os.exit(2)
			}
		case "--debug":
			if i + 1 < len(args) {
				cfg.debug_mode = i32(parse_int(args[i + 1]))
				i += 1
			}
		}
	}

	// A script brings its own stages, so it needs no --scene.
	if script_path != "" {
		raster.exr_compress = bool(cfg.exr_compress)
		if !run_script(script_path, script_args, cfg, raster) {
			os.exit(1)
		}
		return
	}

	// With no scene to render, print help instead of silently launching the
	// heavy procedural sphere scene (which looks like a hang). Require an
	// explicit --scene or --test.
	if cfg.scene_file == "" && !run_test {
		print_help()
		return
	}

	if cfg.scene_file != "" {
		fmt.println("Loading scene:", cfg.scene_file)
	}

	scene, ok := imp.make_scene(cfg)
	if !ok {
		fmt.eprintln("Failed to create scene")
		return
	}
	defer destroy_scene(&scene)

	fmt.println("Spheres:", len(scene.spheres))
	tri_count := 0
	for mesh in scene.meshes {
		tri_count += len(mesh.triangles)
	}
	fmt.println("Triangles:", tri_count)
	fmt.println("Resolution:", cfg.image_width, "x", cfg.image_height)
	fmt.println("Samples:", cfg.samples_per_pixel)
	fmt.println("Max depth:", cfg.max_depth)

	if use_raster {
		raster.exr_compress = bool(cfg.exr_compress)
		if !run_raster_batch(&scene, cfg, raster) {
			os.exit(1)
		}
		return
	}

	// Determine the frame range. A single frame is frame_start ==
	// frame_end (or frame_end <= 0). For a sequence, both must be
	// positive and frame_end >= frame_start.
	frame_start := cfg.frame_start
	frame_end := cfg.frame_end
	if frame_end < frame_start {
		frame_end = frame_start
	}
	if frame_start < 0 {
		frame_start = 0
		frame_end = 0
	}

	// Compute padding width from the frame range.
	// We use enough digits to cover the largest frame number.
	max_frame_num := frame_end
	if max_frame_num == 0 { max_frame_num = frame_start }
	padding := max(cfg.frame_padding, 1)
	for n := max_frame_num; n >= 10; n /= 10 { padding += 1 }

	total_frames := frame_end - frame_start + 1
	if total_frames < 1 { total_frames = 1 }

	for fi in 0 ..< total_frames {
		frame_num := frame_start + i32(fi)

		// Per-frame output path. The frame number is inserted
		// before the extension: "render.exr" -> "render.0001.exr".
		// If the user didn't pass --output, we keep the same
		// path for every frame.
		frame_output: cstring = cfg.file_output
		// Owns the buffer for the per-frame cstring when frame-range
		// is set. We always number the output when --frame-range is
		// provided, even if it's a single frame, so the file name
		// clearly indicates the frame number.
		frame_output_buf: [dynamic]u8
		if cfg.frame_start > 0 || cfg.frame_end > 0 {
			base_path := string(cfg.file_output)
			// Find the last '.' in the path to locate the extension.
			dot_idx := -1
			for ci := len(base_path) - 1; ci >= 0; ci -= 1 {
				if base_path[ci] == '.' {
					dot_idx = ci
					break
				}
			}
			stem, ext := base_path, ""
			if dot_idx > 0 {
				stem = base_path[:dot_idx]
				ext = base_path[dot_idx:]
			}
			// Build "<stem>.<frame>.<ext>".
			append(&frame_output_buf, ..transmute([]u8)stem)
			append(&frame_output_buf, '.')
			// Format the frame number with zero-padding.
			digits_buf: [16]u8
			di := len(digits_buf)
			n := frame_num
			if n == 0 {
				di -= 1
				digits_buf[di] = '0'
			} else {
				for n > 0 {
					di -= 1
					digits_buf[di] = u8('0' + n % 10)
					n /= 10
				}
			}
			// Pad to `padding` digits.
			for di > 0 && i32(len(digits_buf) - di) < i32(padding) {
				di -= 1
				digits_buf[di] = '0'
			}
			append(&frame_output_buf, ..digits_buf[di:])
			if ext != "" {
				append(&frame_output_buf, ..transmute([]u8)ext)
			}
			append(&frame_output_buf, 0) // null terminator
			frame_output = cstring(raw_data(frame_output_buf[:]))
		}

		if cfg.frame_start > 0 || cfg.frame_end > 0 {
			fmt.println()
			fmt.println("=== Frame", frame_num, "of", frame_end, "===")
		}

		core := lumbre_core_init(scene, cfg)
		lumbre_core_render_to_file(&core, frame_output)
	}
}

parse_int :: proc(s: string) -> int {
	result: int
	negative := false
	start := 0
	if len(s) > 0 && s[0] == '-' {
		negative = true
		start = 1
	}
	for i := start; i < len(s); i += 1 {
		if s[i] >= '0' && s[i] <= '9' {
			result = result * 10 + int(s[i] - '0')
		}
	}
	if negative {
		result = -result
	}
	return result
}

// True when `s` is an explicit boolean value accepted by optional-argument
// switches like `--denoise`.
is_bool_arg :: proc(s: string) -> bool {
	return s == "0" || s == "1" || s == "true" || s == "false"
}

// Parse "x,y,z" into a Vec3. Missing components default to the previous one
// (so a single scalar fills all three).
parse_vec3 :: proc(s: string) -> Vec3 {
	parts := strings.split(s, ",")
	defer delete(parts)
	v := Vec3{0, 0, 0}
	last := 0.0
	for i in 0 ..< 3 {
		if i < len(parts) {
			last = parse_float(strings.trim_space(parts[i]))
		}
		v[i] = last
	}
	return v
}

parse_color :: proc(s: string) -> Color {
	return Color(parse_vec3(s))
}

parse_float :: proc(s: string) -> f64 {
	result: f64
	frac: f64
	divisor: f64 = 1.0
	decimal := false
	negative := false
	start := 0
	if len(s) > 0 && s[0] == '-' {
		negative = true
		start = 1
	}
	for i := start; i < len(s); i += 1 {
		if s[i] == '.' {
			decimal = true
		} else if s[i] >= '0' && s[i] <= '9' {
			if !decimal {
				result = result * 10.0 + f64(s[i] - '0')
			} else {
				divisor *= 10.0
				frac = frac * 10.0 + f64(s[i] - '0')
			}
		}
	}
	result += frac / divisor
	if negative {
		result = -result
	}
	return result
}
