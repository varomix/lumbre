package main

import "core:fmt"
import "core:os"
import "core:strings"

USE_GPU :: true

apply_quality_preset :: proc(cfg: ^Render_Config, preset: string) -> bool {
	switch preset {
	case "draft":
		cfg.samples_per_pixel = 16
		cfg.max_depth = 8
		cfg.photon_count = 262144
		cfg.photon_bounces = 4
		cfg.gi_cache_distance = 0.0
		cfg.photon_radius = 0.0
		return true
	case "preview":
		cfg.samples_per_pixel = 50
		cfg.max_depth = 12
		cfg.photon_count = 524288
		cfg.photon_bounces = 6
		cfg.gi_cache_distance = 0.0
		cfg.photon_radius = 0.0
		return true
	case "final":
		cfg.samples_per_pixel = 200
		cfg.max_depth = 20
		cfg.photon_count = 1048576
		cfg.photon_bounces = 8
		cfg.gi_cache_distance = 0.0
		cfg.photon_radius = 0.0
		return true
	}
	return false
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
		photon_count      = 1048576,
		photon_radius     = 0.0,
		photon_bounces    = 8,
	}

	// Simple CLI arg parsing
	args := os.args[1:]
	for i := 0; i < len(args); i += 1 {
		switch args[i] {
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
		case "--help":
			fmt.println("Usage: lumbre [options]")
			fmt.println("  --scene, -s <file.obj>     Load OBJ scene")
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
			fmt.println("  --photon-count <int>       Photon count (default 1048576)")
			fmt.println("  --photon-radius <float>    Photon search radius (default auto; >0 overrides)")
			fmt.println("  --photon-bounces <int>     Max photon bounces (default 8)")
			fmt.println("  --debug <mode>             Debug: 1=albedo, 2=normal, 3=depth, 4=primitive id, 5=direct, 6=light count, 7=direct candidates, 8=shadow visibility, 9=indirect, 10=GI cache hits, 11=photon contribution, 12=GI cache samples, 13=GI cache confidence")
			return
		case "--debug":
			if i + 1 < len(args) {
				cfg.debug_mode = i32(parse_int(args[i + 1]))
				i += 1
			}
		}
	}

	if cfg.scene_file != "" {
		fmt.println("Loading scene:", cfg.scene_file)
	}

	scene, ok := make_scene(cfg)
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

	when ODIN_OS == .Darwin {
		if cfg.use_gpu {
			render_gpu(&scene, cfg.image_width, cfg.image_height, cfg.samples_per_pixel, cfg.max_depth, cfg.max_radiance, cfg.file_output, cfg.debug_mode, cfg.roughness_cutoff, cfg.glossy_bias, cfg.gi_cache_enabled, cfg.gi_cache_distance, cfg.gi_cache_normal_angle, cfg.photon_enabled, cfg.photon_count, cfg.photon_radius, cfg.photon_bounces)
			return
		}
	}

	render_cpu(&scene, cfg.image_width, cfg.image_height, cfg.samples_per_pixel, cfg.max_depth, cfg.max_radiance, cfg.file_output, cfg.roughness_cutoff, cfg.glossy_bias)
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
