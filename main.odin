package main

import "core:fmt"
import "core:os"
import "core:strings"

USE_GPU :: true

main :: proc() {
	// Default config
	cfg := Render_Config{
		image_width       = 1024,
		image_height      = 576,
		samples_per_pixel = 50,
		max_depth         = 20,
		max_radiance      = 1000.0,
		roughness_cutoff  = 0.95,
		file_output       = "render.png",
		use_gpu           = USE_GPU,
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
		case "--output", "-o":
			if i + 1 < len(args) {
				cfg.file_output = cstring(strings.clone_to_cstring(args[i + 1]))
				i += 1
			}
		case "--cpu":
			cfg.use_gpu = false
		case "--gpu":
			cfg.use_gpu = true
		case "--help":
			fmt.println("Usage: lumbre [options]")
			fmt.println("  --scene, -s <file.obj>     Load OBJ scene")
			fmt.println("  --width, -w <int>          Image width (default 1024)")
			fmt.println("  --height, -h <int>         Image height (default 576)")
			fmt.println("  --spp <int>                Samples per pixel (default 50)")
			fmt.println("  --depth <int>              Max bounces (default 20)")
			fmt.println("  --max-radiance <float>     Firefly clamp (default 1000)")
			fmt.println("  --output, -o <file.png>    Output file (default render.png)")
			fmt.println("  --cpu                      Force CPU renderer")
			fmt.println("  --gpu                      Force GPU renderer")
			fmt.println("  --debug <mode>             Debug: 1=albedo, 2=normal, 3=depth, 4=primitive id, 5=direct, 6=light count, 7=direct candidates, 8=shadow visibility")
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
			render_gpu(&scene, cfg.image_width, cfg.image_height, cfg.samples_per_pixel, cfg.max_depth, cfg.max_radiance, cfg.file_output, cfg.debug_mode)
			return
		}
	}

	render_cpu(&scene, cfg.image_width, cfg.image_height, cfg.samples_per_pixel, cfg.max_depth, cfg.max_radiance, cfg.file_output)
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
