package main

// The headless dataset runner: `lumbre --raster`, and `lumbre.render` under
// `lumbre --script`.
//
// Phase 02 of plans/REALTIME_PHASE02.md, and the reason the phase exists. The
// path-tracing CLI renders one beauty image from one camera; this renders the
// rasterizer's output plus the ground truth that makes a frame trainable, once
// per camera the stage carries, with no window anywhere.
//
// No renderer surgery was needed for that. `renderer_create` already took a
// bare `GPUDevice` and `renderer_render` already drew into its own target,
// with the swapchain touched only by the GUI's blit — so a device created
// without `ClaimWindowForGPUDevice` drives the whole thing unchanged.
//
// Phase 03 splits the run into a session. `--raster` renders one scene and
// exits; a script renders hundreds of variations of one, and creating a GPU
// device per frame would spend the run on start-up. A session keeps the device
// and renderer and takes a new scene per call under a new key, which is exactly
// what makes `renderer_set_scene` upload again.

import "core:c"
import "core:fmt"
import "core:os"
import "core:time"
import "core:path/filepath"
import "core:strings"

import lc "./core"
import "./output"
import rt "./realtime"
import sdl "vendor:sdl3"
import stbi "vendor:stb/image"

Raster_Options :: struct {
	// Write the label EXR and the COCO file alongside the beauty image.
	// Without it this is just a fast preview renderer.
	labels: bool,
	// Beauty channel to render. `.Shaded` is the deferred lighting; the debug
	// views are here because "the mask is wrong" and "the G-buffer is wrong"
	// are different bugs and separating them beats guessing.
	view:   rt.Debug_View,
	// Exposure and view transform for the shaded view, from `--exposure` and
	// `--view-transform`, FXAA and AO from `--fxaa`, `--ao` and `--ao-radius`.
	// The default view transform
	// matches the path tracer's display encode.
	settings: rt.Render_Settings,
	// ZIP the label EXR, from `--zip`.
	exr_compress: bool,
}

Raster_Session :: struct {
	gpu:       ^sdl.GPUDevice,
	renderer:  rt.Renderer,
	// Bumped for every scene handed to the session, so the renderer never
	// mistakes a new variation for the scene it already uploaded.
	scene_key: u64,
	open:      bool,
	captures: [2]rt.Capture,
}

// Where one scene's frames go and what they are called.
Raster_Target :: struct {
	width, height: i32,
	// Output path; its extension is replaced per file.
	output:        string,
	// Folded into every file name. Negative
	// leaves names exactly as `--raster` has always written them.
	frame:         int,
	// Full camera prim path or an unambiguous short name. Empty renders all.
	camera:        string,
	classes:       []string, // optional fixed vocabulary; id = index + 1
}

// Brings up SDL, a GPU device and the renderer. False with a message printed
// if any of them fail; nothing needs closing in that case.
raster_session_open :: proc(s: ^Raster_Session) -> bool {
	// VIDEO rather than nothing: SDL's GPU backends are initialised through
	// the video subsystem even when no window is ever created.
	if !sdl.Init({.VIDEO}) {
		fmt.eprintln("SDL_Init failed:", sdl.GetError())
		return false
	}

	driver := os.get_env("LUMBRE_GPU_DRIVER", context.temp_allocator)
	preferred: cstring
	if driver != "" { preferred = strings.clone_to_cstring(driver, context.temp_allocator) }
	debug := os.get_env("LUMBRE_GPU_DEBUG", context.temp_allocator) == "1"
	s.gpu = sdl.CreateGPUDevice({.MSL, .METALLIB, .SPIRV}, debug, preferred)
	if s.gpu == nil {
		fmt.eprintln("SDL_CreateGPUDevice failed:", sdl.GetError())
		sdl.Quit()
		return false
	}
	fmt.println("Realtime GPU backend:", sdl.GetGPUDeviceDriver(s.gpu))
	// No ClaimWindowForGPUDevice: there is no window, and every target this
	// renders into is one it created itself.

	renderer, ok := rt.renderer_create(s.gpu)
	if !ok {
		fmt.eprintln("Failed to create the realtime renderer")
		sdl.DestroyGPUDevice(s.gpu)
		sdl.Quit()
		return false
	}
	s.renderer = renderer
	s.open = true
	return true
}

raster_session_close :: proc(s: ^Raster_Session) {
	if !s.open {
		return
	}
	for &capture in s.captures { rt.capture_destroy(s.gpu, &capture) }
	rt.renderer_destroy(&s.renderer)
	sdl.DestroyGPUDevice(s.gpu)
	sdl.Quit()
	s^ = {}
}

// Uploads `scene` and renders every camera it carries (or the one `target`
// names) into files. Paths written are appended to `files` when given.
//
// Drain every queued camera on failure, but report an incomplete batch as
// failed so scripts cannot silently accept partial datasets.
raster_session_render :: proc(
	s: ^Raster_Session,
	scene: ^lc.Scene,
	target: Raster_Target,
	opts: Raster_Options,
	files: ^[dynamic]string = nil,
) -> (rendered: int, ok: bool) {
	start := time.tick_now()
	defer if os.get_env("LUMBRE_RASTER_BENCH", context.temp_allocator) == "1" {
		ms := time.duration_milliseconds(time.tick_since(start))
		bytes: u64
		for capture in s.captures { for capacity in capture.capacities { bytes += u64(capacity) } }
		fmt.printfln("Realtime batch: frames=%d total_ms=%.3f frames_per_second=%.2f geometry_uploads=%d environment_builds=%d transfer_bytes=%d",
			rendered, ms, f64(rendered) * 1000 / max(ms, 0.001), s.renderer.geometry_uploads, s.renderer.environment_builds, bytes)
	}
	if opts.labels && !output.dataset_classes(scene, target.output, target.classes) {
		fmt.eprintln("Invalid or incompatible dataset class registry for", target.output)
		return 0, false
	}
	s.renderer.settings = opts.settings
	s.scene_key += 1
	if !rt.renderer_set_scene(&s.renderer, scene, s.scene_key) {
		fmt.eprintln("Failed to upload the scene to the GPU")
		return 0, false
	}

	// A stage with no cameras still renders: the importer's framing heuristic
	// put one in `scene.camera`, and a single-object file is exactly the case
	// that has none.
	cameras := scene.cameras
	names := scene.camera_names
	if len(cameras) == 0 {
		cameras = {scene.camera}
		names = {""}
	}

	selected := -1
	if target.camera != "" {
		matches := 0
		for name, i in names {
			path := i < len(scene.camera_paths) ? scene.camera_paths[i] : ""
			if target.camera == path || target.camera == name { selected = i; matches += 1 }
		}
		if matches != 1 {
			fmt.eprintln("Camera must match exactly one prim; use its full path:", target.camera)
			return 0, false
		}
	}

	name_counts := make(map[string]int, context.temp_allocator)
	for name in names { name_counts[name] += 1 }
	duplicate_names := false
	for _, count in name_counts { duplicate_names ||= count > 1 }
	fmt.println("Rasterizing", target.camera != "" ? 1 : len(cameras), "camera(s) at", target.width, "x", target.height)

	Pending :: struct { cam: lc.Camera, stem: string }
	pending: [2]Pending
	head, queued := 0, 0
	failed := false
	// Scene metadata remains alive and immutable until this synchronous call
	// drains both captures. No annotation can observe the next script edit.
	finish :: proc(s: ^Raster_Session, item: Pending, slot: int, scene: ^lc.Scene, target: Raster_Target, opts: Raster_Options, files: ^[dynamic]string) -> bool {
		defer delete(item.stem)
		pixels, frame, ok := rt.capture_finish(s.gpu, &s.captures[slot])
		if !ok { return false }
		defer delete(pixels)
		defer rt.label_frame_destroy(&frame)
		return write_one_camera(scene, item.cam, target, item.stem, opts, files, pixels, frame)
	}
	for cam, i in cameras {
		name := i < len(names) ? names[i] : ""
		if selected >= 0 && i != selected {
			continue
		}
		// A camera picked by name is the only one rendered, so it is left out
		// of the file name just as a stage's single camera is.
		count := target.camera != "" ? 1 : len(cameras)
		// When short names collide, suffix every camera with its full prim
		// path identity. Reordering cameras does not change their filenames.
		if duplicate_names {
			path := i < len(scene.camera_paths) ? scene.camera_paths[i] : fmt.tprintf("camera:%d", i)
			name = fmt.tprintf("%s_%013x", name, output.dataset_id(path))
		}
		stem := raster_output_stem(target.output, name, i, count, target.frame)
		if queued == len(pending) {
			if finish(s, pending[head], head, scene, target, opts, files) { rendered += 1 } else { failed = true }
			head = (head + 1) % len(pending)
			queued -= 1
		}
		slot := (head + queued) % len(pending)
		if rt.capture_begin(&s.renderer, &s.captures[slot], cam, target.width, target.height, opts.view, opts.labels) {
			pending[slot] = {cam, stem}
			queued += 1
		} else { delete(stem); failed = true }
	}
	for queued > 0 {
		if finish(s, pending[head], head, scene, target, opts, files) { rendered += 1 } else { failed = true }
		head = (head + 1) % len(pending)
		queued -= 1
	}
	return rendered, rendered > 0 && !failed
}

// `lumbre --raster`: one scene, every camera, then exit.
run_raster_batch :: proc(scene: ^lc.Scene, cfg: Render_Config, opts: Raster_Options) -> bool {
	session: Raster_Session
	if !raster_session_open(&session) {
		return false
	}
	defer raster_session_close(&session)

	target := Raster_Target {
		width  = cfg.image_width,
		height = cfg.image_height,
		output = string(cfg.file_output),
		frame  = -1,
	}
	_, ok := raster_session_render(&session, scene, target, opts)
	return ok
}

// One camera: beauty, and if asked for, the label EXR and the COCO file.
@(private = "file")
write_one_camera :: proc(
	scene: ^lc.Scene,
	cam: lc.Camera,
	target: Raster_Target,
	stem: string,
	opts: Raster_Options,
	files: ^[dynamic]string,
	pixels: []u8,
	frame: lc.Label_Frame,
) -> bool {
	width, height := target.width, target.height

	beauty_path := strings.concatenate({stem, ".png"}, context.temp_allocator)
	if !write_rgba_png(beauty_path, pixels, width, height) {
		return false
	}
	fmt.println("  wrote", beauty_path)
	record_file(files, beauty_path)

	if !opts.labels {
		return true
	}

	exr_path := strings.concatenate({stem, ".labels.exr"}, context.temp_allocator)
	if msg, exr_ok := output.write_label_exr(frame, exr_path, opts.exr_compress); exr_ok {
		fmt.println(" ", msg)
		record_file(files, exr_path)
	} else {
		fmt.eprintln(" ", msg)
		return false
	}

	anns := rt.annotations_derive(frame, scene)
	defer delete(anns)

	json_path := strings.concatenate({stem, ".coco.json"}, context.temp_allocator)
	coco := output.COCO_Frame {
		image_name  = filepath.base(beauty_path),
		// File identity distinguishes cameras as well as numbered frames.
		image_id    = output.dataset_id(filepath.base(beauty_path)),
		width       = width,
		height      = height,
		class_names = scene.semantic_classes,
		intrinsics  = rt.camera_intrinsics(cam, width, height),
	}
	if msg, json_ok := output.coco_write(coco, anns, json_path); json_ok {
		fmt.println(" ", msg)
		record_file(files, json_path)
	} else {
		fmt.eprintln(" ", msg)
		return false
	}
	return true
}

@(private = "file")
record_file :: proc(files: ^[dynamic]string, path: string) {
	if files != nil {
		append(files, strings.clone(path))
	}
}

// The rasterizer's beauty target is RGBA8 with the top row first, which is
// what stb writes by default — the flip the path-traced path needs is for its
// bottom-row-first GPU buffer, and applying it here would silently invert
// every beauty image against its own labels.
@(private = "file")
write_rgba_png :: proc(path: string, pixels: []u8, width, height: i32) -> bool {
	if !output.ensure_parent_dir(path) {
		fmt.eprintln("Cannot create the directory for", path)
		return false
	}
	stbi.flip_vertically_on_write(false)
	cpath := strings.clone_to_cstring(path, context.temp_allocator)
	if stbi.write_png(cpath, c.int(width), c.int(height), 4, raw_data(pixels), c.int(width * 4)) == 0 {
		fmt.eprintln("Failed to write", path)
		return false
	}
	return true
}

// The output path minus its extension, with the frame and camera folded in.
//
// A stage's camera prim names are what a person recognises in a directory
// listing, so they win when they exist and are unique enough to be a file
// name. The index is the fallback, and a single camera adds nothing at all —
// `render.png` should stay `render.png` when there is only one of it. The
// frame goes before the camera so a directory lists in frame order.
@(private = "file")
raster_output_stem :: proc(output_path, camera_name: string, index, count: int, frame: int) -> string {
	stem := output_path
	stem = strings.trim_suffix(stem, filepath.ext(stem))
	if frame >= 0 {
		stem = fmt.tprintf("%s.%04d", stem, frame)
	}
	if count <= 1 {
		return strings.clone(stem)
	}
	if camera_name != "" && filepath.base(camera_name) == camera_name {
		return strings.concatenate({stem, ".", camera_name})
	}
	return fmt.aprintf("%s.cam%d", stem, index)
}
