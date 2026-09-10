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
}

// Where one scene's frames go and what they are called.
Raster_Target :: struct {
	width, height: i32,
	// Output path; its extension is replaced per file.
	output:        string,
	// Folded into every file name and used as the COCO image id. Negative
	// leaves names exactly as `--raster` has always written them.
	frame:         int,
	// Render only the camera with this prim name. Empty renders them all.
	camera:        string,
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

	s.gpu = sdl.CreateGPUDevice({.MSL, .METALLIB, .SPIRV}, false, nil)
	if s.gpu == nil {
		fmt.eprintln("SDL_CreateGPUDevice failed:", sdl.GetError())
		sdl.Quit()
		return false
	}
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
	rt.renderer_destroy(&s.renderer)
	sdl.DestroyGPUDevice(s.gpu)
	sdl.Quit()
	s^ = {}
}

// Uploads `scene` and renders every camera it carries (or the one `target`
// names) into files. Paths written are appended to `files` when given.
//
// A single camera that fails to read back is reported and skipped, because a
// partial dataset beats none. `ok` is false only when nothing could be
// rendered at all.
raster_session_render :: proc(
	s: ^Raster_Session,
	scene: ^lc.Scene,
	target: Raster_Target,
	opts: Raster_Options,
	files: ^[dynamic]string = nil,
) -> (rendered: int, ok: bool) {
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

	if target.camera != "" {
		found := false
		for name in names {
			found ||= name == target.camera
		}
		if !found {
			fmt.eprintln("No camera named", target.camera, "in the stage")
			return 0, false
		}
	}

	fmt.println("Rasterizing", target.camera != "" ? 1 : len(cameras), "camera(s) at", target.width, "x", target.height)

	for cam, i in cameras {
		name := i < len(names) ? names[i] : ""
		if target.camera != "" && name != target.camera {
			continue
		}
		stem := raster_output_stem(target.output, name, i, len(cameras), target.frame)
		defer delete(stem)

		if render_one_camera(&s.renderer, scene, cam, target, stem, opts, files) {
			rendered += 1
		} else {
			fmt.eprintln("  camera", i, "failed; continuing")
		}
	}
	return rendered, rendered > 0
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
render_one_camera :: proc(
	renderer: ^rt.Renderer,
	scene: ^lc.Scene,
	cam: lc.Camera,
	target: Raster_Target,
	stem: string,
	opts: Raster_Options,
	files: ^[dynamic]string,
) -> bool {
	width, height := target.width, target.height

	pixels, ok := rt.renderer_read_color(renderer, cam, width, height, opts.view)
	if !ok {
		return false
	}
	defer delete(pixels)

	beauty_path := strings.concatenate({stem, ".png"}, context.temp_allocator)
	if !write_rgba_png(beauty_path, pixels, width, height) {
		return false
	}
	fmt.println("  wrote", beauty_path)
	record_file(files, beauty_path)

	if !opts.labels {
		return true
	}

	// A second pass over the same geometry, deliberately: the label channels
	// take their own un-antialiased, unjittered resolve, which is the whole
	// reason they are not extra G-buffer targets.
	frame, label_ok := rt.labels_read(renderer, cam, width, height)
	if !label_ok {
		fmt.eprintln("  label readback failed")
		return false
	}
	defer rt.label_frame_destroy(&frame)

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
		// The frame number when there is one, so frames from one run can be
		// merged into a single dataset without their ids colliding.
		image_id    = max(target.frame, 0),
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
	if dot := strings.last_index_byte(stem, '.'); dot > 0 {
		stem = stem[:dot]
	}
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
