package main

// The headless dataset runner: `lumbre --raster`.
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
// Cameras are what the batch iterates, not frames. `--frame-range` exists but
// is a no-op for content: the scene loads once and the frame number is never
// fed back to it, so a range renders the same image N times under N names.
// Animated stages are Phase 03's problem; stage cameras are what actually
// varies today.

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

// Renders every camera in the scene and writes what each frame is worth.
//
// Returns false only for a failure that makes the whole run pointless — no
// GPU, no scene upload. A single camera that fails to read back is reported
// and skipped, because a partial dataset beats none.
run_raster_batch :: proc(scene: ^lc.Scene, cfg: Render_Config, opts: Raster_Options) -> bool {
	// VIDEO rather than nothing: SDL's GPU backends are initialised through
	// the video subsystem even when no window is ever created.
	if !sdl.Init({.VIDEO}) {
		fmt.eprintln("SDL_Init failed:", sdl.GetError())
		return false
	}
	defer sdl.Quit()

	gpu := sdl.CreateGPUDevice({.MSL, .METALLIB, .SPIRV}, false, nil)
	if gpu == nil {
		fmt.eprintln("SDL_CreateGPUDevice failed:", sdl.GetError())
		return false
	}
	defer sdl.DestroyGPUDevice(gpu)
	// No ClaimWindowForGPUDevice: there is no window, and every target this
	// renders into is one it created itself.

	renderer, ok := rt.renderer_create(gpu)
	if !ok {
		fmt.eprintln("Failed to create the realtime renderer")
		return false
	}
	defer rt.renderer_destroy(&renderer)

	if !rt.renderer_set_scene(&renderer, scene, 1) {
		fmt.eprintln("Failed to upload the scene to the GPU")
		return false
	}

	width := cfg.image_width
	height := cfg.image_height

	// A stage with no cameras still renders: the importer's framing heuristic
	// put one in `scene.camera`, and a single-object file is exactly the case
	// that has none.
	cameras := scene.cameras
	names := scene.camera_names
	if len(cameras) == 0 {
		cameras = {scene.camera}
		names = {""}
	}

	fmt.println("Rasterizing", len(cameras), "camera(s) at", width, "x", height)

	failures := 0
	for cam, i in cameras {
		name := i < len(names) ? names[i] : ""
		stem := raster_output_stem(string(cfg.file_output), name, i, len(cameras))
		defer delete(stem)

		if !render_one_camera(&renderer, scene, cam, width, height, stem, opts) {
			fmt.eprintln("  camera", i, "failed; continuing")
			failures += 1
		}
	}
	return failures < len(cameras)
}

// One camera: beauty, and if asked for, the label EXR and the COCO file.
@(private = "file")
render_one_camera :: proc(
	renderer: ^rt.Renderer,
	scene: ^lc.Scene,
	cam: lc.Camera,
	width, height: i32,
	stem: string,
	opts: Raster_Options,
) -> bool {
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
	} else {
		fmt.eprintln(" ", msg)
		return false
	}

	anns := rt.annotations_derive(frame, scene)
	defer delete(anns)

	json_path := strings.concatenate({stem, ".coco.json"}, context.temp_allocator)
	coco := output.COCO_Frame {
		image_name  = filepath.base(beauty_path),
		image_id    = 0,
		width       = width,
		height      = height,
		class_names = scene.semantic_classes,
		intrinsics  = rt.camera_intrinsics(cam, width, height),
	}
	if msg, json_ok := output.coco_write(coco, anns, json_path); json_ok {
		fmt.println(" ", msg)
	} else {
		fmt.eprintln(" ", msg)
		return false
	}
	return true
}

// The rasterizer's beauty target is RGBA8 with the top row first, which is
// what stb writes by default — the flip the path-traced path needs is for its
// bottom-row-first GPU buffer, and applying it here would silently invert
// every beauty image against its own labels.
@(private = "file")
write_rgba_png :: proc(path: string, pixels: []u8, width, height: i32) -> bool {
	stbi.flip_vertically_on_write(false)
	cpath := strings.clone_to_cstring(path, context.temp_allocator)
	if stbi.write_png(cpath, c.int(width), c.int(height), 4, raw_data(pixels), c.int(width * 4)) == 0 {
		fmt.eprintln("Failed to write", path)
		return false
	}
	return true
}

// The output path minus its extension, with the camera folded in.
//
// A stage's camera prim names are what a person recognises in a directory
// listing, so they win when they exist and are unique enough to be a file
// name. The index is the fallback, and a single camera adds nothing at all —
// `render.png` should stay `render.png` when there is only one of it.
@(private = "file")
raster_output_stem :: proc(output_path, camera_name: string, index, count: int) -> string {
	stem := output_path
	if dot := strings.last_index_byte(stem, '.'); dot > 0 {
		stem = stem[:dot]
	}
	if count <= 1 {
		return strings.clone(stem)
	}
	if camera_name != "" && filepath.base(camera_name) == camera_name {
		return strings.concatenate({stem, ".", camera_name})
	}
	return fmt.aprintf("%s.cam%d", stem, index)
}
