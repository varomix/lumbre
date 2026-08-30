package main

// Viewport panel: shows the IPR's latest result.
//
// The renderer hands back RGB8 on the CPU, so each new batch is expanded to
// RGBA8 and uploaded into an SDL_GPU texture, which ImGui draws directly —
// `ImTextureID` is a raw `SDL_GPUTexture*` in Dear ImGui 1.92.
//
// Upload happens only when a batch actually lands. Redrawing the panel for any
// other reason reuses the texture already on the GPU.

import "core:fmt"
import "core:sync"

import imgui "../third_party/odin-imgui"
import sdl "vendor:sdl3"

Viewport :: struct {
	texture:    ^sdl.GPUTexture,
	tex_w:      i32,
	tex_h:      i32,
	transfer:   ^sdl.GPUTransferBuffer,
	transfer_bytes: u32,

	// Staging buffers. `cpu` is swapped with the IPR's result buffer, so no
	// per-batch copy is needed; `rgba` is the expanded upload source.
	cpu:        []u8,
	rgba:       []u8,

	displayed_spp: i32,
	// Panel size in pixels, used to drive the render resolution.
	last_panel_w: i32,
	last_panel_h: i32,

	// Active navigation drag. Tracked explicitly rather than read from hover
	// each frame, so a drag that leaves the panel keeps controlling the camera
	// until the button is released.
	nav:        Nav_Mode,
}

Nav_Mode :: enum {
	None,
	Tumble,
	Pan,
	Dolly,
}

viewport_destroy :: proc(v: ^Viewport, gpu: ^sdl.GPUDevice) {
	if v.texture != nil {
		sdl.ReleaseGPUTexture(gpu, v.texture)
		v.texture = nil
	}
	if v.transfer != nil {
		sdl.ReleaseGPUTransferBuffer(gpu, v.transfer)
		v.transfer = nil
	}
	delete(v.cpu)
	delete(v.rgba)
}

@(private = "file")
viewport_ensure_texture :: proc(v: ^Viewport, gpu: ^sdl.GPUDevice, w, h: i32) -> bool {
	if v.texture != nil && v.tex_w == w && v.tex_h == h {
		return true
	}
	if v.texture != nil {
		sdl.ReleaseGPUTexture(gpu, v.texture)
		v.texture = nil
	}

	info := sdl.GPUTextureCreateInfo {
		type                 = .D2,
		format               = .R8G8B8A8_UNORM,
		usage                = {.SAMPLER},
		width                = u32(w),
		height               = u32(h),
		layer_count_or_depth = 1,
		num_levels           = 1,
		sample_count         = ._1,
	}
	v.texture = sdl.CreateGPUTexture(gpu, info)
	if v.texture == nil {
		fmt.eprintln("viewport: CreateGPUTexture failed:", sdl.GetError())
		return false
	}
	v.tex_w = w
	v.tex_h = h
	return true
}

// Pulls the newest IPR result, if any, and uploads it. Returns true when a new
// image was uploaded this frame.
viewport_pull :: proc(v: ^Viewport, ipr: ^IPR, gpu: ^sdl.GPUDevice) -> bool {
	pixels, w, h, spp, got := ipr_take_result(ipr, v.cpu)
	if !got {
		return false
	}
	v.cpu = pixels
	v.displayed_spp = spp

	if w <= 0 || h <= 0 || len(v.cpu) < int(w) * int(h) * 3 {
		return false
	}
	if !viewport_ensure_texture(v, gpu, w, h) {
		return false
	}

	// Expand RGB8 to RGBA8; SDL_GPU has no 24-bit format.
	needed := int(w) * int(h) * 4
	if len(v.rgba) != needed {
		delete(v.rgba)
		v.rgba = make([]u8, needed)
	}
	for i in 0 ..< int(w) * int(h) {
		v.rgba[i * 4 + 0] = v.cpu[i * 3 + 0]
		v.rgba[i * 4 + 1] = v.cpu[i * 3 + 1]
		v.rgba[i * 4 + 2] = v.cpu[i * 3 + 2]
		v.rgba[i * 4 + 3] = 255
	}

	if v.transfer == nil || v.transfer_bytes < u32(needed) {
		if v.transfer != nil {
			sdl.ReleaseGPUTransferBuffer(gpu, v.transfer)
		}
		v.transfer = sdl.CreateGPUTransferBuffer(
			gpu,
			sdl.GPUTransferBufferCreateInfo{usage = .UPLOAD, size = u32(needed)},
		)
		if v.transfer == nil {
			fmt.eprintln("viewport: CreateGPUTransferBuffer failed:", sdl.GetError())
			v.transfer_bytes = 0
			return false
		}
		v.transfer_bytes = u32(needed)
	}

	dst := sdl.MapGPUTransferBuffer(gpu, v.transfer, true)
	if dst == nil {
		return false
	}
	copy(([^]u8)(dst)[:needed], v.rgba[:needed])
	sdl.UnmapGPUTransferBuffer(gpu, v.transfer)

	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	if cmd == nil {
		return false
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUTexture(
		pass,
		sdl.GPUTextureTransferInfo {
			transfer_buffer = v.transfer,
			offset = 0,
			pixels_per_row = u32(w),
			rows_per_layer = u32(h),
		},
		sdl.GPUTextureRegion{texture = v.texture, w = u32(w), h = u32(h), d = 1},
		false,
	)
	sdl.EndGPUCopyPass(pass)
	_ = sdl.SubmitGPUCommandBuffer(cmd)
	return true
}

draw_viewport_panel :: proc(app: ^App, v: ^Viewport) {
	if !imgui.Begin(WINDOW_VIEWPORT, &app.show_viewport) {
		imgui.End()
		return
	}
	defer imgui.End()

	avail := imgui.GetContentRegionAvail()
	if avail.x < 16 || avail.y < 16 {
		return
	}

	// Drive the render resolution from the panel size. Changing it restarts
	// accumulation, so only react once the user stops resizing: the comparison
	// below is against the last size we acted on, not every intermediate one.
	pw := i32(avail.x)
	ph := i32(avail.y)
	if pw != v.last_panel_w || ph != v.last_panel_h {
		v.last_panel_w = pw
		v.last_panel_h = ph
		ipr_set_resolution(&app.ipr, pw, ph)
		// The aspect ratio just changed, so the camera's frustum is stale.
		// Without this the image is stretched after any panel resize.
		app_apply_camera(app)
	}

	if v.texture == nil {
		msg := app.scene_loaded \
			? "Waiting for the first samples..." \
			: "Open a scene to start rendering (File > Open Scene)"
		imgui.TextDisabled(tmp_cstring(msg))
		return
	}

	// Fit the image into the panel without distorting it.
	tex_aspect := f32(v.tex_w) / f32(v.tex_h)
	draw_w := avail.x
	draw_h := draw_w / tex_aspect
	if draw_h > avail.y {
		draw_h = avail.y
		draw_w = draw_h * tex_aspect
	}

	cursor := imgui.GetCursorPos()
	imgui.SetCursorPos({cursor.x + (avail.x - draw_w) * 0.5, cursor.y + (avail.y - draw_h) * 0.5})
	image_origin := imgui.GetCursorScreenPos()

	tex_ref := imgui.TextureRef {
		_TexID = imgui.TextureID(uintptr(rawptr(v.texture))),
	}
	// The renderer's buffer is bottom-row-first, so flip V here rather than
	// paying for a CPU flip on every batch.
	imgui.Image(tex_ref, {draw_w, draw_h}, {0, 1}, {1, 0})
	hovered := imgui.IsItemHovered()
	viewport_handle_input(app, v, hovered, draw_h)

	// Plain left click with no modifier picks. Alt+left is tumble, so this
	// cannot be confused with navigation.
	if hovered && v.nav == .None && imgui.IsMouseClicked(.Left) {
		io := imgui.GetIO()
		if !io.KeyAlt && !io.KeySuper {
			viewport_pick(app, image_origin, {draw_w, draw_h})
		}
	}

	draw_viewport_hud(app, v, image_origin)
}

@(private = "file")
draw_viewport_hud :: proc(app: ^App, v: ^Viewport, image_origin: imgui.Vec2) {
	s := ipr_stats(&app.ipr)

	// Pin the HUD to the top-left of the image itself, not the panel, so it
	// stays put when the image is letterboxed.
	imgui.SetNextWindowPos({image_origin.x + 8, image_origin.y + 8})
	imgui.SetNextWindowBgAlpha(0.55)

	// NoDecoration and NoNav are composites in the C header; the generated
	// bindings only expose their constituent bits.
	flags := imgui.WindowFlags {
		.NoTitleBar, .NoResize, .NoScrollbar, .NoCollapse,
		.NoDocking,
		.AlwaysAutoResize,
		.NoSavedSettings,
		.NoFocusOnAppearing,
		.NoNavInputs, .NoNavFocus,
		.NoMove,
	}
	if imgui.Begin("##viewport_hud", nil, flags) {
		imgui.Text("%d x %d", v.tex_w, v.tex_h)
		if s.converged {
			imgui.Text("%d spp (converged)", s.spp)
		} else {
			imgui.Text("%d / %d spp", s.spp, s.target)
		}
		imgui.Text("%.0f ms/batch  |  %.1f s total", s.batch_ms, s.total_ms / 1000.0)

		if s.enabled {
			if imgui.SmallButton("Pause") {
				ipr_set_enabled(&app.ipr, false)
			}
		} else {
			if imgui.SmallButton("Resume") {
				ipr_set_enabled(&app.ipr, true)
			}
		}
		imgui.SameLine()
		if imgui.SmallButton("Restart") {
			ipr_invalidate(&app.ipr)
		}
	}
	imgui.End()
}

// ── navigation input ─────────────────────────────────────────────────────────
//
// Maya-style: Alt with left/middle/right drags to tumble/pan/dolly, plus a
// plain middle-drag to pan and the wheel to dolly. `A` frames the scene.

@(private = "file")
viewport_handle_input :: proc(app: ^App, v: ^Viewport, hovered: bool, image_h: f32) {
	io := imgui.GetIO()
	changed := false

	// Start a drag only from over the image; continue it anywhere.
	if v.nav == .None && hovered {
		alt := io.KeyAlt
		switch {
		case alt && imgui.IsMouseClicked(.Left):   v.nav = .Tumble
		case alt && imgui.IsMouseClicked(.Middle): v.nav = .Pan
		case alt && imgui.IsMouseClicked(.Right):  v.nav = .Dolly
		case imgui.IsMouseClicked(.Middle):        v.nav = .Pan
		}
	}

	if v.nav != .None {
		button: imgui.MouseButton
		switch v.nav {
		case .Tumble: button = .Left
		case .Pan:    button = .Middle
		case .Dolly:  button = .Right
		case .None:   button = .Left
		}
		// A plain middle-drag pan uses the middle button too, so this covers
		// both entry paths.
		if !imgui.IsMouseDown(button) {
			v.nav = .None
		} else {
			d := io.MouseDelta
			if d.x != 0 || d.y != 0 {
				switch v.nav {
				case .Tumble:
					orbit_camera_tumble(&app.cam, f64(d.x), f64(d.y))
				case .Pan:
					orbit_camera_pan(&app.cam, f64(-d.x), f64(d.y), f64(image_h))
				case .Dolly:
					orbit_camera_dolly(&app.cam, f64(d.x + d.y) * 0.1)
				case .None:
				}
				changed = true
			}
		}
	}

	if hovered && io.MouseWheel != 0 {
		orbit_camera_dolly(&app.cam, f64(io.MouseWheel))
		changed = true
	}

	if hovered && imgui.IsKeyPressed(.A, false) {
		app_frame_all(app)
		// app_frame_all applies and invalidates already.
		return
	}

	if changed {
		app_apply_camera(app)
	}
}

// Converts the click into normalised viewport coordinates and resolves it to a
// material. The camera the ray uses is the one the *displayed* image was
// rendered with, so a pick during navigation matches what is on screen.
@(private = "file")
viewport_pick :: proc(app: ^App, image_origin: imgui.Vec2, image_size: imgui.Vec2) {
	if !app.scene_loaded || image_size.x <= 0 || image_size.y <= 0 {
		return
	}

	mouse := imgui.GetIO().MousePos
	u := f64((mouse.x - image_origin.x) / image_size.x)
	// The renderer's v axis runs bottom-up; the panel's runs top-down.
	vv := 1.0 - f64((mouse.y - image_origin.y) / image_size.y)
	if u < 0 || u > 1 || vv < 0 || vv > 1 {
		return
	}

	// The render worker may be writing a camera into the scene between
	// batches, and picking walks the same geometry, so take the lock it holds.
	sync.mutex_lock(&app.ipr.scene_mutex)
	result := pick_at(&app.core.scene, u, vv)
	sync.mutex_unlock(&app.ipr.scene_mutex)

	if !result.hit {
		log_line(&app.log, "[pick] nothing under the cursor")
		return
	}

	app.selected_material = result.material
	app.show_material = true
	log_printf(
		&app.log,
		"[pick] material %d at %.3f, %.3f, %.3f (%.2f away)",
		result.material,
		result.point.x, result.point.y, result.point.z,
		result.distance,
	)
}
