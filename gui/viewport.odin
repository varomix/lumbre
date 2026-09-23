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
import rt "../realtime"
import sdl "vendor:sdl3"

// Which renderer fills the panel. Path traced is the IPR: a worker thread
// converges samples and the pixels arrive over the CPU. Realtime rasterizes
// through SDL_GPU straight into a texture, with no CPU round-trip and no
// worker.
Render_Mode :: enum {
	Path_Traced,
	Realtime,
}

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

	// ── realtime mode ────────────────────────────────────────────────────────
	// Created on first use, so a GPU that cannot build the raster pipeline
	// costs nothing until someone asks for it.
	mode:       Render_Mode,
	raster:     rt.Renderer,
	rt_ready:   bool,
	rt_failed:  bool,
	// Realtime redraws only when something changed; see plans/GUI.md on the
	// idle-driven loop. An unconditional repaint here would put the app back
	// at 60 fps forever.
	rt_dirty:   bool,
	rt_texture: ^sdl.GPUTexture,
	rt_view:    rt.Debug_View,
	// Bumped by the app whenever the uploaded scene stops matching; mirrors
	// the IPR's own scene key so navigation never rebuilds geometry.
	rt_scene_key: u64,
	// Last material/light edit the rasterizer has taken up.
	rt_edit_serial: u64,

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
	rt.renderer_destroy(&v.raster)
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

// Renders one realtime frame, but only when the previous one is stale.
//
// The rasterizer is cheap enough that repainting every frame would work and be
// wrong: the app's whole redraw model is that an untouched Lumbre draws
// nothing. `rt_dirty` is set by the things that actually change the image —
// camera, scene, panel size, entering the mode — and by nothing else.
@(private = "file")
viewport_step_realtime :: proc(app: ^App, v: ^Viewport, gpu: ^sdl.GPUDevice) {
	if v.rt_failed || !app.scene_loaded {
		return
	}
	if !v.rt_ready {
		renderer, ok := rt.renderer_create(gpu)
		if !ok {
			v.rt_failed = true
			return
		}
		v.raster = renderer
		v.rt_ready = true
		v.rt_dirty = true
	}

	// The IPR worker borrows the same scene, and uploading walks all of it.
	key := ipr_scene_key(&app.ipr)
	if key != v.rt_scene_key || !v.raster.has_scene {
		sync.mutex_lock(&app.ipr.scene_mutex)
		ok := rt.renderer_set_scene(&v.raster, &app.core.scene, key)
		sync.mutex_unlock(&app.ipr.scene_mutex)
		if !ok {
			v.rt_failed = true
			return
		}
		v.rt_scene_key = key
		v.rt_dirty = true
	}

	// Material and light edits do not bump the scene key -- the path tracer
	// updates those in place rather than rebuilding -- so they need their own
	// signal or a colour change would never reach the raster image.
	if serial := ipr_edit_serial(&app.ipr); serial != v.rt_edit_serial {
		sync.mutex_lock(&app.ipr.scene_mutex)
		ok := rt.renderer_refresh_scene(&v.raster, &app.core.scene)
		sync.mutex_unlock(&app.ipr.scene_mutex)
		if !ok { v.rt_failed = true; return }
		v.rt_edit_serial = serial
		v.rt_dirty = true
	}

	if !v.rt_dirty && v.rt_texture != nil {
		return
	}

	// Build the camera the same way `app_apply_camera` does rather than reading
	// `core.scene.camera`. That field is NOT the live camera: camera changes are
	// posted to the IPR instead of written into the scene, deliberately, so a
	// mouse move never waits on `scene_mutex`. Reading it here framed the
	// rasterizer with whatever camera the importer happened to leave behind.
	cam := orbit_camera_build(&app.cam, app_render_aspect(app))

	v.rt_texture = rt.renderer_render(
		&v.raster, cam, v.last_panel_w, v.last_panel_h, v.rt_view,
	)
	v.rt_dirty = false
}

draw_viewport_panel :: proc(app: ^App, v: ^Viewport, gpu: ^sdl.GPUDevice) {
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
		v.rt_dirty = true
	}

	if v.mode == .Realtime {
		viewport_step_realtime(app, v, gpu)
	}

	tex := v.mode == .Realtime ? v.rt_texture : v.texture
	tex_w := v.mode == .Realtime ? v.raster.width : v.tex_w
	tex_h := v.mode == .Realtime ? v.raster.height : v.tex_h
	// The path tracer hands back a bottom-row-first buffer; the rasterizer
	// writes top-row-first like every other render target. Only the former
	// needs the V flip.
	uv0 := v.mode == .Realtime ? imgui.Vec2{0, 0} : imgui.Vec2{0, 1}
	uv1 := v.mode == .Realtime ? imgui.Vec2{1, 1} : imgui.Vec2{1, 0}

	if tex == nil || tex_w <= 0 || tex_h <= 0 {
		msg: string
		switch {
		case v.rt_failed:  msg = "Realtime renderer unavailable on this GPU"
		case !app.scene_loaded: msg = "Open a scene to start rendering (File > Open Scene)"
		case:              msg = "Waiting for the first samples..."
		}
		imgui.TextDisabled(tmp_cstring(msg))
		return
	}

	// Fit the image into the panel without distorting it.
	tex_aspect := f32(tex_w) / f32(tex_h)
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
		_TexID = imgui.TextureID(uintptr(rawptr(tex))),
	}
	imgui.Image(tex_ref, {draw_w, draw_h}, uv0, uv1)
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
		if imgui.SmallButton(v.mode == .Realtime ? "Realtime" : "Path traced") {
			viewport_set_mode(app, v, v.mode == .Realtime ? .Path_Traced : .Realtime)
		}

		if v.mode == .Realtime {
			imgui.Text("%d x %d", v.raster.width, v.raster.height)
			draw_realtime_channels(v)
			if v.rt_view == .Shaded {
				draw_realtime_look(v)
			}
		} else {
			draw_ipr_stats(app, v, s)
		}
	}
	imgui.End()
}

// Switches renderer, and parks the one that is not being shown.
//
// Leaving the path tracer running behind the rasterizer would keep a GPU
// dispatch and a worker thread busy converging an image nobody can see, which
// on a large scene is most of the machine. Returning to it re-enables the
// worker; whatever camera changes happened meanwhile were posted to it as they
// occurred, so it restarts from the right view.
//
// This does override a manual Pause. That is the lesser surprise: a paused
// path tracer that silently resumes is easier to notice than one that quietly
// eats a GPU while a different renderer is on screen.
viewport_set_mode :: proc(app: ^App, v: ^Viewport, mode: Render_Mode) {
	if v.mode == mode {
		return
	}
	v.mode = mode
	v.rt_dirty = true
	ipr_set_enabled(&app.ipr, mode == .Path_Traced)
}

// The G-buffer channel selector. Deferred lighting does not exist yet, so
// `Shaded` currently shows albedo; the raw channels are the point of this pass
// until it does.
@(private = "file")
draw_realtime_channels :: proc(v: ^Viewport) {
	views := []rt.Debug_View {
		.Shaded, .Albedo, .Normal, .Roughness,
		.Metallic, .Emission, .Depth, .Instance, .Semantic,
	}
	names := []cstring{"Shaded", "Albedo", "Normal", "Rough", "Metal", "Emiss", "Depth", "Inst", "Sem"}

	for view, i in views {
		if i % 4 != 0 {
			imgui.SameLine()
		}
		active := v.rt_view == view
		if active {
			imgui.PushStyleColorImVec4(.Button, {0.26, 0.42, 0.62, 1.0})
		}
		if imgui.SmallButton(names[i]) {
			v.rt_view = view
			v.rt_dirty = true
		}
		if active {
			imgui.PopStyleColor()
		}
	}
}

// Exposure and view transform for the shaded view. Double-click the slider to
// reset exposure to 0.
@(private = "file")
draw_realtime_look :: proc(v: ^Viewport) {
	st := &v.raster.settings
	transforms := []rt.View_Transform{.Standard, .Neutral, .AgX}
	names := []cstring{"Std", "Neutral", "AgX"}
	for t, i in transforms {
		if i > 0 {
			imgui.SameLine()
		}
		if imgui.RadioButton(names[i], st.view_transform == t) {
			st.view_transform = t
			v.rt_dirty = true
		}
	}
	imgui.SetNextItemWidth(140)
	if imgui.SliderFloat("EV", &st.exposure, -6, 6, "%.1f") {
		v.rt_dirty = true
	}
	if imgui.IsItemHovered() && imgui.IsMouseDoubleClicked(.Left) {
		st.exposure = 0
		v.rt_dirty = true
	}
	if imgui.Checkbox("FXAA", &st.fxaa) {
		v.rt_dirty = true
	}
	imgui.SameLine()
	if imgui.Checkbox("AO", &st.ao) {
		v.rt_dirty = true
	}
	if st.ao {
		imgui.SetNextItemWidth(140)
		if imgui.SliderFloat("AO radius", &st.ao_radius, 0.1, 5, "%.2f", {.Logarithmic}) {
			v.rt_dirty = true
		}
	}
}

// The path-traced HUD: convergence and the IPR's controls. Realtime mode has
// no samples to report and no worker to pause, so it shows none of this.
@(private = "file")
draw_ipr_stats :: proc(app: ^App, v: ^Viewport, s: IPR_Stats) {
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
		v.rt_dirty = true
		return
	}

	if changed {
		app_apply_camera(app)
		// The rasterizer has no accumulation to invalidate, only a stale frame.
		v.rt_dirty = true
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
	cam := orbit_camera_build(&app.cam, app_render_aspect(app))
	sync.mutex_lock(&app.ipr.scene_mutex)
	result := pick_at(&app.core.scene, cam, u, vv)
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
