package main

// Application state shared by every panel. Panels take a `^App` and are pure
// UI: they read this and write edit requests back into it. Nothing in a panel
// talks to the renderer or to USD directly.

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:strings"
import "core:sync"
import "core:time"

import lc "../core"
import m "core:math/linalg/glsl"
import imp "../importers"

import sdl "vendor:sdl3"

App :: struct {
	// ── renderer state ───────────────────────────────────────────────────────
	core:         lc.Lumbre_Core,
	scene_loaded: bool,
	scene_path:   string,

	// ── panel visibility ─────────────────────────────────────────────────────
	show_viewport:   bool,
	show_usd_tree:   bool,
	show_usd_text:   bool,
	show_properties: bool,
	show_material:   bool,
	show_render:     bool,
	show_lights:     bool,
	show_script:     bool,
	show_log:        bool,

	// ── redraw policy (see plans/GUI.md, "retained-mode behaviour") ──────────
	// Budget of frames still owed. An input event grants several, because
	// ImGui needs a few frames to settle (an item becoming active, a tooltip
	// appearing, a dock layout resolving). Zero means "draw nothing, block".
	frames_pending: int,
	quit:           bool,

	// ── stats, so idle behaviour is observable rather than assumed ───────────
	redraw_count:    u64,
	redraw_rate:     f64,
	rate_window_t0:  time.Time,
	rate_window_n0:  u64,
	idle:            bool,

	// ── layout ───────────────────────────────────────────────────────────────
	layout_reset_requested: bool,
	ini_path:               cstring,

	ipr: IPR,
	cam: Orbit_Camera,
	usd: Usd_Stage_View,
	script: Script_State,
	render_job: Render_Job,

	// Offline output settings, separate from the viewport's so a final render
	// can be higher quality than what is being previewed.
	out_width:     i32,
	out_height:    i32,
	out_spp:       i32,
	out_max_depth: i32,
	out_aovs:      bool,
	out_denoise:   bool,
	out_compress:  bool,
	out_path:      [512]u8,
	selected_material: int,
	// IPR path-depth, separate from settings.max_depth so the viewport can be
	// cheaper than a final render without changing what a final render does.
	ipr_max_depth: i32,
	perf: Perf,
	// --nav-bench: drive the camera automatically for this many seconds, then
	// print a profile and exit. Makes the interactive path measurable without a
	// human dragging the mouse.
	nav_bench_seconds: f64,
	log: Log,
}

// Number of frames an event buys. One is not enough: ImGui resolves hover,
// active-item and docking state over successive frames, so a single redraw
// leaves the UI visibly stale (a button that never shows its hover state).
WAKE_FRAMES :: 3

app_wake :: proc(app: ^App) {
	app.frames_pending = WAKE_FRAMES
}

// ── cross-thread wake ────────────────────────────────────────────────────────
//
// The whole point of the retained-mode loop is that the UI thread may block
// indefinitely in WaitEvent. Anything that produces work on another thread (the
// log drain now; the IPR worker in Phase 4) must be able to break that block.
// SDL_PushEvent is thread-safe and is the one mechanism that makes an
// indefinite block safe.

g_wake_event_type: u32 = 0

wake_event_register :: proc() -> bool {
	g_wake_event_type = sdl.RegisterEvents(1)
	return g_wake_event_type != 0
}

request_wake :: proc() {
	if g_wake_event_type == 0 {
		return
	}
	ev: sdl.Event
	ev.type = sdl.EventType(g_wake_event_type)
	_ = sdl.PushEvent(&ev)
}

// ── scene loading ────────────────────────────────────────────────────────────
//
// SDL's file dialog answers on a callback rather than returning a path, so the
// result is parked here and consumed by the main loop.

@(private = "file")
g_pending_scene: struct {
	mutex: sync.Mutex,
	path:  string,
	set:   bool,
}

@(private = "file")
open_dialog_callback :: proc "c" (userdata: rawptr, filelist: [^]cstring, filter: c.int) {
	context = runtime.default_context()

	// filelist is nil on error, and points at a nil first entry when the user
	// cancels. Neither is worth reporting.
	if filelist == nil || filelist[0] == nil {
		return
	}

	sync.mutex_lock(&g_pending_scene.mutex)
	if g_pending_scene.set {
		delete(g_pending_scene.path)
	}
	g_pending_scene.path = strings.clone(string(filelist[0]))
	g_pending_scene.set = true
	sync.mutex_unlock(&g_pending_scene.mutex)

	request_wake()
}

app_open_scene_dialog :: proc(app: ^App, window: ^sdl.Window) {
	@(static) filters := [?]sdl.DialogFileFilter {
		{name = "Scene files", pattern = "usd;usda;usdc;usdz;obj;gltf;glb"},
		{name = "USD", pattern = "usd;usda;usdc;usdz"},
		{name = "OBJ", pattern = "obj"},
		{name = "glTF", pattern = "gltf;glb"},
		{name = "All files", pattern = "*"},
	}
	sdl.ShowOpenFileDialog(open_dialog_callback, app, window, &filters[0], len(filters), nil, false)
}

// Picks up a path chosen by the dialog, if any. Called once per frame from the
// main loop, on the UI thread.
app_poll_pending_scene :: proc(app: ^App) {
	sync.mutex_lock(&g_pending_scene.mutex)
	path: string
	got := g_pending_scene.set
	if got {
		path = g_pending_scene.path
		g_pending_scene.path = ""
		g_pending_scene.set = false
	}
	sync.mutex_unlock(&g_pending_scene.mutex)

	if got {
		app_load_scene(app, path)
		delete(path)
	}
}

app_load_scene :: proc(app: ^App, path: string) {
	fmt.println("Loading scene:", path)

	cfg := app.core.settings
	cfg.scene_file = strings.clone_to_cstring(path)
	defer delete(cfg.scene_file)

	scene, ok := imp.make_scene(cfg)
	if !ok {
		fmt.eprintln("Failed to load scene:", path)
		log_printf(&app.log, "[error] failed to load %s", path)
		return
	}

	// The IPR worker borrows `app.core.scene`, so park it before the old scene
	// is freed. Waiting for the worker to become idle is bounded by one batch;
	// taking scene_mutex directly would instead queue behind the worker's
	// immediate re-acquire and wait for the whole image to converge.
	ipr_pause_and_wait(&app.ipr)
	if app.scene_loaded {
		lc.destroy_scene(&app.core.scene)
		delete(app.scene_path)
	}
	app.core.scene = scene

	app.scene_path = strings.clone(path)
	app.scene_loaded = true

	// Start from the scene's authored camera when there is one, otherwise frame
	// the whole thing. Either way the camera is rebuilt for the viewport's
	// aspect before the first batch runs.
	aspect := app_render_aspect(app)
	if app.core.scene.camera.lens_radius >= 0 && m.length(app.core.scene.camera.horizontal) > 0 {
		orbit_camera_from_scene(&app.cam, &app.core.scene, aspect)
	} else {
		orbit_camera_frame_scene(&app.cam, &app.core.scene, aspect)
	}
	app.core.scene.camera = orbit_camera_build(&app.cam, aspect)

	ipr_set_scene(&app.ipr, &app.core.scene)
	ipr_set_enabled(&app.ipr, true)

	// Apply any saved look before the first batch, so the viewport never shows
	// the un-looked scene first.
	look_load(app)

	// Open a second, read-only stage for the USD panels. Non-USD scenes leave
	// them empty rather than pretending to have a hierarchy.
	usd_view_open(&app.usd, app, path)

	// The importer already prints its own flatten summary to stdout, which the
	// log panel picks up; this is the one-line status the title bar shows.
	log_printf(&app.log, "Loaded %s", path)
}

// Aspect ratio of what the IPR is actually rendering, which is what the camera
// must be built for.
app_render_aspect :: proc(app: ^App) -> f64 {
	sync.mutex_lock(&app.ipr.mutex)
	w := app.ipr.width
	h := app.ipr.height
	sync.mutex_unlock(&app.ipr.mutex)
	if h <= 0 {
		return 1
	}
	return f64(w) / f64(h)
}

// Rebuilds the scene camera from the orbit model and restarts accumulation.
// Every camera change goes through here.
app_apply_camera :: proc(app: ^App) {
	if !app.scene_loaded {
		return
	}
	aspect := app_render_aspect(app)

	// Post the camera rather than writing it into the scene. Taking scene_mutex
	// here made every camera change wait for the worker's dispatch — and since
	// the worker re-acquires immediately, in practice for the whole image to
	// converge: 2.1 s per mouse-move on a 157k-triangle scene.
	ipr_set_camera(&app.ipr, orbit_camera_build(&app.cam, aspect))
}

app_frame_all :: proc(app: ^App) {
	if !app.scene_loaded {
		return
	}
	if orbit_camera_frame_scene(&app.cam, &app.core.scene, app_render_aspect(app)) {
		app_apply_camera(app)
	}
}

// The scene's GPU resources are no longer valid — geometry, light counts, or
// the environment changed. Costs a full cache rebuild, so it is used only where
// an in-place update genuinely cannot work.
ipr_scene_changed :: proc(app: ^App) {
	sync.mutex_lock(&app.ipr.mutex)
	app.ipr.scene_key += 1
	app.ipr.generation += 1
	sync.cond_broadcast(&app.ipr.cond)
	sync.mutex_unlock(&app.ipr.mutex)
}

// Render settings changed in a way that invalidates the accumulated image but
// not the scene's GPU resources.
ipr_settings_changed :: proc(app: ^App) {
	sync.mutex_lock(&app.ipr.mutex)
	app.ipr.render_settings = app.core.settings
	app.ipr.max_depth = app.ipr_max_depth
	app.ipr.generation += 1
	sync.cond_broadcast(&app.ipr.cond)
	sync.mutex_unlock(&app.ipr.mutex)
}

// ── lifecycle ────────────────────────────────────────────────────────────────

app_init :: proc(app: ^App) {
	app.ipr_max_depth = 12
	script_init(&app.script)

	app.out_width = 1920
	app.out_height = 1080
	app.out_spp = 256
	app.out_max_depth = 20
	app.out_compress = true
	copy(app.out_path[:], transmute([]u8)string("render.exr"))
	app.usd.selected = -1
	app.usd.text_for = -2
	app.usd.props_for = -2

	app.show_viewport = true
	app.show_usd_tree = true
	app.show_usd_text = true
	app.show_properties = true
	app.show_material = true
	app.show_render = true
	app.show_lights = true
	app.show_script = true
	app.show_log = true

	app.core.settings = lc.Render_Config {
		image_width           = 1024,
		image_height          = 576,
		samples_per_pixel     = 50,
		max_depth             = 20,
		max_radiance          = 1000.0,
		roughness_cutoff      = 0.95,
		use_gpu               = true,
		gi_cache_enabled      = true,
		gi_cache_normal_angle = 0.5,
		photon_enabled        = true,
		photon_count          = 200000,
		photon_bounces        = 8,
		hdri_intensity        = 1.0,
		sun_dir               = lc.Vec3{-1.0, -1.0, -1.0},
		sun_color             = lc.Color{1.0, 1.0, 1.0},
		sun_angle             = 0.53,
		usd_subdiv_level      = 2,
	}

	app.cam = Orbit_Camera{
		target   = lc.Vec3{0, 0, 0},
		distance = 10,
		yaw      = 0,
		pitch    = 0.35,
		vfov     = 40,
	}

	app.rate_window_t0 = time.now()
	ipr_init(&app.ipr)
	// Seed the worker's settings snapshot. Without this it renders with a
	// zeroed Render_Config — max_radiance 0 clamps everything to black.
	ipr_settings_changed(app)
	app_wake(app)
}

app_destroy :: proc(app: ^App) {
	// Stop the worker before tearing down anything it borrows.
	ipr_shutdown(&app.ipr)
	perf_destroy(&app.perf)
	usd_view_close(&app.usd)
	script_destroy(&app.script)
	render_job_shutdown(&app.render_job)

	if app.scene_loaded {
		lc.destroy_scene(&app.core.scene)
		delete(app.scene_path)
	}
	log_destroy(&app.log)
}

// Rolling redraw-rate estimate, sampled over a short window. Displayed in the
// status bar so the idle behaviour can be seen rather than assumed.
app_update_stats :: proc(app: ^App) {
	app.redraw_count += 1

	elapsed := time.duration_seconds(time.since(app.rate_window_t0))
	if elapsed >= 0.5 {
		app.redraw_rate = f64(app.redraw_count - app.rate_window_n0) / elapsed
		app.rate_window_t0 = time.now()
		app.rate_window_n0 = app.redraw_count
	}
}
