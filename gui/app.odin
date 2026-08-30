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

	// The IPR worker borrows `app.core.scene`. Take the same lock it holds while
	// stepping, so the old scene is never freed out from under an in-flight
	// dispatch. This can wait up to one batch.
	sync.mutex_lock(&app.ipr.scene_mutex)
	if app.scene_loaded {
		lc.destroy_scene(&app.core.scene)
		delete(app.scene_path)
	}
	app.core.scene = scene
	sync.mutex_unlock(&app.ipr.scene_mutex)

	app.scene_path = strings.clone(path)
	app.scene_loaded = true
	ipr_set_scene(&app.ipr, &app.core.scene)

	// The importer already prints its own flatten summary to stdout, which the
	// log panel picks up; this is the one-line status the title bar shows.
	log_printf(&app.log, "Loaded %s", path)
}

// ── lifecycle ────────────────────────────────────────────────────────────────

app_init :: proc(app: ^App) {
	app.show_viewport = true
	app.show_usd_tree = true
	app.show_usd_text = true
	app.show_properties = true
	app.show_material = true
	app.show_render = true
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

	app.rate_window_t0 = time.now()
	ipr_init(&app.ipr)
	app_wake(app)
}

app_destroy :: proc(app: ^App) {
	// Stop the worker before tearing down anything it borrows.
	ipr_shutdown(&app.ipr)

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
