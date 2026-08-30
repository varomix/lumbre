package main

// Lumbre GUI — SDL3 + SDL_GPU + Dear ImGui shell.
//
// Build:  odin build gui -out:lumbre-gui
//
// The output path matters. `libusd_shim.dylib`'s install name is
// `@rpath/lib/darwin/libusd_shim.dylib` and Odin emits exactly one rpath,
// `@loader_path`, so the executable must sit at the repo root or it will link
// fine and then fail at startup. See plans/GUI.md, Phase 1.
//
// The frame loop is event-driven, not a 60 fps spin: when nothing has changed
// the process blocks in SDL and draws nothing at all. See `run` below and
// plans/GUI.md, "retained-mode behaviour".

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:path/filepath"
import "core:strings"
import "core:time"
import "base:intrinsics"
import NS "core:sys/darwin/Foundation"

import imgui "../third_party/odin-imgui"
import "../third_party/odin-imgui/imgui_impl_sdl3"
import "../third_party/odin-imgui/imgui_impl_sdlgpu3"

import sdl "vendor:sdl3"

DOCKSPACE_NAME :: "LumbreDockSpace"

main :: proc() {
	app: App
	app_init(&app)
	defer app_destroy(&app)

	// Capture stdout early so the importers' and renderer's existing prints land
	// in the Log panel. Failure is not fatal — the app just logs less.
	if !log_capture_stdout(&app.log) {
		log_line(&app.log, "[warn] could not capture stdout; renderer output stays in the terminal")
	}

	if !sdl.Init({.VIDEO}) {
		fmt.eprintln("SDL_Init failed:", sdl.GetError())
		return
	}
	defer sdl.Quit()

	if !wake_event_register() {
		fmt.eprintln("SDL_RegisterEvents failed:", sdl.GetError())
		return
	}

	main_scale := sdl.GetDisplayContentScale(sdl.GetPrimaryDisplay())
	if main_scale <= 0 {
		main_scale = 1
	}

	// Fit the default window inside the display's usable area. A fixed default
	// larger than the screen gets silently clipped by the window manager, which
	// looks like a layout bug rather than a sizing one.
	win_w, win_h: i32 = 1600, 1000
	usable: sdl.Rect
	if sdl.GetDisplayUsableBounds(sdl.GetPrimaryDisplay(), &usable) {
		win_w = min(win_w, usable.w - 80)
		win_h = min(win_h, usable.h - 80)
	}

	window := sdl.CreateWindow(
		"Lumbre",
		win_w,
		win_h,
		{.RESIZABLE, .HIDDEN, .HIGH_PIXEL_DENSITY},
	)
	if window == nil {
		fmt.eprintln("SDL_CreateWindow failed:", sdl.GetError())
		return
	}
	defer sdl.DestroyWindow(window)
	sdl.SetWindowPosition(window, sdl.WINDOWPOS_CENTERED, sdl.WINDOWPOS_CENTERED)

	gpu := sdl.CreateGPUDevice({.MSL, .METALLIB}, false, nil)
	if gpu == nil {
		fmt.eprintln("SDL_CreateGPUDevice failed:", sdl.GetError())
		return
	}
	defer sdl.DestroyGPUDevice(gpu)

	if !sdl.ClaimWindowForGPUDevice(gpu, window) {
		fmt.eprintln("SDL_ClaimWindowForGPUDevice failed:", sdl.GetError())
		return
	}
	defer sdl.ReleaseWindowFromGPUDevice(gpu, window)

	// Present only on vsync: with an event-driven loop there is never a reason
	// to outrun the display.
	_ = sdl.SetGPUSwapchainParameters(gpu, window, .SDR, .VSYNC)

	sdl.ShowWindow(window)

	// ── Dear ImGui ───────────────────────────────────────────────────────────

	imgui.CHECKVERSION()
	imgui.CreateContext()
	defer imgui.DestroyContext()

	io := imgui.GetIO()
	io.ConfigFlags += {.NavEnableKeyboard, .DockingEnable}
	// Multi-viewport (docking panels out into OS windows) is deliberately off:
	// each platform window carries its own repaint lifecycle, which makes the
	// event-driven idle loop considerably harder to keep correct.

	ini_path := resolve_ini_path()
	defer delete(ini_path)
	app.ini_path = strings.clone_to_cstring(ini_path)
	defer delete(app.ini_path)
	io.IniFilename = app.ini_path
	// A missing ini means first run, so the built-in layout is applied instead
	// of leaving every panel floating in the top-left corner.
	first_run := !os.exists(ini_path)

	// Clarisse-style dark theme + Inter font; see gui/theme.odin. Applied before
	// Style_ScaleAllSizes so the theme's spacing is DPI-scaled with everything
	// else, and the font is loaded before the first frame builds the atlas.
	style := imgui.GetStyle()
	theme_apply(style)
	theme_load_font(io, 15.0)
	imgui.Style_ScaleAllSizes(style, main_scale)
	style.FontScaleDpi = main_scale
	io.ConfigDpiScaleFonts = true

	imgui_impl_sdl3.InitForSDLGPU(window)
	defer imgui_impl_sdl3.Shutdown()

	init_info := imgui_impl_sdlgpu3.InitInfo {
		Device               = gpu,
		ColorTargetFormat    = sdl.GetGPUSwapchainTextureFormat(gpu, window),
		MSAASamples          = ._1,
		SwapchainComposition = .SDR,
		PresentMode          = .VSYNC,
	}
	imgui_impl_sdlgpu3.Init(&init_info)
	defer imgui_impl_sdlgpu3.Shutdown()

	// Window rect is logged so a screenshot can target this window alone rather
	// than the whole desktop — handy when verifying layout changes.
	wx, wy, ww, wh, pw, ph: i32
	sdl.GetWindowPosition(window, &wx, &wy)
	sdl.GetWindowSize(window, &ww, &wh)
	sdl.GetWindowSizeInPixels(window, &pw, &ph)
	fmt.printfln(
		"Lumbre GUI ready — %s backend, window %dx%d pt / %dx%d px at %d,%d (scale %.2f)",
		sdl.GetGPUDeviceDriver(gpu),
		ww, wh, pw, ph, wx, wy, main_scale,
	)
	// The window's CGWindowID, so `screencapture -l<id>` can grab this window
	// alone. Capturing a screen rectangle instead is unreliable: on a scaled
	// Retina display the rect does not line up with SDL's window coordinates,
	// and anything floating above the window ends up in the shot.
	fmt.printfln("Lumbre window id: %d", cocoa_window_id(window))

	batch_script: string
	defer delete(batch_script)

	// Optional: open a scene straight away, so `lumbre-gui --scene x.usd` lands
	// in a rendering viewport without going through the file dialog.
	args := os.args[1:]
	for i := 0; i < len(args); i += 1 {
		switch args[i] {
		case "--scene", "-s":
			if i + 1 < len(args) {
				i += 1
				app_load_scene(&app, args[i])
			}
		case "--run-script":
			// Batch mode: run a script against the loaded scene, print what it
			// produced, and exit. Also the end-to-end test for the scripting
			// bridge, since it exercises the same path the panel does.
			if i + 1 < len(args) {
				i += 1
				batch_script = strings.clone(args[i])
			}
		case "--nav-bench":
			secs := 5.0
			if i + 1 < len(args) {
				i += 1
				if v, ok := strconv.parse_f64(args[i]); ok {
					secs = v
				}
			}
			app.nav_bench_seconds = secs
			app.perf.enabled = true
		case "--help", "-h":
			fmt.println("Usage: lumbre-gui [--scene <file>] [--run-script <file.py>] [--nav-bench <seconds>]")
			return
		}
	}

	if batch_script != "" {
		source, read_err := os.read_entire_file_from_path(batch_script, context.allocator)
		if read_err != nil {
			fmt.eprintln("could not read script:", batch_script)
			return
		}
		defer delete(source)

		script_run(&app, string(source))
		fmt.print(strings.to_string(app.script.output))
		if !app.script.last_ok {
			fmt.eprintln("script reported an error")
		}
		return
	}

	viewport: Viewport
	defer viewport_destroy(&viewport, gpu)

	run(&app, window, gpu, &viewport, first_run)

	_ = sdl.WaitForGPUIdle(gpu)
}

// ── the loop ─────────────────────────────────────────────────────────────────

run :: proc(app: ^App, window: ^sdl.Window, gpu: ^sdl.GPUDevice, viewport: ^Viewport, first_run: bool) {
	// How long the next wait may block, in ms. Negative means "block until
	// something happens". Recomputed after every drawn frame.
	idle_timeout_ms: i32 = -1
	build_layout := first_run

	bench_start := time.tick_now()

	for !app.quit {
		frame_start := time.tick_now()
		ev: sdl.Event

		// --nav-bench: tumble the camera every iteration, exactly as a drag
		// does, so the interactive path is exercised without mouse input.
		if app.nav_bench_seconds > 0 {
			if time.duration_seconds(time.tick_since(bench_start)) >= app.nav_bench_seconds {
				perf_report(&app.perf, "Lumbre navigation profile (UI thread)")
				app.quit = true
				break
			}
			cam_start := time.tick_now()
			orbit_camera_tumble(&app.cam, 4, 0)
			app_apply_camera(app)
			perf_record(&app.perf, .CameraApply, cam_start)
			app_wake(app)
		}

		wait_start := time.tick_now()
		if app.frames_pending > 0 {
			// Frames are owed: drain the queue without blocking and draw.
			for sdl.PollEvent(&ev) {
				handle_event(app, &ev)
			}
			perf_record(&app.perf, .Events, wait_start)
		} else {
			// Nothing to draw. Park here — this is where an idle Lumbre spends
			// all of its time, at no CPU cost.
			app.idle = true
			got := idle_timeout_ms < 0 \
				? sdl.WaitEvent(&ev) \
				: sdl.WaitEventTimeout(&ev, idle_timeout_ms)

			if got {
				handle_event(app, &ev)
				for sdl.PollEvent(&ev) {
					handle_event(app, &ev)
				}
			} else {
				// The wait timed out. A finite timeout is only ever set because
				// something needs a periodic repaint (a blinking caret, a
				// tooltip timer), so this is a legitimate reason to draw.
				app_wake(app)
			}
			perf_record(&app.perf, .Wait, wait_start)
		}

		app_poll_pending_scene(app)

		// A finished batch is the other reason to redraw. Pulling it here (not
		// inside the frame) keeps the upload out of the ImGui draw path.
		pull_start := time.tick_now()
		if viewport_pull(viewport, &app.ipr, gpu) {
			app_wake(app)
		}
		perf_record(&app.perf, .ViewportPull, pull_start)

		if app.frames_pending == 0 {
			continue
		}
		app.frames_pending -= 1

		draw_start := time.tick_now()
		draw_frame(app, window, gpu, viewport, &build_layout)
		perf_record(&app.perf, .DrawFrame, draw_start)
		perf_record(&app.perf, .FrameTotal, frame_start)
		app.idle = false

		idle_timeout_ms = next_idle_timeout(app)
		free_all(context.temp_allocator)
	}
}

// Longest sleep that is still correct, given what the UI is currently doing.
// Read after the frame is built, so these reflect the frame just drawn.
@(private = "file")
next_idle_timeout :: proc(app: ^App) -> i32 {
	io := imgui.GetIO()

	// Something is being manipulated — a slider dragged, a menu held open.
	// Keep drawing; do not wait at all.
	if imgui.IsAnyItemActive() || imgui.IsAnyMouseDown() {
		app_wake(app)
		return 0
	}

	// A focused text field has a blinking caret.
	if io.WantTextInput {
		return 500
	}

	// Hovering runs tooltip appear-delay timers, which need the clock to move.
	if imgui.IsAnyItemHovered() {
		return 100
	}

	// Truly idle: block until the OS or another thread has something to say.
	return -1
}

@(private = "file")
handle_event :: proc(app: ^App, ev: ^sdl.Event) {
	imgui_impl_sdl3.ProcessEvent(ev)

	// Every event is a reason to repaint. One frame is not enough: ImGui
	// resolves hover and active state across successive frames.
	app_wake(app)

	#partial switch ev.type {
	case .QUIT:
		app.quit = true
	case .WINDOW_CLOSE_REQUESTED:
		app.quit = true
	}
}

@(private = "file")
draw_frame :: proc(app: ^App, window: ^sdl.Window, gpu: ^sdl.GPUDevice, viewport: ^Viewport, build_layout: ^bool) {
	imgui_impl_sdlgpu3.NewFrame()
	imgui_impl_sdl3.NewFrame()
	imgui.NewFrame()

	dockspace_id := imgui.GetID(DOCKSPACE_NAME)

	if app.layout_reset_requested {
		app.layout_reset_requested = false
		build_layout^ = true
	}
	if build_layout^ {
		build_layout^ = false
		layout_build_default(dockspace_id)
	}

	imgui.DockSpaceOverViewport(dockspace_id, nil, {.PassthruCentralNode})

	draw_main_menu(app, window)
	draw_panels(app, viewport)
	draw_status_bar(app)

	imgui.Render()
	draw_data := imgui.GetDrawData()
	minimised := draw_data.DisplaySize.x <= 0 || draw_data.DisplaySize.y <= 0

	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	if cmd == nil {
		return
	}

	swapchain: ^sdl.GPUTexture
	if !sdl.WaitAndAcquireGPUSwapchainTexture(cmd, window, &swapchain, nil, nil) {
		_ = sdl.SubmitGPUCommandBuffer(cmd)
		return
	}

	if swapchain != nil && !minimised {
		imgui_impl_sdlgpu3.PrepareDrawData(draw_data, cmd)

		target := sdl.GPUColorTargetInfo {
			texture     = swapchain,
			clear_color = {0.09, 0.09, 0.11, 1.0},
			load_op     = .CLEAR,
			store_op    = .STORE,
		}
		pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
		imgui_impl_sdlgpu3.RenderDrawData(draw_data, cmd, pass, nil)
		sdl.EndGPURenderPass(pass)
	}

	_ = sdl.SubmitGPUCommandBuffer(cmd)
	app_update_stats(app)
}

// Keeps the layout file beside the executable rather than in whatever directory
// the app happened to be launched from, so the saved layout follows the app.
@(private = "file")
resolve_ini_path :: proc() -> string {
	exe, err := os.get_executable_path(context.allocator)
	if err != nil {
		return strings.clone("lumbre_gui.ini")
	}
	// `filepath.dir` returns a slice of `exe`, so `exe` must outlive the join.
	defer delete(exe)

	joined, join_err := filepath.join({filepath.dir(exe), "lumbre_gui.ini"})
	if join_err != nil {
		return strings.clone("lumbre_gui.ini")
	}
	return joined
}

// NSWindow's windowNumber is the CGWindowID that `screencapture -l` expects.
// Returns 0 when the platform window cannot be reached, which just means the
// caller falls back to a region capture.
@(private = "file")
cocoa_window_id :: proc(window: ^sdl.Window) -> int {
	props := sdl.GetWindowProperties(window)
	if props == 0 {
		return 0
	}
	ns_window := sdl.GetPointerProperty(props, sdl.PROP_WINDOW_COCOA_WINDOW_POINTER, nil)
	if ns_window == nil {
		return 0
	}
	return int(intrinsics.objc_send(NS.Integer, (^NS.Object)(ns_window), "windowNumber"))
}
