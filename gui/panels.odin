package main

// Menu bar, status bar, and the panel set.
//
// Most panels are deliberately empty shells at this stage: Phase 3 is the app
// shell, and each one is filled in by a later phase. They exist now so the dock
// layout, the persisted imgui.ini, and the View menu are real and stable from
// the start — retrofitting a window into a saved layout is what produces
// layouts users have to reset by hand.

import "core:fmt"
import "core:strings"
import "core:sync"

import imgui "../third_party/odin-imgui"
import sdl "vendor:sdl3"

// ── menu bar ─────────────────────────────────────────────────────────────────

draw_main_menu :: proc(app: ^App, window: ^sdl.Window) {
	if !imgui.BeginMainMenuBar() {
		return
	}
	defer imgui.EndMainMenuBar()

	if imgui.BeginMenu("File") {
		if imgui.MenuItem("Open Scene...", "Cmd+O") {
			app_open_scene_dialog(app, window)
		}
		imgui.Separator()
		if imgui.MenuItem("Quit", "Cmd+Q") {
			app.quit = true
		}
		imgui.EndMenu()
	}

	if imgui.BeginMenu("View") {
		imgui.MenuItemBoolPtr(WINDOW_VIEWPORT, nil, &app.show_viewport)
		imgui.MenuItemBoolPtr(WINDOW_USD_TREE, nil, &app.show_usd_tree)
		imgui.MenuItemBoolPtr(WINDOW_USD_TEXT, nil, &app.show_usd_text)
		imgui.MenuItemBoolPtr(WINDOW_PROPERTIES, nil, &app.show_properties)
		imgui.MenuItemBoolPtr(WINDOW_MATERIAL, nil, &app.show_material)
		imgui.MenuItemBoolPtr(WINDOW_RENDER, nil, &app.show_render)
		imgui.MenuItemBoolPtr(WINDOW_SCRIPT, nil, &app.show_script)
		imgui.MenuItemBoolPtr(WINDOW_LOG, nil, &app.show_log)
		imgui.Separator()
		if imgui.MenuItem("Reset Layout") {
			app.layout_reset_requested = true
		}
		imgui.EndMenu()
	}

	if imgui.BeginMenu("Render") {
		s := ipr_stats(&app.ipr)
		if imgui.MenuItem(s.enabled ? "Pause IPR" : "Resume IPR") {
			ipr_set_enabled(&app.ipr, !s.enabled)
		}
		if imgui.MenuItem("Restart IPR") {
			ipr_invalidate(&app.ipr)
		}
		imgui.Separator()
		imgui.BeginDisabled(true)
		imgui.MenuItem("Render to File...")
		imgui.EndDisabled()
		imgui.EndMenu()
	}

	return
}

// ── status bar ───────────────────────────────────────────────────────────────
//
// Carries the redraw readout that makes the retained-mode behaviour observable:
// if this shows a steady 60/s with the mouse still, something is waking the loop
// that should not be.

draw_status_bar :: proc(app: ^App) {
	viewport := imgui.GetMainViewport()
	height := imgui.GetFrameHeight()

	imgui.SetNextWindowPos({viewport.WorkPos.x, viewport.WorkPos.y + viewport.WorkSize.y - height})
	imgui.SetNextWindowSize({viewport.WorkSize.x, height})

	flags := imgui.WindowFlags {
		.NoTitleBar,
		.NoResize,
		.NoMove,
		.NoScrollbar,
		.NoSavedSettings,
		.NoDocking,
		.NoBringToFrontOnFocus,
		.MenuBar,
	}

	if imgui.Begin("##statusbar", nil, flags) {
		if imgui.BeginMenuBar() {
			scene := app.scene_loaded ? app.scene_path : "no scene loaded"
			imgui.TextUnformatted(tmp_cstring(scene))

			// Right-align the redraw readout.
			readout: string
			if app.idle {
				readout = fmt.tprintf("idle | %d frames", app.redraw_count)
			} else {
				readout = fmt.tprintf("%.0f fps | %d frames", app.redraw_rate, app.redraw_count)
			}
			text_w := imgui.CalcTextSize(tmp_cstring(readout)).x
			imgui.SameLine(imgui.GetWindowWidth() - text_w - imgui.GetStyle().ItemSpacing.x * 3)
			imgui.TextUnformatted(tmp_cstring(readout))

			imgui.EndMenuBar()
		}
	}
	imgui.End()
}

// ── panels ───────────────────────────────────────────────────────────────────

draw_panels :: proc(app: ^App, v: ^Viewport) {
	if app.show_viewport {
		draw_viewport_panel(app, v)
	}

	if app.show_usd_tree {
		draw_usd_tree_panel(app)
	}

	if app.show_usd_text {
		draw_usd_text_panel(app)
	}

	if app.show_properties {
		draw_properties_panel(app)
	}

	if app.show_material {
		draw_material_panel(app)
	}

	if app.show_render {
		draw_render_panel(app)
	}

	if app.show_script {
		if imgui.Begin(WINDOW_SCRIPT, &app.show_script) {
			placeholder("Embedded Python editor", "Phase 6")
		}
		imgui.End()
	}

	if app.show_log {
		draw_log_panel(app)
	}
}

@(private = "file")
placeholder :: proc(what: string, phase: string) {
	imgui.TextDisabled(tmp_cstring(what))
	imgui.TextDisabled(tmp_cstring(fmt.tprintf("(%s)", phase)))
}

// ── log panel ────────────────────────────────────────────────────────────────

@(private = "file")
draw_log_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_LOG, &app.show_log) {
		imgui.End()
		return
	}
	defer imgui.End()

	l := &app.log
	sync.mutex_lock(&l.mutex)
	defer sync.mutex_unlock(&l.mutex)

	if imgui.SmallButton("Clear") {
		for line in l.lines {
			delete(line)
		}
		clear(&l.lines)
	}
	imgui.SameLine()
	imgui.TextDisabled(tmp_cstring(fmt.tprintf("%d lines", len(l.lines))))
	imgui.Separator()

	if imgui.BeginChild("##logscroll", {0, 0}, {}, {.HorizontalScrollbar}) {
		// Only submit the visible slice. With a 4000-line backlog, drawing
		// every line each frame is wasted work even when the log is idle.
		clipper: imgui.ListClipper
		imgui.ListClipper_Begin(&clipper, i32(len(l.lines)))
		for imgui.ListClipper_Step(&clipper) {
			for i in clipper.DisplayStart ..< clipper.DisplayEnd {
				line := l.lines[i]
				// Draw straight out of the stored string: passing an explicit
				// end pointer avoids a per-line, per-frame cstring allocation.
				start := raw_data(line)
				end := rawptr(uintptr(start) + uintptr(len(line)))
				imgui.TextUnformatted(cstring(start), cstring(end))
			}
		}

		if l.scroll_to_end {
			imgui.SetScrollHereY(1.0)
			l.scroll_to_end = false
		}
	}
	imgui.EndChild()

	l.dirty = false
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Temp-allocator cstring for UI text. The temp allocator is freed once per
// frame in the main loop, so these never accumulate.
tmp_cstring :: proc(s: string) -> cstring {
	return strings.clone_to_cstring(s, context.temp_allocator)
}
