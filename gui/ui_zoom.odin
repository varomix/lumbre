package main

// UI zoom: Cmd/Ctrl with + and - scale the whole interface, 0 resets it, as in
// a browser or editor. Text scales through style.FontScaleMain -- ImGui 1.92
// re-rasterises glyphs at the new size, so zoomed text stays sharp -- and the
// padding, spacing and rounding are re-scaled from the unscaled theme, so the
// layout grows with the text rather than crowding it.

import "core:math"

import imgui "../third_party/odin-imgui"

UI_ZOOM_MIN :: 0.5
UI_ZOOM_MAX :: 3.0
UI_ZOOM_STEP :: 0.1

// Applies a pending zoom change. Call between frames, before imgui.NewFrame:
// the style should not change under a frame that is half built.
ui_zoom_apply :: proc(app: ^App) {
	if app.ui_zoom == app.ui_zoom_applied {
		return
	}
	style := imgui.GetStyle()
	// FontScaleDpi follows the monitor (io.ConfigDpiScaleFonts), so keep
	// whatever it is now rather than the value captured at start-up.
	dpi_fonts := style.FontScaleDpi
	style^ = app.ui_base_style
	imgui.Style_ScaleAllSizes(style, app.ui_dpi_scale * app.ui_zoom)
	style.FontScaleDpi = dpi_fonts
	style.FontScaleMain = app.ui_zoom
	app.ui_zoom_applied = app.ui_zoom
}

// Reads the zoom shortcuts. Takes effect on the next frame.
ui_zoom_shortcuts :: proc(app: ^App) {
	io := imgui.GetIO()
	if !(io.KeyCtrl || io.KeySuper) {
		return
	}
	// `+` is Shift+= on most layouts, so Equal covers both.
	if imgui.IsKeyPressed(.Equal, true) || imgui.IsKeyPressed(.KeypadAdd, true) {
		ui_zoom_step(app, 1)
	} else if imgui.IsKeyPressed(.Minus, true) || imgui.IsKeyPressed(.KeypadSubtract, true) {
		ui_zoom_step(app, -1)
	} else if imgui.IsKeyPressed(._0, false) || imgui.IsKeyPressed(.Keypad0, false) {
		ui_zoom_reset(app)
	}
}

ui_zoom_step :: proc(app: ^App, direction: f32) {
	// Rounded to the step so repeated presses land on 1.0 exactly again.
	z := math.round((app.ui_zoom + direction * UI_ZOOM_STEP) / UI_ZOOM_STEP) * UI_ZOOM_STEP
	app.ui_zoom = clamp(z, UI_ZOOM_MIN, UI_ZOOM_MAX)
	app_wake(app)
}

ui_zoom_reset :: proc(app: ^App) {
	app.ui_zoom = 1
	app_wake(app)
}
