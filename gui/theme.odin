package main

// Dark DCC theme for the Lumbre GUI.
//
// A flat, low-contrast professional look: near-square corners, thin dark
// borders, compact spacing, light-grey text on medium-grey panels, with a
// green highlight for focus/selection and orange checkbox ticks. Section
// headers are drawn as recessed darker strips. Paired with the Inter typeface
// for body text and Font Awesome for the tree's prim-type and eye icons.

import imgui "../third_party/odin-imgui"

// Inter is compiled into the binary so the theme never depends on a font file
// being found next to the executable. The bytes are static for the life of the
// program, which is why the atlas is told it does not own them (below).
@(private = "file")
INTER_TTF := #load("../assets/fonts/Inter.ttf")

// Font Awesome 6 Free (Solid) supplies the prim-type and visibility glyphs the
// USD tree draws. Merged into Inter so a single Text() call can mix icons and
// text. Also static; also not owned by the atlas.
@(private = "file")
FA_TTF := #load("../assets/fonts/fa-solid-900.ttf")

// Icon glyphs, by Font Awesome codepoint (Private Use Area). Package-visible so
// the panels can reference them by name rather than by raw escape.
ICON_XFORM     :: "\uf047" // arrows-up-down-left-right
ICON_SCOPE     :: "\uf5fd" // layer-group
ICON_MESH      :: "\uf1b2" // cube
ICON_POINTS    :: "\uf2a1" // braille
ICON_CURVES    :: "\uf55b" // bezier-curve
ICON_CAMERA    :: "\uf030" // camera
ICON_MATERIAL  :: "\uf53f" // palette
ICON_NODEGRAPH :: "\uf542" // diagram-project
ICON_LIGHT     :: "\uf0eb" // lightbulb
ICON_LIGHT_SUN :: "\uf185" // sun (distant/dome)
ICON_SPHERE    :: "\uf111" // circle
ICON_PRIM      :: "\uf111" // circle (generic / untyped)
ICON_EYE       :: "\uf06e" // eye
ICON_EYE_SLASH :: "\uf070" // eye-slash
ICON_SEARCH    :: "\uf002" // magnifying-glass

// f32 max, for FontConfig.GlyphMaxAdvanceX (its documented default is FLT_MAX;
// a zero there would collapse every glyph's advance width).
@(private = "file")
F32_MAX :: f32(0h7f7f_ffff)

// 0-255 sRGB byte triple to ImGui's 0-1 float colour.
@(private = "file")
rgb :: proc(r, g, b: int, a: f32 = 1.0) -> imgui.Vec4 {
	return imgui.Vec4{f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0, a}
}

// A FontConfig with the non-zero defaults ImGui's C++ constructor would set;
// zero-initialising instead collapses glyph metrics.
@(private = "file")
font_config :: proc() -> imgui.FontConfig {
	return imgui.FontConfig {
		FontDataOwnedByAtlas = false, // our #load bytes are static; atlas must not free them
		OversampleH          = 2,
		OversampleV          = 1,
		GlyphMaxAdvanceX     = F32_MAX,
		RasterizerMultiply   = 1.0,
		RasterizerDensity    = 1.0,
		ExtraSizeScale       = 1.0,
	}
}

// Loads Inter, then merges Font Awesome on top, at `size_pixels` (before any DPI
// scale, which the caller applies through style.FontScaleDpi). Glyphs are
// rasterised on demand by the 1.92 dynamic atlas, so no glyph ranges are needed.
// Call once, before the first frame.
theme_load_font :: proc(io: ^imgui.IO, size_pixels: f32) {
	body := font_config()
	body.SizePixels = size_pixels
	imgui.FontAtlas_AddFontFromMemoryTTF(
		io.Fonts, raw_data(INTER_TTF), i32(len(INTER_TTF)), size_pixels, &body,
	)

	icons := font_config()
	icons.MergeMode = true
	icons.PixelSnapH = true
	icons.OversampleH = 1
	icons.SizePixels = size_pixels
	// Nudge icons down a hair so they sit on the text baseline rather than above
	// it, and keep them slightly narrower than the em so lines stay tight.
	icons.GlyphOffset = imgui.Vec2{0, 1}
	icons.GlyphMinAdvanceX = size_pixels
	imgui.FontAtlas_AddFontFromMemoryTTF(
		io.Fonts, raw_data(FA_TTF), i32(len(FA_TTF)), size_pixels, &icons,
	)
}

// Applies the colour palette and metrics. Call before Style_ScaleAllSizes so the
// spacing values below are DPI-scaled along with the rest.
theme_apply :: proc(style: ^imgui.Style) {
	// ── palette ───────────────────────────────────────────────────────────────
	bg_menu    := rgb(43, 43, 43)   // menu/title bars — the darkest chrome
	bg_window  := rgb(56, 56, 56)   // panel background
	bg_child   := rgb(51, 51, 51)
	bg_popup   := rgb(46, 46, 46)
	frame      := rgb(36, 36, 36)   // recessed inputs
	frame_hov  := rgb(46, 46, 46)
	frame_act  := rgb(30, 30, 30)
	// Section header bars (CollapsingHeader) sit *darker* than the panel — a
	// recessed strip rather than a raised one. Shared with selectable rows and
	// table headers, which read fine dark.
	header     := rgb(42, 42, 42)
	header_hov := rgb(52, 52, 52)
	header_act := rgb(36, 36, 36)
	button     := rgb(74, 74, 74)
	button_hov := rgb(88, 88, 88)
	button_act := rgb(60, 60, 60)
	border     := rgb(24, 24, 24)   // thin dark seams between panels
	text       := rgb(202, 202, 202)
	text_dis   := rgb(120, 120, 120)
	grab       := rgb(120, 120, 120)
	scroll_bg  := rgb(40, 40, 40)
	scroll_gb  := rgb(90, 90, 90)

	green      := rgb(140, 185, 44)  // focus / selection / active tab
	green_dim  := rgb(105, 138, 40)
	orange     := rgb(224, 138, 43)  // checkbox ticks

	c := &style.Colors
	c[imgui.Col.Text]                  = text
	c[imgui.Col.TextDisabled]          = text_dis
	c[imgui.Col.WindowBg]              = bg_window
	c[imgui.Col.ChildBg]               = bg_child
	c[imgui.Col.PopupBg]               = bg_popup
	c[imgui.Col.Border]                = border
	c[imgui.Col.BorderShadow]          = imgui.Vec4{0, 0, 0, 0}
	c[imgui.Col.FrameBg]               = frame
	c[imgui.Col.FrameBgHovered]        = frame_hov
	c[imgui.Col.FrameBgActive]         = frame_act
	c[imgui.Col.TitleBg]               = bg_menu
	c[imgui.Col.TitleBgActive]         = bg_menu
	c[imgui.Col.TitleBgCollapsed]      = bg_menu
	c[imgui.Col.MenuBarBg]             = bg_menu
	c[imgui.Col.ScrollbarBg]           = scroll_bg
	c[imgui.Col.ScrollbarGrab]         = scroll_gb
	c[imgui.Col.ScrollbarGrabHovered]  = header_hov
	c[imgui.Col.ScrollbarGrabActive]   = green_dim
	c[imgui.Col.CheckMark]             = orange
	c[imgui.Col.SliderGrab]            = grab
	c[imgui.Col.SliderGrabActive]      = green
	c[imgui.Col.Button]                = button
	c[imgui.Col.ButtonHovered]         = button_hov
	c[imgui.Col.ButtonActive]          = button_act
	c[imgui.Col.Header]                = header
	c[imgui.Col.HeaderHovered]         = header_hov
	c[imgui.Col.HeaderActive]          = header_act
	c[imgui.Col.Separator]             = border
	c[imgui.Col.SeparatorHovered]      = green_dim
	c[imgui.Col.SeparatorActive]       = green
	c[imgui.Col.ResizeGrip]            = header
	c[imgui.Col.ResizeGripHovered]     = header_hov
	c[imgui.Col.ResizeGripActive]      = green
	c[imgui.Col.Tab]                   = bg_menu
	c[imgui.Col.TabHovered]            = header_hov
	c[imgui.Col.TabSelected]           = bg_window
	c[imgui.Col.TabSelectedOverline]   = green
	c[imgui.Col.TabDimmed]             = bg_menu
	c[imgui.Col.TabDimmedSelected]     = bg_child
	c[imgui.Col.TabDimmedSelectedOverline] = green_dim
	c[imgui.Col.DockingPreview]        = green_dim
	c[imgui.Col.DockingEmptyBg]        = rgb(32, 32, 32)
	c[imgui.Col.PlotLines]             = green
	c[imgui.Col.PlotLinesHovered]      = orange
	c[imgui.Col.PlotHistogram]         = green
	c[imgui.Col.PlotHistogramHovered]  = orange
	c[imgui.Col.TableHeaderBg]         = header
	c[imgui.Col.TableBorderStrong]     = border
	c[imgui.Col.TableBorderLight]      = rgb(44, 44, 44)
	c[imgui.Col.TableRowBg]            = imgui.Vec4{0, 0, 0, 0}
	c[imgui.Col.TableRowBgAlt]         = imgui.Vec4{1, 1, 1, 0.02}
	c[imgui.Col.TextLink]              = green
	c[imgui.Col.TextSelectedBg]        = imgui.Vec4{green.x, green.y, green.z, 0.35}
	c[imgui.Col.DragDropTarget]        = green
	c[imgui.Col.NavCursor]             = green
	c[imgui.Col.NavWindowingHighlight] = green
	c[imgui.Col.NavWindowingDimBg]     = imgui.Vec4{0, 0, 0, 0.55}
	c[imgui.Col.ModalWindowDimBg]      = imgui.Vec4{0, 0, 0, 0.55}

	// ── metrics ──────────────────────────────────────────────────────────────
	// Flat and compact: DCC panels are dense, and rounded corners read as toys.
	style.WindowRounding    = 0
	style.ChildRounding     = 0
	style.PopupRounding     = 2
	style.FrameRounding     = 2
	style.ScrollbarRounding = 0
	style.GrabRounding      = 2
	style.TabRounding       = 0

	style.WindowBorderSize   = 1
	style.ChildBorderSize    = 1
	style.PopupBorderSize    = 1
	style.FrameBorderSize    = 1
	style.TabBarBorderSize   = 1
	style.TabBarOverlineSize = 2

	style.WindowPadding    = imgui.Vec2{8, 6}
	style.FramePadding     = imgui.Vec2{6, 3}
	style.CellPadding      = imgui.Vec2{4, 2}
	style.ItemSpacing      = imgui.Vec2{6, 4}
	style.ItemInnerSpacing = imgui.Vec2{4, 3}
	style.IndentSpacing    = 16
	style.ScrollbarSize    = 12
	style.GrabMinSize      = 8

	style.WindowTitleAlign = imgui.Vec2{0.0, 0.5}
	style.WindowMenuButtonPosition = .None
}
