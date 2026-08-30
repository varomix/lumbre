package main

// USD tree, text, and properties panels.
//
// All three read from `App.usd`, which caches on selection change — a panel
// never re-queries USD just because it was redrawn.

import "core:fmt"
import "core:strings"

import imgui "../third_party/odin-imgui"

// ── tree ─────────────────────────────────────────────────────────────────────

draw_usd_tree_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_USD_TREE, &app.show_usd_tree) {
		imgui.End()
		return
	}
	defer imgui.End()

	v := &app.usd
	if !v.open {
		if app.scene_loaded {
			imgui.TextDisabled("Scene is not USD — no stage to inspect")
		} else {
			imgui.TextDisabled("Open a scene (File > Open Scene)")
		}
		return
	}

	// Search row: a magnifying-glass icon then a full-width filter box.
	imgui.AlignTextToFramePadding()
	imgui.TextUnformatted(ICON_SEARCH)
	imgui.SameLine()
	imgui.SetNextItemWidth(-1)
	imgui.InputTextWithHint("##usdfilter", "Search", cstring(raw_data(v.filter_buf[:])), len(v.filter_buf))
	filter := strings.trim_space(string(cstring(raw_data(v.filter_buf[:]))))

	// Name | visibility eye | Type, in a scrolling table so the columns line up
	// down the hierarchy the way a stage view does.
	flags :=
		imgui.TableFlags_RowBg |
		imgui.TableFlags_ScrollY |
		imgui.TableFlags_Resizable |
		imgui.TableFlags_BordersInnerV |
		imgui.TableFlags_NoBordersInBody |
		imgui.TableFlags_SizingStretchProp
	if imgui.BeginTable("##usdtree", 3, flags) {
		imgui.TableSetupColumn("Name", {.WidthStretch, .NoHide}, 0.66)
		// The eye column carries the eye glyph as its own header, and stays narrow.
		imgui.TableSetupColumn(ICON_EYE, {.WidthFixed, .NoResize}, 0)
		imgui.TableSetupColumn("Type", {.WidthStretch}, 0.34)
		imgui.TableSetupScrollFreeze(0, 1)
		imgui.TableHeadersRow()

		if len(v.nodes) > 0 {
			// Node 0 is the pseudo-root ("/"); show its children as top level.
			for child in v.nodes[0].children {
				usd_tree_row(app, v, child, filter)
			}
		}
		imgui.EndTable()
	}
}

@(private = "file")
usd_tree_row :: proc(app: ^App, v: ^Usd_Stage_View, idx: int, filter: string) {
	node := &v.nodes[idx]

	if filter != "" && !usd_subtree_matches(v, idx, filter) {
		return
	}

	imgui.TableNextRow()
	imgui.TableSetColumnIndex(0)

	is_leaf := len(node.children) == 0
	flags := imgui.TreeNodeFlags{.OpenOnArrow, .SpanAllColumns, .DefaultOpen}
	if is_leaf {
		flags += {.Leaf, .NoTreePushOnOpen}
	}
	if v.selected == idx {
		flags += {.Selected}
	}

	// The tree node spans the whole row for the selection highlight; AllowOverlap
	// lets the eye button in the next column still take its own clicks.
	imgui.SetNextItemAllowOverlap()
	opened := imgui.TreeNodeEx(tmp_cstring(fmt.tprintf("###n%d", idx)), flags)
	if imgui.IsItemClicked() {
		v.selected = idx
	}

	// Prim-type icon (coloured) then the name, drawn over the spanning node.
	imgui.SameLine()
	imgui.PushStyleColorImVec4(.Text, usd_type_colour(node.type_name))
	imgui.TextUnformatted(tmp_cstring(usd_type_icon(node.type_name)))
	imgui.PopStyleColor()
	imgui.SameLine()
	imgui.TextUnformatted(tmp_cstring(node.name == "" ? "/" : node.name))

	// Visibility eye. Click toggles the UI flag (see Usd_Node.visible).
	imgui.TableSetColumnIndex(1)
	if node.visible {
		imgui.TextUnformatted(ICON_EYE)
	} else {
		imgui.TextDisabled(ICON_EYE_SLASH)
	}
	if imgui.IsItemHovered() {
		imgui.SetMouseCursor(.Hand)
	}
	if imgui.IsItemClicked() {
		node.visible = !node.visible
	}

	// Type name, colour-coded to match its icon.
	imgui.TableSetColumnIndex(2)
	if node.type_name != "" {
		imgui.TextColored(usd_type_colour(node.type_name), tmp_cstring(node.type_name))
	} else {
		imgui.TextDisabled("-")
	}

	if opened && !is_leaf {
		for child in node.children {
			usd_tree_row(app, v, child, filter)
		}
		imgui.TreePop()
	}
}

// Font Awesome glyph for a prim type; see the ICON_* constants in theme.odin.
@(private = "file")
usd_type_icon :: proc(type_name: string) -> string {
	switch type_name {
	case "Mesh", "GeomSubset", "Cube", "Cone", "Cylinder", "Capsule":
		return ICON_MESH
	case "Xform":
		return ICON_XFORM
	case "Scope":
		return ICON_SCOPE
	case "Points", "PointInstancer":
		return ICON_POINTS
	case "BasisCurves", "NurbsCurves":
		return ICON_CURVES
	case "Camera":
		return ICON_CAMERA
	case "Material", "Shader":
		return ICON_MATERIAL
	case "NodeGraph":
		return ICON_NODEGRAPH
	case "Sphere":
		return ICON_SPHERE
	case "SphereLight", "RectLight", "DiskLight", "CylinderLight":
		return ICON_LIGHT
	case "DistantLight", "DomeLight":
		return ICON_LIGHT_SUN
	}
	return ICON_PRIM
}

// A node stays visible when it or anything beneath it matches, so filtering
// never hides the path to a hit.
@(private = "file")
usd_subtree_matches :: proc(v: ^Usd_Stage_View, idx: int, filter: string) -> bool {
	node := &v.nodes[idx]
	if usd_text_contains_fold(node.name, filter) ||
	   usd_text_contains_fold(node.type_name, filter) ||
	   usd_text_contains_fold(node.path, filter) {
		return true
	}
	for child in node.children {
		if usd_subtree_matches(v, child, filter) {
			return true
		}
	}
	return false
}

@(private = "file")
usd_text_contains_fold :: proc(haystack, needle: string) -> bool {
	h := strings.to_lower(haystack, context.temp_allocator)
	n := strings.to_lower(needle, context.temp_allocator)
	return strings.contains(h, n)
}

// Colour by prim type, so hierarchy shape is readable at a glance.
@(private = "file")
usd_type_colour :: proc(type_name: string) -> imgui.Vec4 {
	switch type_name {
	case "Mesh", "GeomSubset":
		return {0.55, 0.75, 1.00, 1}
	case "Xform", "Scope":
		return {0.60, 0.60, 0.65, 1}
	case "Camera":
		return {1.00, 0.80, 0.45, 1}
	case "Material", "Shader", "NodeGraph":
		return {0.80, 0.60, 1.00, 1}
	case "SphereLight", "RectLight", "DiskLight", "DistantLight", "DomeLight", "CylinderLight":
		return {1.00, 0.95, 0.55, 1}
	}
	return {0.55, 0.85, 0.65, 1}
}

// ── text ─────────────────────────────────────────────────────────────────────

draw_usd_text_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_USD_TEXT, &app.show_usd_text) {
		imgui.End()
		return
	}
	defer imgui.End()

	v := &app.usd
	if !v.open {
		imgui.TextDisabled("No USD stage loaded")
		return
	}

	imgui.Checkbox("Whole stage", &v.show_whole_stage)
	imgui.SameLine()
	if imgui.SmallButton("Copy") {
		imgui.SetClipboardText(tmp_cstring(v.text))
	}
	imgui.SameLine()
	if v.show_whole_stage {
		imgui.TextDisabled("(entire flattened layer)")
	} else if v.selected >= 0 {
		imgui.TextDisabled(tmp_cstring(v.nodes[v.selected].path))
	} else {
		imgui.TextDisabled("(select a prim)")
	}
	imgui.Separator()

	usd_view_ensure_text(v)

	// Say so rather than silently showing a mesh with no points.
	imgui.TextDisabled(tmp_cstring(fmt.tprintf(
		"arrays longer than %d elided  |  %d lines", USD_TEXT_MAX_ARRAY, len(v.text_lines))))

	if len(v.text_lines) == 0 {
		imgui.TextDisabled("Nothing to show")
		return
	}

	if imgui.BeginChild("##usdtext", {0, 0}, {}, {.HorizontalScrollbar}) {
		clipper: imgui.ListClipper
		imgui.ListClipper_Begin(&clipper, i32(len(v.text_lines)))
		for imgui.ListClipper_Step(&clipper) {
			for i in clipper.DisplayStart ..< clipper.DisplayEnd {
				line := v.text_lines[i]
				imgui.TextDisabled(tmp_cstring(fmt.tprintf("%5d", i + 1)))
				imgui.SameLine()
				imgui.TextColored(usda_line_colour(line), tmp_cstring(line))
			}
		}
	}
	imgui.EndChild()
}

// Deliberately line-granular rather than a real tokeniser: enough to make the
// structure of a .usda readable without carrying a lexer around.
@(private = "file")
usda_line_colour :: proc(line: string) -> imgui.Vec4 {
	t := strings.trim_left_space(line)
	switch {
	case strings.has_prefix(t, "#"):
		return {0.50, 0.55, 0.50, 1}
	case strings.has_prefix(t, "def "),
	     strings.has_prefix(t, "over "),
	     strings.has_prefix(t, "class "):
		return {0.55, 0.75, 1.00, 1}
	case strings.has_prefix(t, "uniform "), strings.has_prefix(t, "rel "), strings.has_prefix(t, "custom "):
		return {0.80, 0.60, 1.00, 1}
	case strings.has_prefix(t, "("), strings.has_prefix(t, ")"):
		return {0.60, 0.60, 0.65, 1}
	}
	return {0.85, 0.85, 0.88, 1}
}

// ── properties ───────────────────────────────────────────────────────────────

draw_properties_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_PROPERTIES, &app.show_properties) {
		imgui.End()
		return
	}
	defer imgui.End()

	v := &app.usd
	if !v.open {
		imgui.TextDisabled("No USD stage loaded")
		return
	}
	if v.selected < 0 || v.selected >= len(v.nodes) {
		imgui.TextDisabled("Select a prim in the USD Tree")
		return
	}

	node := &v.nodes[v.selected]
	imgui.TextUnformatted(tmp_cstring(node.path))
	imgui.TextDisabled(tmp_cstring(node.type_name == "" ? "(no type)" : node.type_name))
	imgui.Separator()

	usd_view_ensure_props(v)

	if len(v.props) == 0 {
		imgui.TextDisabled("No authored properties")
		return
	}

	// Note TableFlags is a distinct i32 of named constants while
	// TableColumnFlags below is a bit_set — the generator is inconsistent here.
	flags :=
		imgui.TableFlags_Borders |
		imgui.TableFlags_RowBg |
		imgui.TableFlags_Resizable |
		imgui.TableFlags_ScrollY |
		imgui.TableFlags_SizingStretchProp
	if imgui.BeginTable("##props", 3, flags) {
		imgui.TableSetupColumn("Property", {.WidthStretch}, 0.34)
		imgui.TableSetupColumn("Type", {.WidthStretch}, 0.22)
		imgui.TableSetupColumn("Value", {.WidthStretch}, 0.44)
		imgui.TableSetupScrollFreeze(0, 1)
		imgui.TableHeadersRow()

		for p in v.props {
			imgui.TableNextRow()

			imgui.TableSetColumnIndex(0)
			imgui.TextUnformatted(tmp_cstring(p.name))

			imgui.TableSetColumnIndex(1)
			if p.is_relationship {
				imgui.TextColored({0.80, 0.60, 1.00, 1}, "rel")
			} else {
				imgui.TextDisabled(tmp_cstring(p.type_name))
			}

			imgui.TableSetColumnIndex(2)
			if p.value == "" {
				imgui.TextDisabled("-")
			} else {
				imgui.TextUnformatted(tmp_cstring(p.value))
			}
		}
		imgui.EndTable()
	}
}
