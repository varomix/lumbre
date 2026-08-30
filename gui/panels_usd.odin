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

	imgui.SetNextItemWidth(-1)
	imgui.InputTextWithHint("##usdfilter", "filter by name, type or path", cstring(raw_data(v.filter_buf[:])), len(v.filter_buf))
	filter := strings.trim_space(string(cstring(raw_data(v.filter_buf[:]))))
	imgui.Separator()

	if imgui.BeginChild("##usdtree", {0, 0}, {}, {.HorizontalScrollbar}) {
		if len(v.nodes) > 0 {
			// Node 0 is the pseudo-root ("/"); show its children as top level.
			for child in v.nodes[0].children {
				usd_tree_node(app, v, child, filter)
			}
		}
	}
	imgui.EndChild()
}

@(private = "file")
usd_tree_node :: proc(app: ^App, v: ^Usd_Stage_View, idx: int, filter: string) {
	node := &v.nodes[idx]

	if filter != "" && !usd_subtree_matches(v, idx, filter) {
		return
	}

	flags := imgui.TreeNodeFlags{.OpenOnArrow, .SpanAvailWidth, .DefaultOpen}
	if len(node.children) == 0 {
		flags += {.Leaf, .NoTreePushOnOpen}
	}
	if v.selected == idx {
		flags += {.Selected}
	}

	label := fmt.tprintf("%s##%d", node.name == "" ? "/" : node.name, idx)
	opened := imgui.TreeNodeEx(tmp_cstring(label), flags)

	if imgui.IsItemClicked() {
		v.selected = idx
	}

	if node.type_name != "" {
		imgui.SameLine()
		col := usd_type_colour(node.type_name)
		imgui.TextColored(col, tmp_cstring(node.type_name))
	}

	if len(node.children) > 0 && opened {
		for child in node.children {
			usd_tree_node(app, v, child, filter)
		}
		imgui.TreePop()
	}
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
