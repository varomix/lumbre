package main

// USD stage inspection for the tree, text, and properties panels.
//
// This opens its own stage handle rather than reusing the importer's. The
// importer opens, reads, and closes; keeping its stage alive would mean
// threading a handle back through `make_scene` for every format. Opening a
// second flattened stage costs one extra parse at load time and keeps the
// import path completely untouched — and since both are flattened, the tree
// matches what the renderer actually got.
//
// Everything here is read-only. Authoring back to the stage is a later phase.

import "core:strings"
import "core:time"

import imp "../importers"

// Array attributes longer than this have their values dropped from the text
// view. A single guitar mesh exports to 33 MB of .usda with its points and
// indices included, which is neither readable nor cheap to produce; the
// structure, metadata and small values are what the panel is for.
USD_TEXT_MAX_ARRAY :: 16

Usd_Node :: struct {
	prim:      imp.Usd_Shim_Prim,
	name:      string,
	type_name: string,
	path:      string,
	parent:    int,
	children:  [dynamic]int,
}

Usd_Stage_View :: struct {
	stage:  imp.Usd_Shim_Stage,
	open:   bool,
	nodes:  [dynamic]Usd_Node,

	// Index into `nodes`, or -1. Drives the text and properties panels.
	selected: int,

	filter_buf: [128]u8,

	// Cached .usda for whichever node the text panel last rendered. Exporting
	// is expensive enough that doing it per frame would be absurd, so it is
	// done on selection change only.
	text:          string,
	text_for:      int,
	text_lines:    [dynamic]string,
	text_is_stage: bool,
	show_whole_stage: bool,

	// Cached property rows for the selection, same reasoning.
	props:     [dynamic]Usd_Property,
	props_for: int,
}

Usd_Property :: struct {
	name:            string,
	type_name:       string,
	value:           string,
	is_relationship: bool,
}

usd_view_close :: proc(v: ^Usd_Stage_View) {
	if v.open {
		imp.usd_shim_close(v.stage)
		v.open = false
	}
	for node in v.nodes {
		delete(node.name)
		delete(node.type_name)
		delete(node.path)
		delete(node.children)
	}
	clear(&v.nodes)
	usd_view_clear_caches(v)
	v.selected = -1
}

@(private = "file")
usd_view_clear_caches :: proc(v: ^Usd_Stage_View) {
	delete(v.text)
	v.text = ""
	delete(v.text_lines)
	v.text_lines = {}
	v.text_for = -2

	for p in v.props {
		delete(p.name)
		delete(p.type_name)
		delete(p.value)
	}
	clear(&v.props)
	v.props_for = -2
}

// Opens `path` for inspection. Non-USD scenes simply leave the panels empty,
// which is the honest thing to show for an .obj.
usd_view_open :: proc(v: ^Usd_Stage_View, app: ^App, path: string) {
	usd_view_close(v)

	lower := strings.to_lower(path, context.temp_allocator)
	is_usd := strings.has_suffix(lower, ".usd") ||
		strings.has_suffix(lower, ".usda") ||
		strings.has_suffix(lower, ".usdc") ||
		strings.has_suffix(lower, ".usdz")
	if !is_usd {
		return
	}

	start := time.tick_now()
	cpath := strings.clone_to_cstring(path, context.temp_allocator)
	err_buf: [512]u8
	stage := imp.usd_shim_open_flattened(cpath, raw_data(err_buf[:]), len(err_buf))
	if stage == nil {
		log_printf(&app.log, "[usd] could not open for inspection: %s", string(cstring(raw_data(err_buf[:]))))
		return
	}

	v.stage = stage
	v.open = true
	v.selected = -1
	usd_view_clear_caches(v)

	root := imp.usd_shim_get_pseudo_root(stage)
	if root != nil {
		usd_view_add_prim(v, root, -1)
	}

	// Land on something meaningful rather than an empty properties panel: the
	// first Mesh if there is one, else the first typed prim.
	for node, i in v.nodes {
		if node.type_name == "Mesh" {
			v.selected = i
			break
		}
	}
	if v.selected < 0 {
		for node, i in v.nodes {
			if node.type_name != "" {
				v.selected = i
				break
			}
		}
	}

	log_printf(
		&app.log,
		"[usd] stage inspected: %d prims [%.3f s]",
		len(v.nodes),
		time.duration_seconds(time.tick_since(start)),
	)
}

// Walks the whole hierarchy up front. Only names, types, and paths are read —
// no mesh data — so this stays cheap even on large stages, and it means the
// tree can be filtered without re-walking.
@(private = "file")
usd_view_add_prim :: proc(v: ^Usd_Stage_View, prim: imp.Usd_Shim_Prim, parent: int) {
	idx := len(v.nodes)
	append(
		&v.nodes,
		Usd_Node {
			prim = prim,
			name = strings.clone(string(imp.usd_shim_prim_name(prim))),
			type_name = strings.clone(string(imp.usd_shim_prim_type_name(prim))),
			path = strings.clone(string(imp.usd_shim_prim_path(prim))),
			parent = parent,
		},
	)
	if parent >= 0 {
		append(&v.nodes[parent].children, idx)
	}

	count := imp.usd_shim_get_children(prim, nil, 0)
	if count <= 0 {
		return
	}
	kids := make([]imp.Usd_Shim_Prim, count, context.temp_allocator)
	got := imp.usd_shim_get_children(prim, raw_data(kids), i32(count))
	for i in 0 ..< int(got) {
		usd_view_add_prim(v, kids[i], idx)
	}
}

// ── cached exports ───────────────────────────────────────────────────────────

usd_view_ensure_text :: proc(v: ^Usd_Stage_View) {
	if !v.open {
		return
	}
	want := v.show_whole_stage ? -1 : v.selected
	if v.text_for == want && v.text_is_stage == v.show_whole_stage {
		return
	}

	delete(v.text)
	v.text = ""
	delete(v.text_lines)
	v.text_lines = {}
	v.text_for = want
	v.text_is_stage = v.show_whole_stage

	raw: [^]u8
	if v.show_whole_stage {
		raw = imp.usd_shim_export_stage_to_string(v.stage, USD_TEXT_MAX_ARRAY)
	} else if v.selected >= 0 && v.selected < len(v.nodes) {
		raw = imp.usd_shim_export_prim_to_string(v.stage, v.nodes[v.selected].prim, USD_TEXT_MAX_ARRAY)
	}
	if raw == nil {
		return
	}
	defer imp.usd_shim_free_string(raw)

	v.text = strings.clone(string(cstring(raw)))
	// Split once, not per frame: the text panel draws through a clipper and
	// needs random access to lines.
	v.text_lines = make([dynamic]string)
	for line in strings.split_lines_iterator(&(&v.text)^) {
		append(&v.text_lines, line)
	}
}

usd_view_ensure_props :: proc(v: ^Usd_Stage_View) {
	if !v.open || v.props_for == v.selected {
		return
	}
	for p in v.props {
		delete(p.name)
		delete(p.type_name)
		delete(p.value)
	}
	clear(&v.props)
	v.props_for = v.selected

	if v.selected < 0 || v.selected >= len(v.nodes) {
		return
	}
	prim := v.nodes[v.selected].prim
	count := imp.usd_shim_get_property_count(prim)
	if count <= 0 {
		return
	}

	raw := make([]imp.Usd_Shim_Property, count, context.temp_allocator)
	got := imp.usd_shim_get_properties(prim, raw_data(raw), i32(count))
	for i in 0 ..< int(got) {
		append(
			&v.props,
			Usd_Property {
				name = strings.clone(string(raw[i].name)),
				type_name = strings.clone(string(raw[i].type_name)),
				value = strings.clone(string(raw[i].value)),
				is_relationship = raw[i].is_relationship != 0,
			},
		)
	}
}
