package main

// Default dock layout.
//
// ┌──────────┬───────────────────────┬────────────┐
// │ USD Tree │      Viewport         │ Properties │
// │          │                       │ Material   │
// ├──────────┼───────────────────────┤            │
// │ USD Text │  Script Editor / Log  │ Render     │
// └──────────┴───────────────────────┴────────────┘
//
// Built once, when there is no imgui.ini to restore from, and again on demand
// from View ▸ Reset Layout. After that the user's arrangement is authoritative
// and is persisted by ImGui itself.

import imgui "../third_party/odin-imgui"

// `ImGuiDockNodeFlags_DockSpace` (imgui_internal.h, 1 << 10) marks a node as a
// dockspace rather than a floating node. It lives in the private flags enum, so
// the generated bindings expose no name for it — hence reconstructing bit 10 of
// the same bit_set by hand.
@(private = "file")
docknode_flag_dockspace :: proc() -> imgui.DockNodeFlags {
	return transmute(imgui.DockNodeFlags)i32(1 << 10)
}

WINDOW_VIEWPORT   :: "Viewport"
WINDOW_USD_TREE   :: "USD Tree"
WINDOW_USD_TEXT   :: "USD Text"
WINDOW_PROPERTIES :: "Properties"
WINDOW_MATERIAL   :: "Material"
WINDOW_RENDER     :: "Render Settings"
WINDOW_SCRIPT     :: "Script Editor"
WINDOW_LOG        :: "Log"

layout_build_default :: proc(dockspace_id: imgui.ID) {
	viewport := imgui.GetMainViewport()

	imgui.DockBuilderRemoveNode(dockspace_id)
	imgui.DockBuilderAddNode(dockspace_id, docknode_flag_dockspace())
	// Size must be set before splitting, or the split ratios are computed
	// against a zero-sized node and come out wrong.
	imgui.DockBuilderSetNodeSize(dockspace_id, viewport.Size)

	left_id, centre_id, right_id: imgui.ID
	imgui.DockBuilderSplitNode(dockspace_id, .Left, 0.20, &left_id, &centre_id)
	imgui.DockBuilderSplitNode(centre_id, .Right, 0.28, &right_id, &centre_id)

	left_top_id, left_bottom_id: imgui.ID
	imgui.DockBuilderSplitNode(left_id, .Up, 0.55, &left_top_id, &left_bottom_id)

	centre_top_id, centre_bottom_id: imgui.ID
	imgui.DockBuilderSplitNode(centre_id, .Up, 0.68, &centre_top_id, &centre_bottom_id)

	right_top_id, right_bottom_id: imgui.ID
	imgui.DockBuilderSplitNode(right_id, .Up, 0.62, &right_top_id, &right_bottom_id)

	imgui.DockBuilderDockWindow(WINDOW_USD_TREE, left_top_id)
	imgui.DockBuilderDockWindow(WINDOW_USD_TEXT, left_bottom_id)

	imgui.DockBuilderDockWindow(WINDOW_VIEWPORT, centre_top_id)
	imgui.DockBuilderDockWindow(WINDOW_SCRIPT, centre_bottom_id)
	imgui.DockBuilderDockWindow(WINDOW_LOG, centre_bottom_id)

	imgui.DockBuilderDockWindow(WINDOW_PROPERTIES, right_top_id)
	imgui.DockBuilderDockWindow(WINDOW_MATERIAL, right_top_id)
	imgui.DockBuilderDockWindow(WINDOW_RENDER, right_bottom_id)

	imgui.DockBuilderFinish(dockspace_id)
}
