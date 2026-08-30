package main

// Material editor and render settings.
//
// Both edit live state and restart accumulation. Material edits deliberately do
// *not* bump the scene key: the worker rewrites only the material buffer, so a
// slider drag costs a batch rather than a full scene rebuild.
//
// Edits are reported at two levels, because the bounce-lighting caches are far
// more expensive to replace than the material buffer. Every frame of a drag
// reports `changed`, which is just that buffer rewrite; releasing the widget
// reports `settled` as well, which is what lets the worker re-emit the photon
// map and irradiance cache. Reporting `settled` on every frame instead put a
// blocking 200k-photon GPU rebuild between each pair of 2-spp batches, which is
// what made dragging a colour crawl.

import "core:fmt"
import "core:strings"

import lc "../core"
import imgui "../third_party/odin-imgui"

// ── material ─────────────────────────────────────────────────────────────────

draw_material_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_MATERIAL, &app.show_material) {
		imgui.End()
		return
	}
	defer imgui.End()

	if !app.scene_loaded || len(app.core.scene.materials) == 0 {
		imgui.TextDisabled("No materials — open a scene")
		return
	}

	mats := app.core.scene.materials
	if app.selected_material >= len(mats) {
		app.selected_material = 0
	}

	imgui.SetNextItemWidth(-1)
	preview := fmt.tprintf("[%d] %s", app.selected_material, material_kind_name(mats[app.selected_material].kind))
	if imgui.BeginCombo("##matsel", tmp_cstring(preview)) {
		for i in 0 ..< len(mats) {
			label := fmt.tprintf("[%d] %s##m%d", i, material_kind_name(mats[i].kind), i)
			if imgui.Selectable(tmp_cstring(label), i == app.selected_material) {
				app.selected_material = i
			}
		}
		imgui.EndCombo()
	}
	imgui.Separator()

	m := &mats[app.selected_material]
	e: Edit_Flags

	// Kind
	kinds := [?]cstring{"Lambertian", "Metal", "Dielectric", "Principled", "Emissive"}
	kind_idx := i32(m.kind)
	imgui.SetNextItemWidth(-1)
	if imgui.ComboChar("Kind", &kind_idx, raw_data(kinds[:]), len(kinds)) {
		// A discrete pick, not a drag: it is settled the moment it happens.
		m.kind = lc.Material_Kind(kind_idx)
		e.changed = true
		e.settled = true
	}

	colour3(&e, "Base Colour", &m.albedo)
	slider(&e, "Roughness", &m.roughness, 0, 1)
	slider(&e, "Metallic", &m.metallic, 0, 1)
	slider(&e, "Specular", &m.specular, 0, 2)
	colour3(&e, "Specular Tint", &m.specular_tint)
	slider(&e, "IOR", &m.ir, 1, 3)

	if imgui.CollapsingHeader("Clearcoat") {
		slider(&e, "Clearcoat", &m.clearcoat, 0, 1)
		slider(&e, "Clearcoat Roughness", &m.clearcoat_roughness, 0, 1)
	}
	if imgui.CollapsingHeader("Sheen") {
		slider(&e, "Sheen", &m.sheen, 0, 1)
		colour3(&e, "Sheen Tint", &m.sheen_tint)
	}
	if imgui.CollapsingHeader("Anisotropy / Transmission") {
		slider(&e, "Anisotropic", &m.anisotropic, 0, 1)
		slider(&e, "Transmission", &m.spec_trans, 0, 1)
	}
	if imgui.CollapsingHeader("Subsurface") {
		slider(&e, "Subsurface", &m.subsurface, 0, 1)
		colour3(&e, "SSS Colour", &m.subsurface_color)
		colour3(&e, "SSS Radius", &m.subsurface_radius)
		slider(&e, "SSS Scale", &m.subsurface_scale, 0, 10)
	}
	if imgui.CollapsingHeader("Emission") {
		colour3(&e, "Emission", &m.emission)
		slider(&e, "Emission Strength", &m.emission_strength, 0, 100)
	}

	imgui.Separator()
	imgui.TextDisabled(tmp_cstring(texture_summary(m^)))
	// Emissive materials feed the light lists, which are baked with the
	// geometry, so say plainly what a change here will and will not do.
	if m.kind == .Emissive {
		imgui.TextDisabled("emissive: reload the scene to relight from this")
	}

	// `settled` alone still reports: releasing a slider is the frame that makes
	// the drag's final value worth re-lighting, and it carries no `changed`.
	if e.changed || e.settled {
		ipr_materials_changed(&app.ipr, e.settled)
	}
}

@(private = "file")
material_kind_name :: proc(k: lc.Material_Kind) -> string {
	switch k {
	case .Lambertian: return "Lambertian"
	case .Metal:      return "Metal"
	case .Dielectric: return "Dielectric"
	case .Principled: return "Principled"
	case .Emissive:   return "Emissive"
	}
	return "?"
}

@(private = "file")
texture_summary :: proc(m: lc.Material) -> string {
	b := strings.builder_make(context.temp_allocator)
	strings.write_string(&b, "maps: ")
	any := false
	for pair in ([?]struct{name: string, tex: lc.TextureMap}{
		{"base", m.albedo_tex},
		{"rough/metal", m.metallic_roughness_tex},
		{"normal", m.normal_tex},
		{"emissive", m.emissive_tex},
	}) {
		if pair.tex.has_data {
			if any { strings.write_string(&b, ", ") }
			strings.write_string(&b, fmt.tprintf("%s %dx%d", pair.name, pair.tex.width, pair.tex.height))
			any = true
		}
	}
	if !any {
		strings.write_string(&b, "none")
	}
	return strings.to_string(b)
}

// ── shared widgets ───────────────────────────────────────────────────────────
//
// The renderer stores shading parameters as f64 while ImGui edits f32, so these
// round-trip explicitly rather than scattering casts through the panel.

// What a run of widgets did this frame. `changed` means a value moved;
// `settled` means the user let go of one, which is the frame the expensive
// caches may be rebuilt on. A widget that is released without having moved
// reports neither.
Edit_Flags :: struct {
	changed: bool,
	settled: bool,
}

// IsItemDeactivatedAfterEdit must be asked immediately after the widget, and
// asked whether or not the widget reported a change: the release frame of a
// drag reports no change but is precisely the frame that settles the edit.
@(private = "file")
slider :: proc(e: ^Edit_Flags, label: cstring, value: ^f64, lo, hi: f32) {
	v := f32(value^)
	if imgui.SliderFloat(label, &v, lo, hi) {
		value^ = f64(v)
		e.changed = true
	}
	e.settled |= imgui.IsItemDeactivatedAfterEdit()
}

@(private = "file")
colour3 :: proc(e: ^Edit_Flags, label: cstring, value: ^lc.Color) {
	v := [3]f32{f32(value.x), f32(value.y), f32(value.z)}
	if imgui.ColorEdit3(label, &v, {.Float, .HDR}) {
		value^ = lc.Color{f64(v[0]), f64(v[1]), f64(v[2])}
		e.changed = true
	}
	e.settled |= imgui.IsItemDeactivatedAfterEdit()
}

// ── render settings ──────────────────────────────────────────────────────────

draw_render_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_RENDER, &app.show_render) {
		imgui.End()
		return
	}
	defer imgui.End()

	s := ipr_stats(&app.ipr)
	imgui.TextDisabled(tmp_cstring(fmt.tprintf("%d / %d spp  |  %.0f ms/batch", s.spp, s.target, s.batch_ms)))
	imgui.Separator()

	restart := false

	if imgui.CollapsingHeader("Sampling", {.DefaultOpen}) {
		target := app.ipr.target_spp
		imgui.SetNextItemWidth(-100)
		if imgui.DragInt("Target spp", &target, 4, 1, 8192) {
			// Raising the target simply lets the worker keep going; it does not
			// invalidate what has already accumulated.
			ipr_set_target_spp(&app.ipr, target)
		}

		depth := app.ipr_max_depth
		imgui.SetNextItemWidth(-100)
		if imgui.DragInt("Max depth", &depth, 1, 1, 64) {
			app.ipr_max_depth = depth
			restart = true
		}
	}

	if imgui.CollapsingHeader("Bias", {.DefaultOpen}) {
		bias: Edit_Flags
		imgui.SetNextItemWidth(-100)
		slider(&bias, "Roughness cutoff", &app.core.settings.roughness_cutoff, 0, 1)
		imgui.SetNextItemWidth(-100)
		slider(&bias, "Glossy bias", &app.core.settings.glossy_bias, 0, 1)
		// Only the accumulated image is invalidated here, which is cheap, so
		// these follow the drag rather than waiting for it to settle.
		restart |= bias.changed
	}

	if imgui.CollapsingHeader("Biased GI", {.DefaultOpen}) {
		// These describe how the bounce-lighting caches are built, so what is
		// already in them no longer answers the question being asked.
		regather := false

		gi := bool(app.core.settings.gi_cache_enabled)
		if imgui.Checkbox("Irradiance cache", &gi) {
			app.core.settings.gi_cache_enabled = b32(gi)
			regather = true
		}
		ph := bool(app.core.settings.photon_enabled)
		if imgui.Checkbox("Photon map", &ph) {
			app.core.settings.photon_enabled = b32(ph)
			regather = true
		}
		count := app.core.settings.photon_count
		imgui.SetNextItemWidth(-100)
		if imgui.DragInt("Photons", &count, 5000, 0, 1048576) {
			// A different photon count resizes the buffers, so this one goes
			// all the way through a cache rebuild in gpu_scene_cache_ensure.
			app.core.settings.photon_count = count
			restart = true
		}
		bounces := app.core.settings.photon_bounces
		imgui.SetNextItemWidth(-100)
		if imgui.DragInt("Photon bounces", &bounces, 1, 1, 32) {
			app.core.settings.photon_bounces = bounces
			regather = true
		}

		if regather {
			ipr_gi_changed(&app.ipr)
			restart = true
		}
	}

	if imgui.CollapsingHeader("Debug") {
		// These modes are already implemented in the kernel; exposing them is
		// free and they are the fastest way to inspect a shading problem.
		modes := [?]cstring{
			"Beauty", "Albedo", "Normal", "Depth", "Primitive ID", "Direct",
			"Light count", "Direct candidates", "Shadow visibility", "Indirect",
			"GI cache hits", "Photon contribution", "GI cache samples",
			"GI cache confidence", "UV", "Albedo texture", "Roughness/Metallic",
		}
		mode := app.core.settings.debug_mode
		imgui.SetNextItemWidth(-100)
		if imgui.ComboChar("Mode", &mode, raw_data(modes[:]), len(modes)) {
			app.core.settings.debug_mode = mode
			restart = true
		}
	}

	draw_output_section(app)

	imgui.Separator()
	if imgui.Button("Restart viewport") {
		restart = true
	}

	if restart {
		ipr_settings_changed(app)
	}
}

// ── output / render to file ──────────────────────────────────────────────────

draw_output_section :: proc(app: ^App) {
	if !imgui.CollapsingHeader("Output", {.DefaultOpen}) {
		return
	}

	running, progress, status, elapsed := render_job_state(&app.render_job)

	imgui.BeginDisabled(running)
	imgui.SetNextItemWidth(-100)
	res := [2]i32{app.out_width, app.out_height}
	if imgui.DragInt2("Resolution", &res, 8, 16, 16384) {
		app.out_width = max(res[0], 16)
		app.out_height = max(res[1], 16)
	}
	imgui.SetNextItemWidth(-100)
	imgui.DragInt("Samples", &app.out_spp, 8, 1, 65536)
	imgui.SetNextItemWidth(-100)
	imgui.DragInt("Depth", &app.out_max_depth, 1, 1, 64)

	imgui.Checkbox("AOVs (EXR only)", &app.out_aovs)
	imgui.SameLine()
	imgui.Checkbox("Denoise", &app.out_denoise)
	imgui.Checkbox("ZIP compress EXR", &app.out_compress)

	imgui.SetNextItemWidth(-100)
	imgui.InputText("Path", cstring(raw_data(app.out_path[:])), len(app.out_path))
	imgui.TextDisabled(".exr writes linear beauty plus AOVs; .png writes 8-bit sRGB")
	imgui.EndDisabled()

	if running {
		imgui.ProgressBar(f32(progress), {-1, 0}, tmp_cstring(status))
		if imgui.Button("Cancel") {
			render_job_cancel(&app.render_job)
		}
	} else {
		if imgui.Button("Render to File") {
			if !render_job_start(app) {
				log_line(&app.log, "[render] a render is already running")
			}
		}
		if status != "" && status != "starting" {
			imgui.TextDisabled(tmp_cstring(fmt.tprintf("%s  [%.1f s]", status, elapsed)))
		}
	}
}
