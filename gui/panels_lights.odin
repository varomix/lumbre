package main

// Lights panel.
//
// Lights are baked into the GPU scene cache, so an edit takes the same route
// materials do: rewrite just the light buffers rather than rebuilding geometry.
// Changing a light's *type*, though, changes the per-kind buffer sizes, so that
// one case does need a full rebuild — the panel says so instead of silently
// costing half a second per click.

import "core:fmt"
import "core:math"

import lc "../core"
import imgui "../third_party/odin-imgui"

draw_lights_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_LIGHTS, &app.show_lights) {
		imgui.End()
		return
	}
	defer imgui.End()

	if !app.scene_loaded {
		imgui.TextDisabled("Open a scene")
		return
	}

	lights := app.core.scene.lights
	env := &app.core.scene.environment

	// The dome/HDRI is not in the light list — it lives on the scene's
	// environment — but it is a light as far as anyone using this panel is
	// concerned.
	if imgui.CollapsingHeader("Environment (dome)", {.DefaultOpen}) {
		if env.has_data {
			imgui.TextDisabled(tmp_cstring(fmt.tprintf("HDRI %d x %d", env.width, env.height)))
			changed := false
			imgui.SetNextItemWidth(-120)
			if slider_f64("Intensity", &env.intensity, 0, 10) { changed = true }
			imgui.SetNextItemWidth(-120)
			if slider_f64("Rotation", &env.rotation, 0, 360) { changed = true }
			if changed {
				// The environment is uploaded with the scene, and its importance
				// tables are built at the same time, so this needs a rebuild.
				ipr_scene_changed(app)
			}
		} else {
			imgui.TextDisabled("no HDRI loaded")
		}
	}

	imgui.Separator()

	if len(lights) == 0 {
		imgui.TextDisabled("No analytic lights in this scene")
		imgui.TextDisabled("(emissive geometry lights the scene instead)")
		return
	}

	imgui.TextDisabled(tmp_cstring(fmt.tprintf("%d lights", len(lights))))

	for i in 0 ..< len(lights) {
		l := &lights[i]
		label := fmt.tprintf("[%d] %s##light%d", i, light_kind_name(l.kind), i)
		if !imgui.CollapsingHeader(tmp_cstring(label), i == 0 ? imgui.TreeNodeFlags{.DefaultOpen} : {}) {
			continue
		}

		imgui.PushIDInt(i32(i))
		defer imgui.PopID()

		changed := false
		needs_rebuild := false

		kinds := [?]cstring{"Quad", "Sphere", "Mesh", "Disc", "Cylinder", "Point", "Spot", "Distant", "Dome"}
		kind_idx := i32(l.kind)
		imgui.SetNextItemWidth(-120)
		if imgui.ComboChar("Type", &kind_idx, raw_data(kinds[:]), len(kinds)) {
			l.kind = lc.Light_Kind(kind_idx)
			// Per-kind buffers are sized at build time, so this one is not a
			// cheap in-place update.
			needs_rebuild = true
		}

		imgui.SetNextItemWidth(-120)
		if colour3_hdr("Intensity", &l.intensity) { changed = true }

		#partial switch l.kind {
		case .Quad:
			if vec3_f64("Corner", &l.position) { changed = true }
			if vec3_f64("Edge U", &l.u) { changed = true }
			if vec3_f64("Edge V", &l.v) { changed = true }
			if imgui.Checkbox("Two sided", &l.two_sided) { changed = true }
		case .Sphere:
			if vec3_f64("Centre", &l.position) { changed = true }
			if slider_f64("Radius", &l.radius, 0.001, 50) { changed = true }
		case .Disc:
			if vec3_f64("Centre", &l.position) { changed = true }
			if vec3_f64("Normal", &l.direction) { changed = true }
			if slider_f64("Radius", &l.radius, 0.001, 50) { changed = true }
		case .Cylinder:
			if vec3_f64("Base", &l.position) { changed = true }
			if vec3_f64("Axis", &l.direction) { changed = true }
			if slider_f64("Radius", &l.radius, 0.001, 50) { changed = true }
			if slider_f64("Height", &l.height, 0.001, 100) { changed = true }
		case .Point:
			if vec3_f64("Position", &l.position) { changed = true }
		case .Spot:
			if vec3_f64("Position", &l.position) { changed = true }
			if vec3_f64("Direction", &l.direction) { changed = true }
			// Stored as cosines because that is what the kernel wants; shown as
			// degrees because that is what anyone lighting a shot thinks in.
			if cone_angle("Inner angle", &l.cos_inner) { changed = true }
			if cone_angle("Outer angle", &l.cos_outer) { changed = true }
		case .Distant:
			if vec3_f64("Direction", &l.direction) { changed = true }
			if slider_f64("Angular radius", &l.angular_radius, 0, 0.5) { changed = true }
		case .Mesh, .Dome:
			imgui.TextDisabled("driven by the scene, not editable here")
		}

		if needs_rebuild {
			ipr_scene_changed(app)
		} else if changed {
			ipr_lights_changed(&app.ipr)
		}
	}
}

@(private = "file")
light_kind_name :: proc(k: lc.Light_Kind) -> string {
	switch k {
	case .Quad:     return "Quad"
	case .Sphere:   return "Sphere"
	case .Mesh:     return "Mesh"
	case .Disc:     return "Disc"
	case .Cylinder: return "Cylinder"
	case .Point:    return "Point"
	case .Spot:     return "Spot"
	case .Distant:  return "Distant"
	case .Dome:     return "Dome"
	}
	return "?"
}

// ── f64 <-> ImGui f32 helpers ────────────────────────────────────────────────

@(private = "file")
slider_f64 :: proc(label: cstring, value: ^f64, lo, hi: f32) -> bool {
	v := f32(value^)
	if imgui.SliderFloat(label, &v, lo, hi) {
		value^ = f64(v)
		return true
	}
	return false
}

@(private = "file")
vec3_f64 :: proc(label: cstring, value: ^lc.Vec3) -> bool {
	v := [3]f32{f32(value.x), f32(value.y), f32(value.z)}
	imgui.SetNextItemWidth(-120)
	if imgui.DragFloat3(label, &v, 0.01) {
		value^ = lc.Vec3{f64(v[0]), f64(v[1]), f64(v[2])}
		return true
	}
	return false
}

@(private = "file")
colour3_hdr :: proc(label: cstring, value: ^lc.Color) -> bool {
	v := [3]f32{f32(value.x), f32(value.y), f32(value.z)}
	if imgui.ColorEdit3(label, &v, {.Float, .HDR}) {
		value^ = lc.Color{f64(v[0]), f64(v[1]), f64(v[2])}
		return true
	}
	return false
}

@(private = "file")
cone_angle :: proc(label: cstring, cosine: ^f64) -> bool {
	deg := f32(math.to_degrees(math.acos(clamp(cosine^, -1, 1))))
	imgui.SetNextItemWidth(-120)
	if imgui.SliderFloat(label, &deg, 0, 90) {
		cosine^ = math.cos(math.to_radians(f64(deg)))
		return true
	}
	return false
}
