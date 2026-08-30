package main

// Look files: persisting material edits.
//
// Lumbre does not author back to the USD stage yet, so without this every
// material change made in the GUI is lost when the scene is reloaded — which
// makes the material panel a toy rather than a tool. A look file is a small
// JSON sidecar of material overrides, saved beside the scene and applied after
// import.
//
// It stores only the shading parameters the panel can edit. Textures, geometry
// and lights come from the scene file and are not duplicated here.

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:strings"

import lc "../core"
import imp "../importers"

LOOK_EXT :: ".lumbrelook"
LOOK_VERSION :: 1

Look_Material :: struct {
	index:               int     `json:"index"`,
	kind:                int     `json:"kind"`,
	base_color:          [3]f64  `json:"base_color"`,
	roughness:           f64     `json:"roughness"`,
	metallic:            f64     `json:"metallic"`,
	specular:            f64     `json:"specular"`,
	specular_tint:       [3]f64  `json:"specular_tint"`,
	ior:                 f64     `json:"ior"`,
	clearcoat:           f64     `json:"clearcoat"`,
	clearcoat_roughness: f64     `json:"clearcoat_roughness"`,
	sheen:               f64     `json:"sheen"`,
	sheen_tint:          [3]f64  `json:"sheen_tint"`,
	anisotropic:         f64     `json:"anisotropic"`,
	transmission:        f64     `json:"transmission"`,
	subsurface:          f64     `json:"subsurface"`,
	subsurface_color:    [3]f64  `json:"subsurface_color"`,
	subsurface_radius:   [3]f64  `json:"subsurface_radius"`,
	subsurface_scale:    f64     `json:"subsurface_scale"`,
	emission:            [3]f64  `json:"emission"`,
	emission_strength:   f64     `json:"emission_strength"`,
}

Look_File :: struct {
	version:   int             `json:"version"`,
	scene:     string          `json:"scene"`,
	materials: []Look_Material `json:"materials"`,
}

// Sidecar path for a scene: "kitchen.usd" -> "kitchen.usd.lumbrelook".
look_path_for :: proc(scene_path: string, allocator := context.allocator) -> string {
	return strings.concatenate({scene_path, LOOK_EXT}, allocator)
}

look_save :: proc(app: ^App) -> bool {
	if !app.scene_loaded || len(app.core.scene.materials) == 0 {
		log_line(&app.log, "[look] nothing to save")
		return false
	}

	mats := make([]Look_Material, len(app.core.scene.materials), context.temp_allocator)
	for m, i in app.core.scene.materials {
		mats[i] = Look_Material {
			index               = i,
			kind                = int(m.kind),
			base_color          = {m.albedo.x, m.albedo.y, m.albedo.z},
			roughness           = m.roughness,
			metallic            = m.metallic,
			specular            = m.specular,
			specular_tint       = {m.specular_tint.x, m.specular_tint.y, m.specular_tint.z},
			ior                 = m.ir,
			clearcoat           = m.clearcoat,
			clearcoat_roughness = m.clearcoat_roughness,
			sheen               = m.sheen,
			sheen_tint          = {m.sheen_tint.x, m.sheen_tint.y, m.sheen_tint.z},
			anisotropic         = m.anisotropic,
			transmission        = m.spec_trans,
			subsurface          = m.subsurface,
			subsurface_color    = {m.subsurface_color.x, m.subsurface_color.y, m.subsurface_color.z},
			subsurface_radius   = {m.subsurface_radius.x, m.subsurface_radius.y, m.subsurface_radius.z},
			subsurface_scale    = m.subsurface_scale,
			emission            = {m.emission.x, m.emission.y, m.emission.z},
			emission_strength   = m.emission_strength,
		}
	}

	file := Look_File {
		version   = LOOK_VERSION,
		scene     = app.scene_path,
		materials = mats,
	}

	data, err := json.marshal(file, {pretty = true}, context.temp_allocator)
	if err != nil {
		log_printf(&app.log, "[look] could not encode: %v", err)
		return false
	}

	path := look_path_for(app.scene_path, context.temp_allocator)
	if write_err := os.write_entire_file(path, data); write_err != nil {
		log_printf(&app.log, "[look] could not write %s: %v", path, write_err)
		return false
	}
	log_printf(&app.log, "[look] saved %d materials to %s", len(mats), path)
	return true
}

// Applies a sidecar if one exists. Called after import, so it overrides what
// the scene file supplied. A missing file is not an error — most scenes have
// no look saved.
look_load :: proc(app: ^App) -> bool {
	if !app.scene_loaded {
		return false
	}
	path := look_path_for(app.scene_path, context.temp_allocator)
	if !os.exists(path) {
		return false
	}

	data, read_err := os.read_entire_file_from_path(path, context.temp_allocator)
	if read_err != nil {
		log_printf(&app.log, "[look] could not read %s", path)
		return false
	}

	file: Look_File
	if err := json.unmarshal(data, &file, allocator = context.temp_allocator); err != nil {
		log_printf(&app.log, "[look] %s is not valid: %v", path, err)
		return false
	}
	if file.version > LOOK_VERSION {
		log_printf(&app.log, "[look] %s was written by a newer Lumbre (v%d)", path, file.version)
		return false
	}

	applied := 0
	for lm in file.materials {
		// Material indices come from import order. A look saved against a
		// different version of the scene can legitimately be out of range, so
		// skip rather than fail the whole file.
		if lm.index < 0 || lm.index >= len(app.core.scene.materials) {
			continue
		}
		m := &app.core.scene.materials[lm.index]
		m.kind = lc.Material_Kind(clamp(lm.kind, 0, int(lc.Material_Kind.Emissive)))
		m.albedo = {lm.base_color[0], lm.base_color[1], lm.base_color[2]}
		m.roughness = lm.roughness
		m.metallic = lm.metallic
		m.specular = lm.specular
		m.specular_tint = {lm.specular_tint[0], lm.specular_tint[1], lm.specular_tint[2]}
		m.ir = lm.ior
		m.clearcoat = lm.clearcoat
		m.clearcoat_roughness = lm.clearcoat_roughness
		m.sheen = lm.sheen
		m.sheen_tint = {lm.sheen_tint[0], lm.sheen_tint[1], lm.sheen_tint[2]}
		m.anisotropic = lm.anisotropic
		m.spec_trans = lm.transmission
		m.subsurface = lm.subsurface
		m.subsurface_color = {lm.subsurface_color[0], lm.subsurface_color[1], lm.subsurface_color[2]}
		m.subsurface_radius = {lm.subsurface_radius[0], lm.subsurface_radius[1], lm.subsurface_radius[2]}
		m.subsurface_scale = lm.subsurface_scale
		m.emission = {lm.emission[0], lm.emission[1], lm.emission[2]}
		m.emission_strength = lm.emission_strength
		applied += 1
	}

	skipped := len(file.materials) - applied
	if skipped > 0 {
		log_printf(
			&app.log,
			"[look] applied %d materials from %s (%d skipped: scene has %d)",
			applied, path, skipped, len(app.core.scene.materials),
		)
	} else {
		log_printf(&app.log, "[look] applied %d materials from %s", applied, path)
	}
	return applied > 0
}

look_status :: proc(app: ^App) -> string {
	if !app.scene_loaded {
		return "no scene"
	}
	path := look_path_for(app.scene_path, context.temp_allocator)
	if os.exists(path) {
		return fmt.tprintf("look file: %s", path)
	}
	return "no look file saved"
}

// ── USD export ───────────────────────────────────────────────────────────────

// Writes the current materials as a USD overlay layer beside the scene:
// `kitchen.usd` -> `kitchen_look.usda`. The overlay sublayers the original, so
// opening it gives the scene with the look applied and the asset on disk is
// never modified.
//
// Only materials that were imported from a USD material prim can be authored —
// an OBJ has no prim to override, and a display-colour fallback is synthesised
// from the mesh rather than read from a material.
look_export_usd :: proc(app: ^App) -> bool {
	if !app.scene_loaded || !app.usd.open {
		log_line(&app.log, "[look] USD export needs a USD scene")
		return false
	}

	paths := app.core.scene.material_paths
	if len(paths) == 0 {
		log_line(&app.log, "[look] this scene has no material provenance to author onto")
		return false
	}

	entries := make([dynamic]imp.Lumbre_Look_Material, context.temp_allocator)
	for m, i in app.core.scene.materials {
		if i >= len(paths) || paths[i] == "" {
			continue
		}
		strength := m.emission_strength if m.emission_strength > 0 else 1
		append(
			&entries,
			imp.Lumbre_Look_Material {
				material_path = strings.clone_to_cstring(paths[i], context.temp_allocator),
				base_color = {f32(m.albedo.x), f32(m.albedo.y), f32(m.albedo.z)},
				roughness = f32(m.roughness),
				metallic = f32(m.metallic),
				ior = f32(m.ir),
				emissive_color = {
					f32(m.emission.x * strength),
					f32(m.emission.y * strength),
					f32(m.emission.z * strength),
				},
				clearcoat = f32(m.clearcoat),
				clearcoat_roughness = f32(m.clearcoat_roughness),
				// UsdPreviewSurface has no transmission input; opacity is the
				// closest thing it does have.
				opacity = f32(1.0 - clamp(m.spec_trans, 0, 1)),
			},
		)
	}

	if len(entries) == 0 {
		log_line(&app.log, "[look] no materials in this scene came from a USD material prim")
		return false
	}

	// A shader input that is *connected* to a texture ignores an authored
	// value — the connection wins, and that is correct USD composition, not a
	// bug here. Say so rather than letting the override look like it failed.
	textured := 0
	for m, i in app.core.scene.materials {
		if i >= len(paths) || paths[i] == "" {
			continue
		}
		if m.albedo_tex.has_data || m.metallic_roughness_tex.has_data || m.emissive_tex.has_data {
			textured += 1
		}
	}

	out_path := look_usd_path_for(app.scene_path, context.temp_allocator)
	// USD refuses to create a layer that already exists.
	if os.exists(out_path) {
		os.remove(out_path)
	}

	// The overlay sits beside the scene, so reference it by file name. Using
	// the path as given would bake in whatever it was relative to when the
	// scene was opened, and break as soon as the pair is moved.
	scene_name := app.scene_path
	if idx := strings.last_index_byte(scene_name, '/'); idx >= 0 {
		scene_name = scene_name[idx + 1:]
	}

	err_buf: [512]u8
	written := imp.lumbre_usd_export_look(
		app.usd.stage,
		strings.clone_to_cstring(out_path, context.temp_allocator),
		strings.clone_to_cstring(scene_name, context.temp_allocator),
		raw_data(entries[:]),
		i32(len(entries)),
		raw_data(err_buf[:]),
		len(err_buf),
	)
	if written == 0 {
		log_printf(&app.log, "[look] USD export failed: %s", string(cstring(raw_data(err_buf[:]))))
		return false
	}

	log_printf(&app.log, "[look] wrote %d material overrides to %s", written, out_path)
	if textured > 0 {
		log_printf(
			&app.log,
			"[look] note: %d of them drive inputs from textures, where a connection overrides an authored value",
			textured,
		)
	}
	return true
}

// "kitchen.usd" -> "kitchen_look.usda". Always .usda so the result is readable.
look_usd_path_for :: proc(scene_path: string, allocator := context.allocator) -> string {
	stem := scene_path
	if idx := strings.last_index_byte(scene_path, '.'); idx > 0 {
		stem = scene_path[:idx]
	}
	return strings.concatenate({stem, "_look.usda"}, allocator)
}
