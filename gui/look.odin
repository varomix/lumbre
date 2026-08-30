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
