package lumbre_importers

import "core:strings"
import m "core:math/linalg/glsl"

// The UsdLux → Lumbre light math lives in core/usd_light.odin so the CLI and the
// Houdini Hydra delegate produce identical lights. This file only adapts the
// `usd_shim`-facing `Usd_Light_Info` onto the shared `core.Usd_Light_Params`.

// Builds the shared, renderer-core light parameters from a collected shim light.
usd_light_info_to_params :: proc(info: Usd_Light_Info) -> Usd_Light_Params {
	return Usd_Light_Params{
		world                 = info.world,
		kind                  = Usd_Light_Kind(info.kind),
		intensity             = info.intensity,
		exposure              = info.exposure,
		color                 = info.color,
		normalize             = info.normalize,
		width                 = info.width,
		height                = info.height,
		radius                = info.radius,
		length                = info.length,
		angle                 = info.angle,
		treat_as_point        = info.treat_as_point,
		has_shaping           = info.has_shaping,
		shaping_cone_angle    = info.shaping_cone_angle,
		shaping_cone_softness = info.shaping_cone_softness,
		texture_file          = info.texture_file,
	}
}

// Converts a collected UsdLux light into Lumbre's Light. Returns false for Dome
// (and an unrecognized kind); a dome becomes Scene.environment via
// usd_dome_to_environment instead.
usd_make_light_from_info :: proc(info: Usd_Light_Info) -> (Light, bool) {
	return usd_make_light_from_params(usd_light_info_to_params(info))
}

// DomeLight -> Environment via the shared HDRI loader.
usd_dome_to_environment :: proc(info: Usd_Light_Info) -> (Environment, bool) {
	return usd_core_dome_to_environment(usd_light_info_to_params(info))
}

// Reads a UsdLux light's parameters (already confirmed live by
// usd_shim_get_light_data) and appends an entry to `lights`.
usd_emit_light :: proc(prim: Usd_Shim_Prim, data: Usd_Shim_Light_Data, world: m.mat4, lights: ^[dynamic]Usd_Light_Info) {
	data := data
	texture_file := ""
	if data.texture_file[0] != 0 {
		texture_file = strings.clone(string(cstring(&data.texture_file[0])), context.allocator)
	}
	append(lights, Usd_Light_Info {
		world                 = world,
		kind                  = data.kind,
		intensity             = f64(data.intensity),
		exposure              = f64(data.exposure),
		color                 = Color{f64(data.color[0]), f64(data.color[1]), f64(data.color[2])},
		normalize             = data.normalize != 0,
		width                 = f64(data.width),
		height                = f64(data.height),
		radius                = f64(data.radius),
		length                = f64(data.length),
		angle                 = f64(data.angle),
		treat_as_point        = data.treat_as_point != 0,
		has_shaping           = data.has_shaping != 0,
		shaping_cone_angle    = f64(data.shaping_cone_angle),
		shaping_cone_softness = f64(data.shaping_cone_softness),
		texture_file          = texture_file,
	})
}
