package lumbre_core

import "core:math"
import m "core:math/linalg/glsl"

// Shared UsdLux → Lumbre light translation. This is the single source of truth
// for how a USD light's authored parameters (intensity, exposure, colour, shape
// size, area-normalization, shaping cone) become a `core.Light` or a dome
// `Environment`. Both frontends drive it from raw UsdLux parameters:
//
//   * the CLI importer (`usd_light_import.odin`) fills `Usd_Light_Params` from
//     `usd_shim`'s light data, and
//   * the Houdini Hydra delegate forwards the same authored parameters plus the
//     light's world transform through the bridge.
//
// Keeping the normalization and shape math here — rather than re-deriving it in
// the C++ delegate — is what makes the viewport and the CLI agree on light
// brightness and falloff. See plans/HOUDINI_21_HYDRA_PLUGIN.md.

// UsdLux light types, matching the `usd_shim` enum ordinals so the CLI can pass
// its shim kind straight through.
Usd_Light_Kind :: enum i32 {
	None     = 0,
	Sphere   = 1,
	Rect     = 2,
	Disk     = 3,
	Cylinder = 4,
	Distant  = 5,
	Dome     = 6,
}

// Authored UsdLux parameters plus the light's world transform. Shape sizes are
// the light's *local* authored values (width/height/radius/length); world-space
// scale is recovered from `world` inside the conversion so a scaled light prim
// behaves identically to the CLI.
Usd_Light_Params :: struct {
	world:                 m.mat4,
	kind:                  Usd_Light_Kind,
	intensity:             f64,
	exposure:              f64,
	color:                 Color,
	normalize:             bool,
	width, height:         f64,
	radius:                f64,
	length:                f64,
	angle:                 f64,
	treat_as_point:        bool,
	has_shaping:           bool,
	shaping_cone_angle:    f64,
	shaping_cone_softness: f64,
	texture_file:          string,
}

// UsdLux radiance = intensity * 2^exposure * color, before any shape-area
// normalization. Every conversion below folds exposure in via this helper so
// it's applied exactly once.
//
// NOTE: UsdLux leaves the physical unit of "intensity" implementation-defined,
// and DCCs/renderers (Houdini/Karma, RenderMan, ...) do not agree on a shared
// calibration. This applies the UsdLux formula literally.
usd_light_radiance :: proc(params: Usd_Light_Params) -> Color {
	scale := params.intensity * math.pow(2.0, params.exposure)
	return params.color * scale
}

// Converts authored UsdLux parameters into Lumbre's Light. Returns false for
// Dome (and an unrecognized kind), since a dome has no Light representation --
// it becomes Scene.environment instead, via usd_dome_to_environment.
usd_make_light_from_params :: proc(params: Usd_Light_Params) -> (Light, bool) {
	radiance := usd_light_radiance(params)
	position := transform_point(Vec3{0, 0, 0}, params.world)

	switch params.kind {
	case .Sphere:
		// UsdLux emits from the sphere's *surface*; its world-space radius
		// follows the transform's scale, approximated here by the length a unit
		// +X vector picks up (uniform-scale assumption -- a non-uniformly-scaled
		// SphereLight is a documented v1 gap).
		radius := params.radius * m.length(transform_dir(Vec3{1, 0, 0}, params.world))
		if params.treat_as_point || radius <= 0.0 {
			return make_point_light(position, radiance), true
		}
		if params.has_shaping {
			// ShapingAPI's cone:softness is a *fraction* (0-1) of the cone angle
			// over which intensity falls off, not a second angle --
			// smoothStart = coneAngle * (1 - softness), matching the smoothstep
			// falloff sample_spot_light already implements.
			softness := clamp(params.shaping_cone_softness, 0.0, 1.0)
			outer := math.to_radians(params.shaping_cone_angle)
			inner := math.to_radians(params.shaping_cone_angle * (1.0 - softness))
			// A spot's aim has no dedicated USD attribute; a SphereLight with
			// ShapingAPI aims down local -Z, the same convention a USD camera
			// and RectLight/DiskLight use.
			direction := transform_dir(Vec3{0, 0, -1}, params.world)
			return make_spot_light(position, direction, inner, outer, radiance), true
		}
		if params.normalize {
			area := 4.0 * math.PI * radius * radius
			if area > 0.0 { radiance /= area }
		}
		return make_sphere_light(position, radius, radiance), true

	case .Rect:
		// RectLight is a 1x1 quad in local XY (normal -Z), scaled by
		// width/height; u/v are its world-space edge vectors.
		u := transform_dir(Vec3{params.width, 0, 0}, params.world)
		v := transform_dir(Vec3{0, params.height, 0}, params.world)
		if params.normalize {
			area := m.length(m.cross(u, v))
			if area > 0.0 { radiance /= area }
		}
		// make_area_light's `position` is a corner, but RectLight's transform
		// places its *center* at the origin -- offset by -u/2 - v/2 to match.
		corner := position - u * 0.5 - v * 0.5
		return make_area_light(corner, u, v, radiance), true

	case .Disk:
		radius := params.radius * m.length(transform_dir(Vec3{1, 0, 0}, params.world))
		normal := transform_dir(Vec3{0, 0, -1}, params.world) // DiskLight emits along local -Z
		if params.normalize {
			area := math.PI * radius * radius
			if area > 0.0 { radiance /= area }
		}
		return make_disc_light(position, normal, radius, radiance), true

	case .Cylinder:
		// CylinderLight's major axis is local X (unlike most "spine is Z" USD
		// shapes); its radius is the cross-section in YZ.
		axis_vec := transform_dir(Vec3{1, 0, 0}, params.world)
		length_scale := m.length(axis_vec)
		radius_scale := m.length(transform_dir(Vec3{0, 1, 0}, params.world))
		radius := params.radius * radius_scale
		length := params.length * length_scale
		if params.normalize {
			area := 2.0 * math.PI * radius * length
			if area > 0.0 { radiance /= area }
		}
		return make_cylinder_light(position, axis_vec, radius, length, radiance), true

	case .Distant:
		direction := transform_dir(Vec3{0, 0, -1}, params.world)
		half_angle := math.to_radians(params.angle * 0.5)
		return make_distant_light(direction, half_angle, radiance), true

	case .Dome, .None:
		return Light{}, false
	}
	return Light{}, false
}

// DomeLight -> Environment via the existing HDRI loader. Rotation is read off
// the light's world transform (its local +X axis projected onto the world XZ
// plane), since UsdLuxDomeLight has no dedicated rotation attribute of its own
// -- only a Y-axis rotation is representable this way; a tilted dome is a
// documented v1 gap.
//
// Always uses the light's own intensity/exposure.
usd_dome_to_environment :: proc(params: Usd_Light_Params) -> (Environment, bool) {
	intensity := params.intensity * math.pow(2.0, params.exposure)
	// A textureless DomeLight is a uniform colour dome (the default LOP Dome
	// Light). Emit a constant environment tinted by its colour so it lights the
	// scene instead of doing nothing.
	if params.texture_file == "" {
		return make_constant_environment(params.color, intensity)
	}
	world_x := transform_dir(Vec3{1, 0, 0}, params.world)
	yaw_deg := math.to_degrees(math.atan2(world_x.z, world_x.x))
	return load_environment(params.texture_file, yaw_deg, intensity)
}
