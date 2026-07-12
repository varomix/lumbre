package main

import "core:math"
import "core:strings"
import m "core:math/linalg/glsl"

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

// UsdLux radiance = intensity * 2^exposure * color, before any shape-area
// normalization. Every conversion below folds exposure in via this helper
// so it's applied exactly once.
//
// NOTE: unlike geometry, there is no independently verified reference for
// absolute light brightness here -- UsdLux leaves the physical unit of
// "intensity" implementation-defined, and DCCs/renderers (Houdini/Karma,
// RenderMan, ...) do not agree on a shared calibration. This applies the
// UsdLux formula literally; matching a specific renderer's on-screen
// brightness is unverified (plans/USD_SCENE_FORMAT.md verification step 4).
usd_light_radiance :: proc(info: Usd_Light_Info) -> Color {
	scale := info.intensity * math.pow(2.0, info.exposure)
	return info.color * scale
}

// Converts a collected UsdLux light into Lumbre's Light. Returns false for
// Dome (and an unrecognized kind), since a dome has no Light representation
// -- it becomes Scene.environment instead, via usd_dome_to_environment.
usd_make_light_from_info :: proc(info: Usd_Light_Info) -> (Light, bool) {
	radiance := usd_light_radiance(info)
	position := transform_point(Vec3{0, 0, 0}, info.world)

	switch info.kind {
	case .Sphere:
		// UsdLux emits from the sphere's *surface*; its world-space radius
		// follows the transform's scale, approximated here by the length a
		// unit +X vector picks up (uniform-scale assumption -- a
		// non-uniformly-scaled SphereLight is a documented v1 gap).
		radius := info.radius * m.length(transform_dir(Vec3{1, 0, 0}, info.world))
		if info.treat_as_point || radius <= 0.0 {
			return make_point_light(position, radiance), true
		}
		if info.has_shaping {
			// ShapingAPI's cone:softness is a *fraction* (0-1) of the cone
			// angle over which intensity falls off, not a second angle --
			// smoothStart = coneAngle * (1 - softness), matching the
			// smoothstep falloff sample_spot_light already implements.
			softness := clamp(info.shaping_cone_softness, 0.0, 1.0)
			outer := math.to_radians(info.shaping_cone_angle)
			inner := math.to_radians(info.shaping_cone_angle * (1.0 - softness))
			// A spot's aim has no dedicated USD attribute; a SphereLight
			// with ShapingAPI aims down local -Z, the same convention a
			// USD camera and RectLight/DiskLight use.
			direction := transform_dir(Vec3{0, 0, -1}, info.world)
			return make_spot_light(position, direction, inner, outer, radiance), true
		}
		if info.normalize {
			area := 4.0 * math.PI * radius * radius
			if area > 0.0 { radiance /= area }
		}
		return make_sphere_light(position, radius, radiance), true

	case .Rect:
		// RectLight is a 1x1 quad in local XY (normal -Z), scaled by
		// width/height; u/v are its world-space edge vectors.
		u := transform_dir(Vec3{info.width, 0, 0}, info.world)
		v := transform_dir(Vec3{0, info.height, 0}, info.world)
		if info.normalize {
			area := m.length(m.cross(u, v))
			if area > 0.0 { radiance /= area }
		}
		// make_area_light's `position` is a corner, but RectLight's
		// transform places its *center* at the origin -- offset by
		// -u/2 - v/2 to match.
		corner := position - u * 0.5 - v * 0.5
		return make_area_light(corner, u, v, radiance), true

	case .Disk:
		radius := info.radius * m.length(transform_dir(Vec3{1, 0, 0}, info.world))
		normal := transform_dir(Vec3{0, 0, -1}, info.world) // DiskLight emits along local -Z
		if info.normalize {
			area := math.PI * radius * radius
			if area > 0.0 { radiance /= area }
		}
		return make_disc_light(position, normal, radius, radiance), true

	case .Cylinder:
		// CylinderLight's major axis is local X (unlike most "spine is Z"
		// USD shapes); its radius is the cross-section in YZ.
		axis_vec := transform_dir(Vec3{1, 0, 0}, info.world)
		length_scale := m.length(axis_vec)
		radius_scale := m.length(transform_dir(Vec3{0, 1, 0}, info.world))
		radius := info.radius * radius_scale
		length := info.length * length_scale
		if info.normalize {
			area := 2.0 * math.PI * radius * length
			if area > 0.0 { radiance /= area }
		}
		return make_cylinder_light(position, axis_vec, radius, length, radiance), true

	case .Distant:
		direction := transform_dir(Vec3{0, 0, -1}, info.world)
		half_angle := math.to_radians(info.angle * 0.5)
		return make_distant_light(direction, half_angle, radiance), true

	case .Dome, .None:
		return Light{}, false
	}
	return Light{}, false
}

// DomeLight -> Environment via the existing HDRI loader. Rotation is read
// off the light's world transform (its local +X axis projected onto the
// world XZ plane), since UsdLuxDomeLight has no dedicated rotation
// attribute of its own -- only a Y-axis rotation is representable this way;
// a tilted dome is a documented v1 gap. The sign convention against
// Lumbre's rotate_y is unverified -- check against a reference render
// (plans/USD_SCENE_FORMAT.md verification step 4) before trusting an
// off-axis dome.
//
// Always uses the stage's own intensity/exposure -- the --hdri-rotation/
// --hdri-intensity CLI flags exist specifically to tune a --hdri-supplied
// file (see main.odin's help text) and only apply there; a USD dome is
// only reached when --hdri was not given (see scene.odin).
usd_dome_to_environment :: proc(info: Usd_Light_Info) -> (Environment, bool) {
	if info.texture_file == "" {
		return Environment{}, false
	}
	world_x := transform_dir(Vec3{1, 0, 0}, info.world)
	yaw_deg := math.to_degrees(math.atan2(world_x.z, world_x.x))
	intensity := info.intensity * math.pow(2.0, info.exposure)

	return load_environment(info.texture_file, yaw_deg, intensity)
}
