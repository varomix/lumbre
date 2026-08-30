package lumbre_importers

import "core:math"
import "core:strings"
import m "core:math/linalg/glsl"

// Reads a UsdGeomCamera's lens parameters and appends an entry (world
// transform + params) to `cameras`. A no-op if `prim` isn't a Camera or
// carries no readable camera data.
usd_emit_camera :: proc(prim: Usd_Shim_Prim, world: m.mat4, cameras: ^[dynamic]Usd_Camera_Info) {
	cam_data: Usd_Shim_Camera_Data
	if usd_shim_get_camera_data(prim, &cam_data) == 0 {
		return
	}
	name := string(usd_shim_prim_name(prim))
	append(cameras, Usd_Camera_Info {
		world                  = world,
		name                   = strings.clone(name, context.allocator),
		focal_length_mm        = f64(cam_data.focal_length_mm),
		horizontal_aperture_mm = f64(cam_data.horizontal_aperture_mm),
		vertical_aperture_mm   = f64(cam_data.vertical_aperture_mm),
		clip_near              = f64(cam_data.clipping_range[0]),
		clip_far               = f64(cam_data.clipping_range[1]),
		focus_distance         = f64(cam_data.focus_distance),
		f_stop                 = f64(cam_data.f_stop),
	})
}

// Converts a USD camera's world transform + lens parameters into Lumbre's
// Camera. USD cameras look down local -Z with +Y up. `aspect_ratio` comes
// from the render's own --width/--height, not the camera's aperture ratio,
// since output resolution is a Lumbre/CLI concern -- see
// plans/USD_SCENE_FORMAT.md Phase 2's "open question" (CLI wins; a
// mismatched aperture just changes framing, not image distortion).
usd_make_camera_from_info :: proc(info: Usd_Camera_Info, aspect_ratio: f64) -> Camera {
	lookfrom := transform_point(Vec3{0, 0, 0}, info.world)
	forward := transform_dir(Vec3{0, 0, -1}, info.world)
	vup := transform_dir(Vec3{0, 1, 0}, info.world)
	if m.length(forward) < 1.0e-8 {
		forward = Vec3{0, 0, -1}
	}
	forward = m.normalize(forward)
	if m.length(vup) < 1.0e-8 {
		vup = Vec3{0, 1, 0}
	}

	// focalLength/aperture are both authored in "tenths of a scene unit"
	// per UsdGeom_CameraUnits, so their ratio -- and therefore vfov -- is
	// unit-agnostic even though the individual numbers aren't literally
	// millimeters unless the stage's scene unit happens to be centimeters
	// (the common case; metersPerUnit is not yet read -- plan Phase 4).
	vfov := 40.0
	if info.focal_length_mm > 0.0 && info.vertical_aperture_mm > 0.0 {
		half_angle := math.atan(info.vertical_aperture_mm / (2.0 * info.focal_length_mm))
		vfov = math.to_degrees(2.0 * half_angle)
	}

	focus := info.focus_distance
	if focus <= 0.0 {
		focus = 10.0
	}
	lookat := lookfrom + forward * focus

	// Depth of field: aperture *diameter* in scene units from focalLength/
	// fStop (focalLength is in tenths of a scene unit, hence the /10). No
	// fStop authored means a pinhole camera, matching Lumbre's default.
	aperture := 0.0
	if info.f_stop > 0.0 {
		aperture = (info.focal_length_mm / 10.0) / info.f_stop
	}

	return make_camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus)
}
