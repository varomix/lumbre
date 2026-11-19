package lumbre_importers

// The shim's time API, against assets/usd_scene_tests/animated_transforms.usda.
// Run from the repo root (`odin test importers`): the shim resolves its
// vendored USD relative to the test binary.

import "core:c"
import "core:math"
import "core:testing"

ANIMATED_SCENE :: "assets/usd_scene_tests/animated_transforms.usda"
STATIC_SCENE :: "assets/usd_scene_tests/distant_light.usda"

@(private = "file")
open_stage :: proc(t: ^testing.T, path: cstring) -> Usd_Shim_Stage {
	err: [512]u8
	stage := usd_shim_open_flattened(path, &err[0], c.int(len(err)))
	testing.expectf(t, stage != nil, "open %s: %s", path, cstring(&err[0]))
	return stage
}

// Finds a prim by its path below the pseudo root, one name at a time.
@(private = "file")
find_prim :: proc(stage: Usd_Shim_Stage, names: ..string) -> Usd_Shim_Prim {
	prim := usd_shim_get_pseudo_root(stage)
	for name in names {
		children: [32]Usd_Shim_Prim
		n := int(usd_shim_get_children(prim, &children[0], c.int(len(children))))
		next: Usd_Shim_Prim
		for child in children[:min(n, len(children))] {
			if string(usd_shim_prim_name(child)) == name {
				next = child
				break
			}
		}
		if next == nil do return nil
		prim = next
	}
	return prim
}

@(private = "file")
local_at :: proc(prim: Usd_Shim_Prim, time: f64) -> (mat: [16]f64, ok: bool) {
	ok = usd_shim_get_local_transform_at(prim, time, &mat[0]) != 0
	return
}

@(private = "file")
close_to :: proc(a, b: f64) -> bool {
	return abs(a - b) < 1e-9
}

@(test)
test_usd_time_info :: proc(t: ^testing.T) {
	stage := open_stage(t, ANIMATED_SCENE)
	if stage == nil do return
	defer usd_shim_close(stage)

	info: Usd_Shim_Time_Info
	testing.expect(t, usd_shim_get_time_info(stage, &info) != 0)
	testing.expect_value(t, info.has_authored_range, 1)
	testing.expect_value(t, info.start_time_code, 1.0)
	testing.expect_value(t, info.end_time_code, 24.0)
	testing.expect_value(t, info.time_codes_per_second, 24.0)
	testing.expect_value(t, info.frames_per_second, 24.0)

	still := open_stage(t, STATIC_SCENE)
	if still == nil do return
	defer usd_shim_close(still)
	still_info: Usd_Shim_Time_Info
	testing.expect(t, usd_shim_get_time_info(still, &still_info) != 0)
	testing.expect_value(t, still_info.has_authored_range, 0)
	testing.expect_value(t, still_info.time_codes_per_second, 24.0)
}

@(test)
test_usd_local_transform_at_time :: proc(t: ^testing.T) {
	stage := open_stage(t, ANIMATED_SCENE)
	if stage == nil do return
	defer usd_shim_close(stage)

	mover := find_prim(stage, "World", "mover")
	testing.expect(t, mover != nil)
	if mover == nil do return

	// Row-major with USD's row vectors: translation is elements 12..14.
	f1, ok1 := local_at(mover, 1)
	f2, ok2 := local_at(mover, 2)
	f24, ok24 := local_at(mover, 24)
	testing.expect(t, ok1 && ok2 && ok24)
	testing.expectf(t, close_to(f1[12], -3), "frame 1 x = %v", f1[12])
	testing.expectf(t, close_to(f2[12] - f1[12], 0.25), "frame step = %v", f2[12] - f1[12])
	testing.expectf(t, close_to(f24[12], 2.75), "frame 24 x = %v", f24[12])
	testing.expect(t, close_to(f2[13], 0.5) && close_to(f2[14], 1.5))
	// Between samples USD interpolates linearly.
	half, _ := local_at(mover, 1.5)
	testing.expectf(t, close_to(half[12], -2.875), "frame 1.5 x = %v", half[12])

	// At default time a translate authored only as samples has no value, so
	// the op contributes nothing: the old importer's collapse to the origin.
	def, _ := local_at(mover, USD_SHIM_TIME_DEFAULT)
	testing.expectf(t, close_to(def[12], 0) && close_to(def[13], 0), "default = %v", def[12:15])
	// The time-less entry point is exactly default time.
	old: [16]f64
	usd_shim_get_local_transform(mover, &old[0])
	testing.expect_value(t, old, def)

	// 15 degrees about Y per frame; frame 3 is 30 degrees. With row vectors
	// the first row is the image of +X: (cos, 0, -sin).
	spinner := find_prim(stage, "World", "spinner")
	s3, _ := local_at(spinner, 3)
	rad := math.to_radians(f64(30))
	testing.expectf(t, abs(s3[0] - math.cos(rad)) < 1e-6 && abs(s3[2] + math.sin(rad)) < 1e-6,
		"frame 3 row 0 = %v", s3[0:3])

	// A static prim reads the same at every time and at default time.
	box := find_prim(stage, "World", "static_box")
	b1, _ := local_at(box, 1)
	b17, _ := local_at(box, 17)
	bd, _ := local_at(box, USD_SHIM_TIME_DEFAULT)
	testing.expect_value(t, b1, b17)
	testing.expect_value(t, b1, bd)
}

@(test)
test_usd_transform_might_be_time_varying :: proc(t: ^testing.T) {
	stage := open_stage(t, ANIMATED_SCENE)
	if stage == nil do return
	defer usd_shim_close(stage)

	cases := [?]struct {
		path:    []string,
		varying: c.int,
	} {
		{{"World", "mover"}, 1},
		{{"World", "spinner"}, 1},
		{{"World", "parent"}, 1},
		// Its own ops are static; it moves only through its parent.
		{{"World", "parent", "child"}, 0},
		{{"World", "static_box"}, 0},
		{{"World", "floor"}, 0},
		{{"World", "main_cam"}, 0},
	}
	for tc in cases {
		prim := find_prim(stage, ..tc.path)
		testing.expectf(t, prim != nil, "missing %v", tc.path)
		if prim == nil do continue
		testing.expectf(t, usd_shim_transform_might_be_time_varying(prim) == tc.varying,
			"%v: expected %v", tc.path, tc.varying)
	}
}

@(test)
test_usd_point_instancer_at_time :: proc(t: ^testing.T) {
	stage := open_stage(t, ANIMATED_SCENE)
	if stage == nil do return
	defer usd_shim_close(stage)

	scatter := find_prim(stage, "World", "scatter")
	testing.expect(t, scatter != nil)
	if scatter == nil do return

	read :: proc(prim: Usd_Shim_Prim, time: f64) -> (ys: [2]f64, count: int) {
		data: Usd_Shim_Point_Instancer_Data
		if usd_shim_get_point_instancer_at(prim, time, &data) == 0 do return
		defer usd_shim_free_point_instancer(&data)
		count = int(data.instance_count)
		for i in 0 ..< min(count, 2) do ys[i] = data.transforms[i * 16 + 13]
		return
	}

	y1, n1 := read(scatter, 1)
	y2, n2 := read(scatter, 2)
	y24, _ := read(scatter, 24)
	testing.expect_value(t, n1, 2)
	testing.expect_value(t, n2, 2)
	for i in 0 ..< 2 {
		testing.expectf(t, abs(y1[i] - 0.2) < 1e-6, "frame 1 y[%d] = %v", i, y1[i])
		testing.expectf(t, abs(y2[i] - y1[i] - 0.05) < 1e-6, "frame step y[%d] = %v", i, y2[i] - y1[i])
		testing.expectf(t, abs(y24[i] - 1.35) < 1e-6, "frame 24 y[%d] = %v", i, y24[i])
	}

	// Default time keeps the old fallback to the earliest sample, through
	// both entry points.
	yd, nd := read(scatter, USD_SHIM_TIME_DEFAULT)
	testing.expect_value(t, nd, 2)
	testing.expect_value(t, yd, y1)
	old: Usd_Shim_Point_Instancer_Data
	testing.expect(t, usd_shim_get_point_instancer(scatter, &old) != 0)
	defer usd_shim_free_point_instancer(&old)
	testing.expect_value(t, int(old.instance_count), 2)
	if old.instance_count == 2 do testing.expect_value(t, old.transforms[13], y1[0])
}
