package lumbre_importers

// The shim's time API, against assets/usd_scene_tests/animated_transforms.usda.
// Run from the repo root (`odin test importers`): the shim resolves its
// vendored USD relative to the test binary.

import "core:c"
import "core:math"
import m "core:math/linalg/glsl"
import "core:mem/virtual"
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

// ── Import at a time: step 2 ────────────────────────────────────────────────

@(private = "file")
find_mesh :: proc(data: ObjData, path: string) -> ^Mesh {
	for &msh in data.meshes {
		if msh.path == path do return &msh
	}
	return nil
}

@(private = "file")
translation :: proc(mat: m.mat4) -> [3]f32 {
	return {mat[3][0], mat[3][1], mat[3][2]}
}

@(private = "file")
near3 :: proc(a, b: [3]f32) -> bool {
	return abs(a.x - b.x) < 1e-5 && abs(a.y - b.y) < 1e-5 && abs(a.z - b.z) < 1e-5
}

@(test)
test_usd_import_motion_at_frame :: proc(t: ^testing.T) {
	arena: virtual.Arena
	testing.expect(t, virtual.arena_init_growing(&arena) == nil)
	defer virtual.arena_destroy(&arena)
	context.allocator = virtual.arena_allocator(&arena)

	data, cameras, _, ok := load_usd(ANIMATED_SCENE, 0, 12.0)
	testing.expect(t, ok)
	if !ok do return

	// mover: x = -3 + 0.25 * (frame - 1), so -0.25 at 12 and -0.5 at 11.
	mover := find_mesh(data, "/World/mover")
	testing.expect(t, mover != nil)
	if mover != nil {
		testing.expect(t, mover.has_prev_transform)
		testing.expectf(t, near3(translation(mover.transform), {-0.25, 0.5, 1.5}), "now %v", translation(mover.transform))
		testing.expectf(t, near3(translation(mover.prev_transform), {-0.5, 0.5, 1.5}), "prev %v", translation(mover.prev_transform))
	}

	// child moves only through its animated parent: y = 0.1 * (frame - 1) + 0.5.
	child := find_mesh(data, "/World/parent/child")
	testing.expect(t, child != nil)
	if child != nil {
		testing.expect(t, child.has_prev_transform)
		testing.expectf(t, near3(translation(child.transform), {-2, 1.6, -1.5}), "now %v", translation(child.transform))
		testing.expectf(t, near3(translation(child.prev_transform), {-2, 1.5, -1.5}), "prev %v", translation(child.prev_transform))
	}

	// spinner turns 15 degrees a frame in place: rotation moves, translation not.
	spinner := find_mesh(data, "/World/spinner")
	testing.expect(t, spinner != nil && spinner.has_prev_transform)
	if spinner != nil {
		testing.expect(t, near3(translation(spinner.transform), translation(spinner.prev_transform)))
		testing.expect(t, spinner.transform != spinner.prev_transform)
	}

	// Each PointInstancer copy carries its own point's motion: y rises 0.05.
	for path in ([]string{"/World/scatter[0]/pebble", "/World/scatter[1]/pebble"}) {
		pebble := find_mesh(data, path)
		testing.expectf(t, pebble != nil, "missing %s", path)
		if pebble == nil do continue
		testing.expect(t, pebble.has_prev_transform)
		dy := translation(pebble.transform).y - translation(pebble.prev_transform).y
		testing.expectf(t, abs(dy - 0.05) < 1e-5, "%s dy = %v", path, dy)
	}

	// Static geometry is untouched.
	for path in ([]string{"/World/static_box", "/World/floor"}) {
		msh := find_mesh(data, path)
		testing.expectf(t, msh != nil && !msh.has_prev_transform, "%s should be static", path)
	}

	// The camera is still: its previous world is its world.
	testing.expect_value(t, len(cameras), 1)
	if len(cameras) == 1 {
		testing.expect(t, cameras[0].prev_world == cameras[0].world)
	}
}

@(test)
test_usd_import_time_defaults :: proc(t: ^testing.T) {
	arena: virtual.Arena
	testing.expect(t, virtual.arena_init_growing(&arena) == nil)
	defer virtual.arena_destroy(&arena)
	context.allocator = virtual.arena_allocator(&arena)

	// No --frame: a stage with a range is read at its start time, so
	// sample-only ops resolve instead of collapsing to the origin. Frame 1's
	// previous frame is before the first sample, which USD holds: no motion.
	data, _, _, ok := load_usd(ANIMATED_SCENE, 0)
	testing.expect(t, ok)
	if ok {
		mover := find_mesh(data, "/World/mover")
		testing.expect(t, mover != nil)
		if mover != nil {
			testing.expectf(t, near3(translation(mover.transform), {-3, 0.5, 1.5}), "start %v", translation(mover.transform))
		}
		for msh in data.meshes {
			testing.expectf(t, !msh.has_prev_transform, "%s moves at the first frame", msh.path)
		}
	}

	// A stage without a range is read at default time, as before: no motion.
	still, cams, _, still_ok := load_usd(STATIC_SCENE, 0)
	testing.expect(t, still_ok)
	for msh in still.meshes {
		testing.expectf(t, !msh.has_prev_transform, "%s moves in a static stage", msh.path)
	}
	for cam in cams {
		testing.expect(t, cam.prev_world == cam.world)
	}
}

// Playback: moving an imported scene to a frame gives exactly what importing
// at that frame gives, mesh by mesh, previous transforms included.
@(test)
test_usd_playback_matches_import :: proc(t: ^testing.T) {
	arena: virtual.Arena
	testing.expect(t, virtual.arena_init_growing(&arena) == nil)
	defer virtual.arena_destroy(&arena)
	context.allocator = virtual.arena_allocator(&arena)

	start, _, _, ok := load_usd(ANIMATED_SCENE, 0)
	testing.expect(t, ok)
	if !ok do return
	scene := Scene{meshes = start.meshes}
	build_default_scene_graph(&scene)

	stage := open_stage(t, ANIMATED_SCENE)
	if stage == nil do return
	defer usd_shim_close(stage)

	for frame in ([]f64{12, 24, 1}) {
		motion := usd_eval_motion(stage, frame)
		changed := scene_apply_usd_motion(&scene, motion)
		usd_motion_destroy(&motion)
		testing.expectf(t, changed > 0, "frame %v moved nothing", frame)

		want, _, _, want_ok := load_usd(ANIMATED_SCENE, 0, frame)
		testing.expect(t, want_ok)
		testing.expect_value(t, len(scene.meshes), len(want.meshes))
		for msh, i in scene.meshes {
			w := want.meshes[i]
			testing.expectf(t, msh.path == w.path, "order differs at %d", i)
			testing.expectf(t, msh.transform == w.transform, "frame %v %s: transform", frame, msh.path)
			testing.expectf(t, msh.has_prev_transform == w.has_prev_transform, "frame %v %s: moving", frame, msh.path)
			if w.has_prev_transform {
				testing.expectf(t, msh.prev_transform == w.prev_transform, "frame %v %s: prev", frame, msh.path)
			}
			node := scene.nodes[1 + i]
			testing.expect(t, node.world_transform == w.transform)
			want_prev := w.prev_transform if w.has_prev_transform else w.transform
			testing.expectf(t, node.prev_world_transform == want_prev, "frame %v %s: node prev", frame, msh.path)
		}
	}

	// Applying the frame the scene is already at changes nothing.
	motion := usd_eval_motion(stage, 1.0)
	defer usd_motion_destroy(&motion)
	testing.expect_value(t, scene_apply_usd_motion(&scene, motion), 0)
}
