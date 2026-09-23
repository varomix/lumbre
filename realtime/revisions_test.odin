package lumbre_realtime
import "core:math/rand"
import "core:testing"

import lc "../core"

@(test)
test_edits_split_between_geometry_and_instance_revisions :: proc(t: ^testing.T) {
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)
	geometry := geometry_revision(&scene)
	placement := instance_revision(&scene)
	scene.camera.origin.x = 37
	scene.materials[0].albedo = {0.3, 0.2, 0.1}
	scene.environment.rotation = 1.2
	testing.expect_value(t, geometry_revision(&scene), geometry)
	testing.expect_value(t, instance_revision(&scene), placement)
	scene.nodes[0].local_transform[0, 3] = 2
	testing.expect_value(t, geometry_revision(&scene), geometry)
	testing.expect(t, instance_revision(&scene) != placement, "transforms must invalidate instances")
	scene.nodes[0].local_transform[0, 3] = 0
	scene.meshes[0].triangles[0].mat_idx = 0
	testing.expect(t, geometry_revision(&scene) != geometry, "material assignment must invalidate batches")
}

@(test)
test_material_refresh_does_not_touch_geometry :: proc(t: ^testing.T) {
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)
	cpu, ok := scene_build_cpu(&scene)
	testing.expect(t, ok)
	defer scene_free_cpu(&cpu)
	gpu := Scene_GPU{batches = cpu.batches}
	// Poison the world transform: a material update must not recompute it.
	scene.nodes[0].world_transform[0, 3] = 123
	scene.materials[0].albedo = {0.2, 0.3, 0.4}
	scene_refresh_materials(&gpu, &scene)
	testing.expect_value(t, scene.nodes[0].world_transform[0, 3], 123)
	testing.expect_value(t, gpu.batches[0].material.base_color.x, f32(0.2))
}

@(test)
test_culling_keeps_crossing_boxes_and_rejects_each_clip_plane :: proc(t: ^testing.T) {
	identity := (matrix[4, 4]f32)(1)
	testing.expect(t, bounds_visible({-2, -2, -1}, {2, 2, 2}, identity), "box enclosing frustum")
	testing.expect(t, bounds_visible({-2, 0, 0.5}, {0, 1, 0.8}, identity), "partly visible")
	testing.expect(t, bounds_visible({1, 0, 0}, {1, 1, 1}, identity), "touching clip boundary")
	for axis in 0 ..< 3 {
		lo, hi := [3]f32{-0.5, -0.5, 0.1}, [3]f32{0.5, 0.5, 0.9}
		lo[axis], hi[axis] = 2, 3
		testing.expect(t, !bounds_visible(lo, hi, identity))
		lo[axis], hi[axis] = -3, -2
		testing.expect(t, !bounds_visible(lo, hi, identity))
	}
}

@(test)
test_instance_runs_split_at_culled_instances :: proc(t: ^testing.T) {
	visible := []bool{false, true, true, false, true}
	first, count := instance_run(visible, 0, 5)
	testing.expect_value(t, first, 1)
	testing.expect_value(t, count, 2)
	first, count = instance_run(visible, 3, 5)
	testing.expect_value(t, first, 4)
	testing.expect_value(t, count, 1)
	_, count = instance_run(visible, 5, 5)
	testing.expect_value(t, count, 0)
	// A batch's range ends a run even when the next instance is visible.
	first, count = instance_run(visible, 1, 2)
	testing.expect_value(t, first, 1)
	testing.expect_value(t, count, 1)
}

// The eight-corner test the plane test replaced, kept as the reference.
@(private = "file")
corners_visible :: proc(lo, hi: [3]f32, view_proj: matrix[4, 4]f32) -> bool {
	outside := [6]bool{true, true, true, true, true, true}
	for i in 0 ..< 8 {
		p := view_proj * [4]f32{i & 1 == 0 ? lo.x : hi.x, i & 2 == 0 ? lo.y : hi.y, i & 4 == 0 ? lo.z : hi.z, 1}
		outside[0] &&= p.x < -p.w
		outside[1] &&= p.x > p.w
		outside[2] &&= p.y < -p.w
		outside[3] &&= p.y > p.w
		outside[4] &&= p.z < 0
		outside[5] &&= p.z > p.w
	}
	for rejected in outside { if rejected { return false } }
	return true
}

@(test)
test_plane_culling_matches_corner_culling :: proc(t: ^testing.T) {
	cam := lc.make_camera(
		lookfrom = {3, 2, 6}, lookat = {0, 0.5, 0}, vup = {0, 1, 0},
		vfov = 50, aspect_ratio = 16.0 / 9.0, aperture = 0, focus_dist = 7,
	)
	f := camera_frame(cam)
	view_proj := camera_projection(f) * camera_view(f)

	rng := rand.create(7)
	context.random_generator = rand.default_random_generator(&rng)
	kept, mismatches := 0, 0
	for _ in 0 ..< 20000 {
		c := [3]f32{rand.float32_range(-30, 30), rand.float32_range(-30, 30), rand.float32_range(-30, 30)}
		e := [3]f32{rand.float32_range(0, 4), rand.float32_range(0, 4), rand.float32_range(0, 4)}
		lo, hi := c - e, c + e
		want := corners_visible(lo, hi, view_proj)
		if bounds_visible(lo, hi, view_proj) != want {
			mismatches += 1
		}
		if want { kept += 1 }
	}
	testing.expectf(t, mismatches == 0, "%d of 20000 boxes culled differently", mismatches)
	testing.expectf(t, kept > 100 && kept < 19900, "degenerate sample: %d kept", kept)
}
