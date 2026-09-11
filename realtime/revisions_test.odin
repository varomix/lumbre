package lumbre_realtime
import "core:testing"

@(test)
test_camera_material_and_environment_edits_preserve_geometry_revision :: proc(t: ^testing.T) {
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)
	before := geometry_revision(&scene)
	scene.camera.origin.x = 37
	scene.materials[0].albedo = {0.3, 0.2, 0.1}
	scene.environment.rotation = 1.2
	testing.expect_value(t, geometry_revision(&scene), before)
	scene.nodes[0].local_transform[0, 3] = 2
	testing.expect(t, geometry_revision(&scene) != before, "transforms must invalidate geometry")
	scene.nodes[0].local_transform[0, 3] = 0
	scene.meshes[0].triangles[0].mat_idx = 0
	testing.expect(t, geometry_revision(&scene) != before, "material assignment must invalidate batches")
}

@(test)
test_material_refresh_does_not_touch_geometry :: proc(t: ^testing.T) {
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)
	batches, verts, _, _, ok := scene_build_cpu(&scene)
	testing.expect(t, ok)
	defer scene_free_cpu(batches, verts)
	gpu := Scene_GPU{batches = batches}
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
test_visible_ranges_merge_without_crossing_culled_geometry :: proc(t: ^testing.T) {
	batches := []Draw_Batch{
		{first_vertex = 0, vertex_count = 3, material_index = 0, bounds_min = {-0.5, -0.5, 0.1}, bounds_max = {0.5, 0.5, 0.9}},
		{first_vertex = 3, vertex_count = 6, material_index = 0, bounds_min = {-0.5, -0.5, 0.1}, bounds_max = {0.5, 0.5, 0.9}},
		{first_vertex = 9, vertex_count = 3, material_index = 1, bounds_min = {2, 2, 2}, bounds_max = {3, 3, 3}},
		{first_vertex = 12, vertex_count = 6, material_index = 1, bounds_min = {-0.5, -0.5, 0.1}, bounds_max = {0.5, 0.5, 0.9}},
	}
	vp := (matrix[4, 4]f32)(1)
	first, next, count := visible_range(batches, 0, vp, true)
	testing.expect_value(t, first, 0)
	testing.expect_value(t, next, 2)
	testing.expect_value(t, count, 9)
	first, next, count = visible_range(batches, next, vp, false)
	testing.expect_value(t, first, 3)
	testing.expect_value(t, next, 4)
	testing.expect_value(t, count, 6)
}
