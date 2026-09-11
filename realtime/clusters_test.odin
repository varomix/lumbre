package lumbre_realtime

// Froxel assignment, which the GPU cannot be asked about after the fact: a
// light dropped from the wrong cell is a dim patch in a corner of one frame,
// not a crash.
//
//   odin test realtime

import "core:testing"

@(private = "file")
test_frame :: proc() -> Camera_Frame {
	return Camera_Frame {
		eye = {0, 0, 0},
		forward = {0, 0, -1},
		right = {1, 0, 0},
		up = {0, 1, 0},
		vfov = 1.0,
		aspect = 16.0 / 9.0,
		focus = 10,
	}
}

@(private = "file")
point_light :: proc(pos: [3]f32, intensity: f32) -> Light_GPU {
	return Light_GPU {
		position = {pos.x, pos.y, pos.z, 0},
		emission = {intensity, intensity, intensity, 0},
		params = {f32(i32(Light_Kind_GPU.Point)), 0, 0, 0},
	}
}

@(test)
test_every_cell_gets_a_valid_range :: proc(t: ^testing.T) {
	g: Cluster_Grid
	defer clusters_destroy(nil, &g)
	f := test_frame()
	lights := []Light_GPU{point_light({0, 0, -10}, 100), point_light({5, 0, -20}, 50)}
	clusters_build(&g, lights, f)

	testing.expectf(t, len(g.ranges) == CLUSTER_COUNT, "got %d cells, want %d", len(g.ranges), CLUSTER_COUNT)
	// Ranges must tile the index list end to end: a gap or an overlap would
	// make a pixel read another cell's lights.
	next: u32 = 0
	for r, i in g.ranges {
		testing.expectf(t, r.offset == next, "cell %d starts at %d, want %d", i, r.offset, next)
		next = r.offset + r.count
		testing.expectf(t, r.count <= u32(MAX_LIGHTS_PER_CLUSTER), "cell %d holds %d lights", i, r.count)
		// Ascending order is what keeps the shading order equal to the
		// unclustered loop's.
		prev: i64 = -1
		for k in 0 ..< r.count {
			id := g.indices[r.offset + k]
			testing.expectf(t, i64(id) > prev, "cell %d lists light %d out of order", i, id)
			testing.expectf(t, int(id) < len(lights), "cell %d lists light %d, only %d exist", i, id, len(lights))
			prev = i64(id)
		}
	}
	testing.expectf(t, next == u32(len(g.indices)), "ranges cover %d of %d indices", next, len(g.indices))
}

@(test)
test_a_distant_light_reaches_every_cell :: proc(t: ^testing.T) {
	g: Cluster_Grid
	defer clusters_destroy(nil, &g)
	sun := Light_GPU {
		direction = {0, -1, 0, 0},
		emission = {1, 1, 1, 0},
		params = {f32(i32(Light_Kind_GPU.Distant)), 0, 0, 0},
	}
	clusters_build(&g, []Light_GPU{sun}, test_frame())
	for r, i in g.ranges {
		testing.expectf(t, r.count == 1, "cell %d has %d lights, want the sun", i, r.count)
	}
}

@(test)
test_a_dim_local_light_is_culled_but_kept_where_it_reaches :: proc(t: ^testing.T) {
	g: Cluster_Grid
	defer clusters_destroy(nil, &g)
	f := test_frame()
	// Dim enough to reach about a unit, and placed in front of the camera.
	clusters_build(&g, []Light_GPU{point_light({0, 0, -10}, 0.001)}, f)

	populated := 0
	for r in g.ranges {
		if r.count > 0 {
			populated += 1
		}
	}
	testing.expect(t, populated > 0, "a light in view must reach some cell")
	testing.expectf(t, populated < CLUSTER_COUNT, "a one-unit light reached all %d cells", CLUSTER_COUNT)

	// The same light, bright enough to cover the scene, must reach far more.
	g2: Cluster_Grid
	defer clusters_destroy(nil, &g2)
	clusters_build(&g2, []Light_GPU{point_light({0, 0, -10}, 1e6)}, f)
	bright := 0
	for r in g2.ranges {
		if r.count > 0 {
			bright += 1
		}
	}
	testing.expectf(t, bright > populated, "a brighter light reached %d cells, dim one reached %d", bright, populated)
}

@(test)
test_lights_behind_the_camera_do_not_fill_the_grid :: proc(t: ^testing.T) {
	g: Cluster_Grid
	defer clusters_destroy(nil, &g)
	f := test_frame()
	// Behind the eye, and far enough that its reach cannot cross the near plane.
	clusters_build(&g, []Light_GPU{point_light({0, 0, 40}, 1)}, f)
	for r, i in g.ranges {
		testing.expectf(t, r.count == 0, "cell %d kept a light behind the camera", i)
	}
}
