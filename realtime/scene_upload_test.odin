package lumbre_realtime

// Checks the CPU half of the scene upload: the batching, the tangent basis and
// the bounds. The GPU half needs a device, so it is covered by actually running
// the app rather than here.
//
//   odin test realtime
//
// Batching is what makes this worth testing. Every triangle must end up in
// exactly one batch, batches must be contiguous vertex ranges, and each must
// carry the material its triangles actually reference — an off-by-one there
// shades the whole scene with the wrong material, which looks plausible enough
// in a screenshot to go unnoticed.

import "core:math"
import "core:testing"

import m "core:math/linalg/glsl"

import lc "../core"

// Edges are deliberately longer than one UV unit (3 and 2 against a 0..1 UV
// range), so the raw UV-gradient tangent comes out at length 3 rather than
// coincidentally unit — otherwise the normalization below is untested.
@(private = "file")
tri :: proc(mat_idx: i32, y: f64) -> lc.Triangle {
	return lc.Triangle {
		v0 = {0, y, 0},
		v1 = {3, y, 0},
		v2 = {0, y, 2},
		n0 = {0, 1, 0},
		n1 = {0, 1, 0},
		n2 = {0, 1, 0},
		uv0 = {0, 0, 0},
		uv1 = {1, 0, 0},
		uv2 = {0, 1, 0},
		has_uv = true,
		mat_idx = mat_idx,
	}
}

// A scene whose triangles are deliberately INTERLEAVED by material, so a
// grouping that only works on already-sorted input fails here.
@(private = "file")
make_test_scene :: proc() -> (lc.Scene, []lc.Triangle) {
	tris := make([]lc.Triangle, 6)
	tris[0] = tri(2, 0)
	tris[1] = tri(0, 1)
	tris[2] = tri(2, 2)
	tris[3] = tri(1, 3)
	tris[4] = tri(0, 4)
	tris[5] = tri(2, 5)

	meshes := make([]lc.Mesh, 1)
	meshes[0] = lc.Mesh{triangles = tris, transform = m.mat4(1)}

	nodes := make([]lc.SceneNode, 1)
	nodes[0] = lc.SceneNode {
		local_transform = m.mat4(1),
		world_transform = m.mat4(1),
		mesh_idx = 0,
		material_override_idx = -1,
		parent = -1,
	}

	materials := make([]lc.Material, 3)
	materials[0] = lc.Material{kind = .Principled, albedo = {1, 0, 0}, roughness = 0.25}
	materials[1] = lc.Material{kind = .Principled, albedo = {0, 1, 0}, roughness = 0.50}
	materials[2] = lc.Material{kind = .Principled, albedo = {0, 0, 1}, roughness = 0.75}

	return lc.Scene{meshes = meshes, nodes = nodes, materials = materials}, tris
}

@(private = "file")
destroy_test_scene :: proc(s: ^lc.Scene) {
	for mesh in s.meshes {
		delete(mesh.triangles)
	}
	delete(s.meshes)
	delete(s.nodes)
	delete(s.materials)
}

@(test)
test_batches_cover_every_triangle_once :: proc(t: ^testing.T) {
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)

	batches, verts, lo, hi, ok := scene_build_cpu(&scene)
	defer scene_free_cpu(batches, verts)
	testing.expect(t, ok, "build must succeed")

	// Three materials, so three batches — not six, which is what one draw per
	// triangle would give.
	testing.expectf(t, len(batches) == 3, "got %d batches, want 3", len(batches))

	total: u32 = 0
	next_expected: u32 = 0
	for b in batches {
		testing.expectf(
			t, b.first_vertex == next_expected,
			"batch starts at %d, want %d (ranges must be contiguous)", b.first_vertex, next_expected,
		)
		next_expected += b.vertex_count
		total += b.vertex_count
	}
	testing.expectf(t, total == u32(len(verts)), "batches cover %d of %d vertices", total, len(verts))
	testing.expectf(t, total == 18, "got %d vertices, want 6 triangles * 3", total)

	// Materials 0, 1, 2 have 2, 1 and 3 triangles: the batch sizes must follow
	// the material order, not the authored triangle order.
	want := [3]u32{6, 3, 9}
	for b, i in batches {
		testing.expectf(t, b.vertex_count == want[i], "batch %d has %d vertices, want %d", i, b.vertex_count, want[i])
	}

	_ = lo
	_ = hi
}

@(test)
test_batch_carries_its_own_material :: proc(t: ^testing.T) {
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)

	batches, verts, _, _, ok := scene_build_cpu(&scene)
	defer scene_free_cpu(batches, verts)
	testing.expect(t, ok, "build must succeed")

	// Red/green/blue with distinct roughness, so a batch shaded from the wrong
	// material is unmistakable.
	want_albedo := [3][3]f32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	want_rough := [3]f32{0.25, 0.50, 0.75}

	for b, i in batches {
		c := b.material.base_color
		testing.expectf(
			t,
			c.x == want_albedo[i].x && c.y == want_albedo[i].y && c.z == want_albedo[i].z,
			"batch %d albedo (%v %v %v), want %v", i, c.x, c.y, c.z, want_albedo[i],
		)
		testing.expectf(
			t, math.abs(b.material.params.x - want_rough[i]) < 1e-6,
			"batch %d roughness %v, want %v", i, b.material.params.x, want_rough[i],
		)
	}
}

@(test)
test_bounds_and_tangents :: proc(t: ^testing.T) {
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)

	batches, verts, lo, hi, ok := scene_build_cpu(&scene)
	defer scene_free_cpu(batches, verts)
	testing.expect(t, ok, "build must succeed")

	// The test triangles span x in [0,3], z in [0,2] and y in [0,5].
	testing.expectf(t, lo == [3]f32{0, 0, 0}, "bounds min %v", lo)
	testing.expectf(t, hi == [3]f32{3, 5, 2}, "bounds max %v", hi)

	// UVs run along +x and +z with the normal on +y, so the tangent must be
	// unit length and lie in the triangle's plane (perpendicular to the
	// normal). A tangent that picked up a y component would tilt every normal
	// map sample.
	for v in verts {
		tg := [3]f32{v.tangent.x, v.tangent.y, v.tangent.z}
		length := math.sqrt(tg.x * tg.x + tg.y * tg.y + tg.z * tg.z)
		testing.expectf(t, math.abs(length - 1) < 1e-5, "tangent length %v", length)

		d := tg.x * v.normal.x + tg.y * v.normal.y + tg.z * v.normal.z
		testing.expectf(t, math.abs(d) < 1e-5, "tangent . normal = %v, want 0", d)
	}
}
