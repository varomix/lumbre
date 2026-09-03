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

// A scene with several nodes, each a separate mesh, whose materials interleave
// across nodes. That combination is what makes the material sort able to
// decouple a triangle from its node: neither the node order nor the material
// order matches the final vertex order.
@(private = "file")
make_multi_node_scene :: proc() -> lc.Scene {
	NODES :: 4
	meshes := make([]lc.Mesh, NODES)
	nodes := make([]lc.SceneNode, NODES)

	for i in 0 ..< NODES {
		tris := make([]lc.Triangle, 2)
		// Materials cycle the opposite way to the node order.
		tris[0] = tri(i32((NODES - 1 - i) % 3), f64(i))
		tris[1] = tri(i32(i % 3), f64(i) + 0.5)
		meshes[i] = lc.Mesh{triangles = tris, transform = m.mat4(1)}
		nodes[i] = lc.SceneNode {
			local_transform = m.mat4(1),
			world_transform = m.mat4(1),
			mesh_idx = i32(i),
			material_override_idx = -1,
			parent = -1,
		}
	}

	materials := make([]lc.Material, 3)
	for i in 0 ..< 3 {
		materials[i] = lc.Material{kind = .Principled, roughness = f64(i) * 0.25}
	}
	return lc.Scene{meshes = meshes, nodes = nodes, materials = materials}
}

@(test)
test_instance_id_survives_the_material_sort :: proc(t: ^testing.T) {
	scene := make_multi_node_scene()
	defer destroy_test_scene(&scene)

	batches, verts, _, _, ok := scene_build_cpu(&scene)
	defer scene_free_cpu(batches, verts)
	testing.expect(t, ok, "build must succeed")

	// Every node must be represented, and only real node indices may appear.
	seen: [4]int
	for v in verts {
		id := int(v.instance)
		testing.expectf(t, id >= 0 && id < 4, "instance id %d outside the node range", id)
		seen[id] += 1
	}
	for count, node in seen {
		// Two triangles per node, three vertices each.
		testing.expectf(t, count == 6, "node %d has %d vertices, want 6", node, count)
	}

	// The three vertices of any triangle must agree: a per-vertex id that
	// disagreed within a face would tear labels along triangle edges.
	for i := 0; i < len(verts); i += 3 {
		a, b, c := verts[i].instance, verts[i + 1].instance, verts[i + 2].instance
		testing.expectf(t, a == b && b == c, "triangle %d spans instance ids %v %v %v", i / 3, a, b, c)
	}
}

@(test)
test_instance_id_follows_its_geometry :: proc(t: ^testing.T) {
	scene := make_multi_node_scene()
	defer destroy_test_scene(&scene)

	batches, verts, _, _, ok := scene_build_cpu(&scene)
	defer scene_free_cpu(batches, verts)
	testing.expect(t, ok, "build must succeed")
	_ = batches

	// `tri(mat, y)` puts every vertex of a triangle at height y, and
	// make_multi_node_scene gives node i the heights i and i+0.5. So the id a
	// vertex carries must match the node its POSITION came from -- which is the
	// property the sort could break while leaving every count above correct.
	for v in verts {
		want := i32(v.pos.y) // heights i and i+0.5 both floor to i
		testing.expectf(
			t, i32(v.instance) == want,
			"vertex at y=%v carries instance %v, want %v", v.pos.y, v.instance, want,
		)
	}
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

@(test)
test_refresh_picks_up_material_edits :: proc(t: ^testing.T) {
	scene, _ := make_test_scene()
	defer destroy_test_scene(&scene)

	batches, verts, _, _, ok := scene_build_cpu(&scene)
	defer scene_free_cpu(batches, verts)
	testing.expect(t, ok, "build must succeed")

	// Edit a material the way the panel or a script does. The path tracer
	// updates its material buffer in place without bumping the scene key, so
	// the rasterizer cannot rely on a rebuild to notice.
	scene.materials[1].albedo = {0.1, 0.2, 0.3}
	scene.materials[1].roughness = 0.05

	// Stale until refreshed — the batches hold values copied at build time.
	testing.expect(t, batches[1].material.base_color.x != 0.1, "batch must be stale before a refresh")

	batches_refresh_materials(batches, scene.materials)

	c := batches[1].material.base_color
	testing.expectf(
		t,
		math.abs(c.x - 0.1) < 1e-6 && math.abs(c.y - 0.2) < 1e-6 && math.abs(c.z - 0.3) < 1e-6,
		"albedo after refresh (%v %v %v), want (0.1 0.2 0.3)", c.x, c.y, c.z,
	)
	testing.expectf(
		t, math.abs(batches[1].material.params.x - 0.05) < 1e-6,
		"roughness after refresh %v, want 0.05", batches[1].material.params.x,
	)

	// Geometry must be untouched: an edit that rebuilt the mesh would stall on
	// every slider drag, which is the whole reason this path exists.
	testing.expectf(t, len(batches) == 3, "refresh changed the batch count to %d", len(batches))
	testing.expectf(t, batches[1].vertex_count == 3, "refresh changed a vertex range")
}

@(test)
test_emission_mirrors_the_path_tracer :: proc(t: ^testing.T) {
	// An emissive MAP modulates `emission` alone. glTF leaves strength at zero
	// for these, so folding it in multiplies the map away — which is exactly
	// what hid the damaged helmet's HUD graphics.
	mapped := lc.Material {
		kind = .Principled,
		emission = {1, 1, 1},
		emission_strength = 0, // as glTF leaves it
		emissive_tex = lc.TextureMap{width = 1, height = 1, has_data = true},
	}
	m := material_uniforms(mapped)
	testing.expectf(t, m.emission.x == 1, "mapped emitter emission %v, want 1 (strength must not apply)", m.emission.x)
	testing.expectf(t, m.emission.w == 1, "mapped emitter must flag its map")

	// A pure emitter uses emission * strength, with strength defaulting to 20
	// rather than to zero.
	emitter := lc.Material{kind = .Emissive, emission = {0.5, 0.5, 0.5}, emission_strength = 0}
	e := material_uniforms(emitter)
	testing.expectf(t, math.abs(e.emission.x - 10) < 1e-5, "pure emitter emission %v, want 0.5 * 20", e.emission.x)

	// With emission black it falls back to albedo, as `emissive_radiance` does.
	fallback := lc.Material{kind = .Emissive, albedo = {0.25, 0, 0}, emission_strength = 4}
	f := material_uniforms(fallback)
	testing.expectf(t, math.abs(f.emission.x - 1) < 1e-5, "black-emission emitter %v, want albedo * strength", f.emission.x)

	// Everything else emits nothing, with or without an emission value set.
	plain := lc.Material{kind = .Principled, emission = {1, 1, 1}, emission_strength = 5}
	p := material_uniforms(plain)
	testing.expectf(t, p.emission.x == 0, "a non-emissive material must not emit (%v)", p.emission.x)
}
