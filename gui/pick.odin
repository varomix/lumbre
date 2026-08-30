package main

// Viewport picking.
//
// Clicking a surface resolves it to a material and drives the Material panel,
// which turns "I want this surface rougher" into two clicks instead of hunting
// through a material list.
//
// The ray is traced on the CPU by brute force. That sounds wasteful next to the
// GPU's acceleration structure, but it is one ray on a click, and it avoids a
// GPU readback, a pick pass in the kernel, and any synchronisation with the
// render worker.
//
// The ray is transformed into each node's local space once rather than
// transforming three vertices per triangle, and triangles are iterated by
// reference — `Triangle` is nine dvec3s plus flags, over 200 bytes to copy.
// `t` is preserved by the change of basis, so hits stay comparable across
// nodes.
//
// Cost on the 157k-triangle guitar is 35-85 ms per pick, the spread being
// contention with the render worker rather than anything in this code; the
// scripting bridge itself is 0.01 ms, so that is all ray tracing. Neither of
// the two changes above moved it much, which says the loop is simply linear
// and memory-bound. That is fine for a click and wrong for anything
// continuous: a hover highlight or a marquee select wants the BVH, not this.
//
// Note this selects a *material*, not a USD prim: the importer flattens meshes
// without recording which prim each triangle came from, so a triangle cannot be
// traced back to the tree. Prim-level selection needs that provenance recorded
// at import.

import "core:math"

import lc "../core"
import m "core:math/linalg/glsl"

Pick_Result :: struct {
	hit:      bool,
	material: int,
	distance: f64,
	point:    lc.Vec3,
	normal:   lc.Vec3,
}

// `u` and `v` are normalised viewport coordinates with the origin bottom-left,
// matching the camera basis the renderer samples with.
pick_at :: proc(scene: ^lc.Scene, u, v: f64) -> Pick_Result {
	cam := scene.camera
	origin := cam.origin
	dir := cam.lower_left_corner + u * cam.horizontal + v * cam.vertical - cam.origin

	best := Pick_Result{distance = math.F64_MAX}

	lc.compute_world_transforms(scene.nodes)
	for node in scene.nodes {
		if node.mesh_idx < 0 || int(node.mesh_idx) >= len(scene.meshes) {
			continue
		}
		mesh := scene.meshes[node.mesh_idx]
		has_override :=
			node.material_override_idx >= 0 &&
			int(node.material_override_idx) < len(scene.materials)

		// One inverse per node instead of three point transforms per triangle.
		inv := m.inverse(node.world_transform)
		local_origin := lc.transform_point(origin, inv)
		local_dir := lc.transform_dir(dir, inv)

		for &tri in mesh.triangles {
			t, ok := ray_triangle(local_origin, local_dir, tri.v0, tri.v1, tri.v2)
			if !ok || t >= best.distance {
				continue
			}

			mat := has_override ? int(node.material_override_idx) : int(tri.mat_idx)
			if mat < 0 || mat >= len(scene.materials) {
				mat = 0
			}
			best.hit = true
			best.distance = t
			best.material = mat
			// Report in world space; the caller has no idea about node bases.
			best.point = origin + dir * t
			best.normal = lc.transform_normal_dir(
				m.normalize(m.cross(tri.v1 - tri.v0, tri.v2 - tri.v0)),
				node.world_transform,
			)
		}
	}

	// Spheres are rendered as icospheres on the GPU but kept analytic here;
	// solving directly is both simpler and exact.
	for s, i in scene.spheres {
		t, ok := ray_sphere(origin, dir, s.center, s.radius)
		if !ok || t >= best.distance {
			continue
		}
		best.hit = true
		best.distance = t
		// Sphere materials are appended after the scene's own, in order.
		best.material = clamp(len(scene.materials) - len(scene.spheres) + i, 0, max(len(scene.materials) - 1, 0))
		best.point = origin + dir * t
		best.normal = m.normalize(best.point - s.center)
	}

	if !best.hit {
		best.distance = 0
	}
	return best
}

// Möller-Trumbore. `dir` is deliberately not normalised, so `t` comes back in
// units of `dir` and can be compared directly against other hits.
@(private = "file")
ray_triangle :: proc(origin, dir, v0, v1, v2: lc.Vec3) -> (f64, bool) {
	EPSILON :: 1.0e-9

	edge1 := v1 - v0
	edge2 := v2 - v0
	h := m.cross(dir, edge2)
	a := m.dot(edge1, h)
	if a > -EPSILON && a < EPSILON {
		return 0, false // parallel
	}

	f := 1.0 / a
	s := origin - v0
	u := f * m.dot(s, h)
	if u < 0.0 || u > 1.0 {
		return 0, false
	}

	q := m.cross(s, edge1)
	v := f * m.dot(dir, q)
	if v < 0.0 || u + v > 1.0 {
		return 0, false
	}

	t := f * m.dot(edge2, q)
	if t <= EPSILON {
		return 0, false // behind the camera
	}
	return t, true
}

@(private = "file")
ray_sphere :: proc(origin, dir, centre: lc.Vec3, radius: f64) -> (f64, bool) {
	oc := origin - centre
	a := m.dot(dir, dir)
	half_b := m.dot(oc, dir)
	c := m.dot(oc, oc) - radius * radius
	disc := half_b * half_b - a * c
	if disc < 0 {
		return 0, false
	}
	sqrt_d := math.sqrt(disc)

	t := (-half_b - sqrt_d) / a
	if t <= 1.0e-9 {
		t = (-half_b + sqrt_d) / a
		if t <= 1.0e-9 {
			return 0, false
		}
	}
	return t, true
}
