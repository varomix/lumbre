package lumbre_core

import m "core:math/linalg/glsl"
import "core:slice"

hit_sphere :: proc(sphere: Sphere, r: Ray, ray_t_min, ray_t_max: f64, rec: ^Hit_Record) -> bool {
	oc := r.origin - sphere.center
	a := length_squared(r.direction)
	half_b := m.dot(oc, r.direction)
	c := length_squared(oc) - sphere.radius * sphere.radius

	discriminant := half_b * half_b - a * c
	if discriminant < 0.0 {
		return false
	}

	sqrtd := m.sqrt(discriminant)
	root := (-half_b - sqrtd) / a
	if root <= ray_t_min || ray_t_max <= root {
		root = (-half_b + sqrtd) / a
		if root <= ray_t_min || ray_t_max <= root {
			return false
		}
	}

	rec.t = root
	rec.p = at(r, rec.t)
	outward_normal := (rec.p - sphere.center) / sphere.radius
	set_face_normal(rec, r, outward_normal)
	rec.material = sphere.material
	return true
}

sphere_aabb :: proc(s: Sphere) -> AABB {
	r := Vec3{s.radius, s.radius, s.radius}
	return AABB{s.center - r, s.center + r}
}

surrounding_box :: proc(b0, b1: AABB) -> AABB {
	small := m.min(b0.min, b1.min)
	big := m.max(b0.max, b1.max)
	return AABB{small, big}
}

aabb_hit :: proc(b: AABB, r: Ray, t_min, t_max: f64) -> bool {
	tmin := t_min
	tmax := t_max
	for axis in 0 ..< 3 {
		inv_d := 1.0 / r.direction[axis]
		t0 := (b.min[axis] - r.origin[axis]) * inv_d
		t1 := (b.max[axis] - r.origin[axis]) * inv_d
		if inv_d < 0.0 {
			t0, t1 = t1, t0
		}
		tmin = m.max(t0, tmin)
		tmax = m.min(t1, tmax)
		if tmax <= tmin {
			return false
		}
	}
	return true
}

// `nodes` must hold at least `bvh_node_capacity(end - start)` entries.
build_bvh :: proc(world: []Sphere, nodes: []BVH_Node, node_count: ^i32, start, end: i32) -> i32 {
	node_idx := node_count^
	assert(int(node_idx) < len(nodes), "BVH node overflow: nodes buffer too small for this primitive count")
	node_count^ += 1
	node := &nodes[node_idx]

	node.aabb = sphere_aabb(world[start])
	cmin, cmax := world[start].center, world[start].center
	for s in world[start + 1:end] {
		node.aabb = surrounding_box(node.aabb, sphere_aabb(s))
		cmin = m.min(cmin, s.center)
		cmax = m.max(cmax, s.center)
	}
	node.start = start
	node.end = end
	axis, splittable := bvh_split_axis(cmin, cmax)
	if end - start <= BVH_LEAF_SIZE || !splittable {
		node.left = -1
		node.right = -1
		return node_idx
	}

	switch axis {
	case 0: slice.sort_by(world[start:end], proc(a, b: Sphere) -> bool { return a.center.x < b.center.x })
	case 1: slice.sort_by(world[start:end], proc(a, b: Sphere) -> bool { return a.center.y < b.center.y })
	case:   slice.sort_by(world[start:end], proc(a, b: Sphere) -> bool { return a.center.z < b.center.z })
	}
	mid := start + (end - start) / 2
	node.axis = axis
	node.left = build_bvh(world, nodes, node_count, start, mid)
	node.right = build_bvh(world, nodes, node_count, mid, end)
	return node_idx
}

// Primitives a leaf may hold. Splitting down to one per leaf doubled the
// node count and the box tests for no gain in the primitive tests saved.
BVH_LEAF_SIZE :: 4

// The axis along which the centroids spread furthest: splitting there gives
// children that overlap least. The builder used to pick an axis at random,
// which on a flat or elongated mesh often cut across its thin side. False when
// every centroid coincides and no split can separate them.
bvh_split_axis :: proc(cmin, cmax: Vec3) -> (axis: i32, splittable: bool) {
	extent := cmax - cmin
	axis = 0
	if extent.y > extent[axis] { axis = 1 }
	if extent.z > extent[axis] { axis = 2 }
	return axis, extent[axis] > 0
}

triangle_aabb :: proc(t: Triangle) -> AABB {
	min := m.min(m.min(t.v0, t.v1), t.v2)
	max := m.max(m.max(t.v0, t.v1), t.v2)
	eps := 1.0e-6
	for axis in 0 ..< 3 {
		if max[axis] - min[axis] < eps {
			min[axis] -= eps
			max[axis] += eps
		}
	}
	return AABB{min, max}
}

triangle_centroid :: proc(t: Triangle) -> Vec3 {
	return (t.v0 + t.v1 + t.v2) / 3.0
}

// Nodes a median-split BVH can need for `n` primitives: at most n leaves (one
// primitive each) and n-1 interior nodes. Leaves hold up to BVH_LEAF_SIZE, so
// the build usually uses far fewer.
bvh_node_capacity :: proc(n: int) -> int {
	if n <= 0 {
		return 0
	}
	return 2 * n - 1
}

// `nodes` must hold at least `bvh_node_capacity(end - start)` entries.
build_triangle_bvh :: proc(triangles: []Triangle, nodes: []BVH_Node, node_count: ^i32, start, end: i32) -> i32 {
	node_idx := node_count^
	assert(int(node_idx) < len(nodes), "BVH node overflow: nodes buffer too small for this triangle count")
	node_count^ += 1
	node := &nodes[node_idx]

	node.aabb = triangle_aabb(triangles[start])
	cmin := triangle_centroid(triangles[start])
	cmax := cmin
	for t in triangles[start + 1:end] {
		node.aabb = surrounding_box(node.aabb, triangle_aabb(t))
		c := triangle_centroid(t)
		cmin = m.min(cmin, c)
		cmax = m.max(cmax, c)
	}
	node.start = start
	node.end = end
	axis, splittable := bvh_split_axis(cmin, cmax)
	if end - start <= BVH_LEAF_SIZE || !splittable {
		node.left = -1
		node.right = -1
		return node_idx
	}

	// Sums rather than centroids: the same order, one division fewer per
	// comparison.
	switch axis {
	case 0: slice.sort_by(triangles[start:end], proc(a, b: Triangle) -> bool { return a.v0.x + a.v1.x + a.v2.x < b.v0.x + b.v1.x + b.v2.x })
	case 1: slice.sort_by(triangles[start:end], proc(a, b: Triangle) -> bool { return a.v0.y + a.v1.y + a.v2.y < b.v0.y + b.v1.y + b.v2.y })
	case:   slice.sort_by(triangles[start:end], proc(a, b: Triangle) -> bool { return a.v0.z + a.v1.z + a.v2.z < b.v0.z + b.v1.z + b.v2.z })
	}
	mid := start + (end - start) / 2
	node.axis = axis
	node.left = build_triangle_bvh(triangles, nodes, node_count, start, mid)
	node.right = build_triangle_bvh(triangles, nodes, node_count, mid, end)
	return node_idx
}

hit_triangle_bvh :: proc(triangles: []Triangle, nodes: []BVH_Node, node_idx: i32, r: Ray, ray_t_min, ray_t_max: f64, rec: ^Hit_Record) -> bool {
	node := &nodes[node_idx]
	if !aabb_hit(node.aabb, r, ray_t_min, ray_t_max) {
		return false
	}

	if node.left == -1 {
		temp_rec: Hit_Record
		hit_anything := false
		closest_so_far := ray_t_max
		for s in node.start ..< node.end {
			if hit_triangle(triangles[s], r, ray_t_min, closest_so_far, &temp_rec) {
				hit_anything = true
				closest_so_far = temp_rec.t
				rec^ = temp_rec
			}
		}
		return hit_anything
	}

	temp_rec: Hit_Record
	hit_anything := false
	closest_so_far := ray_t_max

	// Nearer child first: a hit there shrinks closest_so_far, and the far
	// child's box test can then reject it outright.
	first, second := node.left, node.right
	if r.direction[node.axis] < 0 {
		first, second = second, first
	}
	if hit_triangle_bvh(triangles, nodes, first, r, ray_t_min, closest_so_far, &temp_rec) {
		hit_anything = true
		closest_so_far = temp_rec.t
		rec^ = temp_rec
	}
	if hit_triangle_bvh(triangles, nodes, second, r, ray_t_min, closest_so_far, &temp_rec) {
		hit_anything = true
		closest_so_far = temp_rec.t
		rec^ = temp_rec
	}
	return hit_anything
}

hit_bvh :: proc(world: []Sphere, nodes: []BVH_Node, node_idx: i32, r: Ray, ray_t_min, ray_t_max: f64, rec: ^Hit_Record) -> bool {
	node := &nodes[node_idx]
	if !aabb_hit(node.aabb, r, ray_t_min, ray_t_max) {
		return false
	}

	if node.left == -1 {
		temp_rec: Hit_Record
		hit_anything := false
		closest_so_far := ray_t_max
		for s in node.start ..< node.end {
			if hit_sphere(world[s], r, ray_t_min, closest_so_far, &temp_rec) {
				hit_anything = true
				closest_so_far = temp_rec.t
				rec^ = temp_rec
			}
		}
		return hit_anything
	}

	temp_rec: Hit_Record
	hit_anything := false
	closest_so_far := ray_t_max

	// Nearer child first: a hit there shrinks closest_so_far, and the far
	// child's box test can then reject it outright.
	first, second := node.left, node.right
	if r.direction[node.axis] < 0 {
		first, second = second, first
	}
	if hit_bvh(world, nodes, first, r, ray_t_min, closest_so_far, &temp_rec) {
		hit_anything = true
		closest_so_far = temp_rec.t
		rec^ = temp_rec
	}
	if hit_bvh(world, nodes, second, r, ray_t_min, closest_so_far, &temp_rec) {
		hit_anything = true
		closest_so_far = temp_rec.t
		rec^ = temp_rec
	}
	return hit_anything
}
