package main

import m "core:math/linalg/glsl"
import "core:sort"

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

build_bvh :: proc(world: []Sphere, nodes: ^[MAX_BVH_NODES]BVH_Node, node_count: ^i32, start, end: i32) -> i32 {
	node_idx := node_count^
	node_count^ += 1
	node := &nodes[node_idx]

	span := end - start
	if span == 1 {
		node.left = -1
		node.right = -1
		node.start = start
		node.end = end
		node.aabb = sphere_aabb(world[start])
		return node_idx
	}

	axis := i32(rng_f64_range(&global_bvh_rng, 0.0, 3.0))
	switch axis {
	case 0:
		sort.quick_sort_proc(world[start:end], proc(a, b: Sphere) -> int {
			return -1 if a.center.x < b.center.x else +1 if a.center.x > b.center.x else 0
		})
	case 1:
		sort.quick_sort_proc(world[start:end], proc(a, b: Sphere) -> int {
			return -1 if a.center.y < b.center.y else +1 if a.center.y > b.center.y else 0
		})
	case 2:
		sort.quick_sort_proc(world[start:end], proc(a, b: Sphere) -> int {
			return -1 if a.center.z < b.center.z else +1 if a.center.z > b.center.z else 0
		})
	}

	mid := start + span / 2
	left := build_bvh(world, nodes, node_count, start, mid)
	right := build_bvh(world, nodes, node_count, mid, end)
	node.left = left
	node.right = right
	node.start = start
	node.end = end
	node.aabb = surrounding_box(nodes[left].aabb, nodes[right].aabb)
	return node_idx
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

build_triangle_bvh :: proc(triangles: []Triangle, nodes: ^[MAX_BVH_NODES]BVH_Node, node_count: ^i32, start, end: i32) -> i32 {
	node_idx := node_count^
	node_count^ += 1
	node := &nodes[node_idx]

	span := end - start
	if span == 1 {
		node.left = -1
		node.right = -1
		node.start = start
		node.end = end
		node.aabb = triangle_aabb(triangles[start])
		return node_idx
	}

	axis := i32(rng_f64_range(&global_bvh_rng, 0.0, 3.0))
	switch axis {
	case 0:
		sort.quick_sort_proc(triangles[start:end], proc(a, b: Triangle) -> int {
			ca := triangle_centroid(a); cb := triangle_centroid(b)
			return -1 if ca.x < cb.x else +1 if ca.x > cb.x else 0
		})
	case 1:
		sort.quick_sort_proc(triangles[start:end], proc(a, b: Triangle) -> int {
			ca := triangle_centroid(a); cb := triangle_centroid(b)
			return -1 if ca.y < cb.y else +1 if ca.y > cb.y else 0
		})
	case 2:
		sort.quick_sort_proc(triangles[start:end], proc(a, b: Triangle) -> int {
			ca := triangle_centroid(a); cb := triangle_centroid(b)
			return -1 if ca.z < cb.z else +1 if ca.z > cb.z else 0
		})
	}

	mid := start + span / 2
	left := build_triangle_bvh(triangles, nodes, node_count, start, mid)
	right := build_triangle_bvh(triangles, nodes, node_count, mid, end)
	node.left = left
	node.right = right
	node.start = start
	node.end = end
	node.aabb = surrounding_box(nodes[left].aabb, nodes[right].aabb)
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

	if hit_triangle_bvh(triangles, nodes, node.left, r, ray_t_min, closest_so_far, &temp_rec) {
		hit_anything = true
		closest_so_far = temp_rec.t
		rec^ = temp_rec
	}
	if hit_triangle_bvh(triangles, nodes, node.right, r, ray_t_min, closest_so_far, &temp_rec) {
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

	if hit_bvh(world, nodes, node.left, r, ray_t_min, closest_so_far, &temp_rec) {
		hit_anything = true
		closest_so_far = temp_rec.t
		rec^ = temp_rec
	}
	if hit_bvh(world, nodes, node.right, r, ray_t_min, closest_so_far, &temp_rec) {
		hit_anything = true
		closest_so_far = temp_rec.t
		rec^ = temp_rec
	}
	return hit_anything
}
