package main

import m "core:math/linalg/glsl"

Vec3 :: m.dvec3
Point3 :: m.dvec3
Color :: m.dvec3

MAX_SPHERES :: 500

Ray :: struct {
	origin:    Point3,
	direction: Vec3,
}

Camera :: struct {
	origin:            Point3,
	lower_left_corner: Point3,
	horizontal:        Vec3,
	vertical:          Vec3,
	u:                 Vec3,
	v:                 Vec3,
	lens_radius:       f64,
}

Material_Kind :: enum {
	Lambertian,
	Metal,
	Dielectric,
}

Material :: struct {
	kind:   Material_Kind,
	albedo: Color,
	fuzz:   f64,
	ir:     f64,
}

Hit_Record :: struct {
	p:          Point3,
	normal:     Vec3,
	t:          f64,
	front_face: bool,
	material:   Material,
}

Sphere :: struct {
	center:   Point3,
	radius:   f64,
	material: Material,
}

AABB :: struct {
	min: Vec3,
	max: Vec3,
}

MAX_BVH_NODES :: 1024

BVH_Node :: struct {
	left, right: i32,
	start, end:   i32,
	aabb:         AABB,
}

Rng :: struct {
	state: u64,
}
