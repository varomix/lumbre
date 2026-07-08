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
	Principled,
	Emissive,
}

Material :: struct {
	kind:             Material_Kind,
	albedo:           Color,
	fuzz:             f64,
	ir:               f64,
	emission:         Color,
	emission_strength: f64,
	roughness:        f64,
	metallic:         f64,
}

Hit_Record :: struct {
	p:          Point3,
	normal:     Vec3,
	t:          f64,
	front_face: bool,
	material:   Material,
	uv:         Vec3,
	mat_idx:    i32,
}

Sphere :: struct {
	center:   Point3,
	radius:   f64,
	material: Material,
}

Triangle :: struct {
	v0, v1, v2: Point3,
	n0, n1, n2: Vec3,
	uv0, uv1, uv2: Vec3,
	mat_idx: i32,
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

Light_Kind :: enum {
	Quad,
	Sphere,
	Mesh,
}

Light :: struct {
	kind:      Light_Kind,
	position:  Point3,
	u:         Vec3,
	v:         Vec3,
	intensity: Color,
	area:      f64,
	two_sided: bool,
}

Mesh :: struct {
	name:          string,
	triangles:     []Triangle,
	material:      Material,
	transform:     m.mat4,
}

SceneNode :: struct {
	local_transform:      m.mat4,
	world_transform:      m.mat4,
	mesh_idx:             i32,
	material_override_idx: i32,
	parent:               i32,
}

Scene :: struct {
	meshes:    []Mesh,
	spheres:   []Sphere,
	nodes:     []SceneNode,
	materials: []Material,
	lights:    []Light,
	camera:    Camera,
}

Render_Config :: struct {
	image_width:       i32,
	image_height:      i32,
	samples_per_pixel: i32,
	max_depth:         i32,
	max_radiance:      f64,
	roughness_cutoff:  f64,
	file_output:       cstring,
	scene_file:        cstring,
	debug_mode:        i32,
	use_gpu:           bool,
}
