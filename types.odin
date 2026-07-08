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
	kind:              Material_Kind,
	albedo:            Color,
	fuzz:              f64,
	ir:                f64,
	emission:          Color,
	emission_strength: f64,
	roughness:         f64,
	metallic:          f64,
	specular:          f64,
	specular_tint:     Color,
	clearcoat:         f64,
	clearcoat_roughness: f64,
	sheen:             f64,
	sheen_tint:        Color,
	albedo_tex:        TextureMap,
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
	radius:    f64,
	two_sided: bool,
}

Light_Sample :: struct {
	point:     Point3,
	normal:    Vec3,
	emission:  Color,
	pdf:       f64,
	direction: Vec3,
	distance:  f64,
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
	glossy_bias:       f64,
	file_output:       cstring,
	scene_file:        cstring,
	debug_mode:        i32,
	use_gpu:           bool,
	gi_cache_enabled:  b32,
	gi_cache_distance: f32,
	gi_cache_normal_angle: f32,
	photon_enabled:    b32,
	photon_count:      i32,
	photon_radius:     f32,
	photon_bounces:    i32,
	enable_aovs:       b32,
}

// CPU-side texture map. Pixels are stored as RGBA8 in row-major order.
// `width * height * 4` bytes total when `has_data` is true.
TextureMap :: struct {
	width:    i32,
	height:   i32,
	pixels:   []u8, // RGBA8, length = width * height * 4
	has_data: bool,
}

make_texture :: proc(width, height: i32) -> TextureMap {
	return TextureMap{
		width    = width,
		height   = height,
		pixels   = make([]u8, int(width) * int(height) * 4),
		has_data = true,
	}
}

destroy_texture :: proc(tex: ^TextureMap) {
	if tex.has_data {
		delete(tex.pixels)
		tex.has_data = false
		tex.width = 0
		tex.height = 0
	}
}
