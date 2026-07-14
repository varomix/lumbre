package lumbre_core

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
	// Anisotropy in [0,1]: 0 = isotropic GGX (unchanged), 1 = maximally
	// stretched highlight along the surface tangent. Reduces exactly to the
	// isotropic path at 0. See plans/PRINCIPLED_BSDF.md Phase B.
	anisotropic:       f64,
	// Specular transmission (glass) in [0,1]: 0 = opaque, 1 = pure dielectric
	// glass using `ir` as the index of refraction, tinted by `albedo`. Rough
	// at high `roughness` (microfacet BTDF), exact at roughness 0. See
	// plans/PRINCIPLED_BSDF.md Phase C.
	spec_trans:        f64,
	// Volumetric subsurface lobe imported from MaterialX standard_surface.
	// `subsurface_radius * subsurface_scale` is the RGB mean free path.
	subsurface:        f64,
	subsurface_color:  Color,
	subsurface_radius: Color,
	subsurface_scale:  f64,
	albedo_tex:        TextureMap,
	// glTF metallicRoughness packing: G = roughness, B = metallic. Linear.
	metallic_roughness_tex: TextureMap,
	// Tangent-space normal map, OpenGL convention (green = +Y = up). Linear.
	normal_tex:        TextureMap,
	normal_scale:      f64,
	// Emissive map, sRGB. Modulated by `emission`.
	emissive_tex:      TextureMap,
}

// True when any of the material's maps need per-hit UVs.
material_needs_uv :: proc(mat: Material) -> bool {
	return mat.albedo_tex.has_data ||
		mat.metallic_roughness_tex.has_data ||
		mat.normal_tex.has_data ||
		mat.emissive_tex.has_data
}

destroy_material_textures :: proc(mat: ^Material) {
	destroy_texture(&mat.albedo_tex)
	destroy_texture(&mat.metallic_roughness_tex)
	destroy_texture(&mat.normal_tex)
	destroy_texture(&mat.emissive_tex)
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
	has_uv: b32,
	mat_idx: i32,
}

AABB :: struct {
	min: Vec3,
	max: Vec3,
}

BVH_Node :: struct {
	left, right: i32,
	start, end:   i32,
	aabb:         AABB,
}

Rng :: struct {
	state: u64,
}

Light_Kind :: enum {
	Quad,     // rectangle area light: corner `position` + edge vectors `u`, `v`
	Sphere,   // spherical area light: `position` center, `radius`
	Mesh,     // emissive-triangle light (handled via material emission)
	Disc,     // disc area light: `position` center, `direction` normal, `radius`
	Cylinder, // cylinder area light: `position` base center, `direction` axis, `radius`, `height`
	Point,    // delta point light: `position`, `intensity` (radiant intensity)
	Spot,     // delta cone light: `position`, `direction`, `cos_inner`/`cos_outer`
	Distant,  // directional / sun: `direction`, `angular_radius` for soft shadows
	Dome,     // HDRI environment; radiance comes from `Scene.environment`
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
	// Extended fields for the analytic / shape lights.
	direction:      Vec3, // spot & distant aim; disc & cylinder axis (normalized)
	cos_inner:      f64,  // spot: cosine of the inner (full-intensity) cone half-angle
	cos_outer:      f64,  // spot: cosine of the outer (zero-intensity) cone half-angle
	angular_radius: f64,  // distant: half-angle of the source disc (0 = hard sun)
	height:         f64,  // cylinder: length along `direction`
}

Light_Sample :: struct {
	point:     Point3,
	normal:    Vec3,
	emission:  Color,
	pdf:       f64,
	direction: Vec3,
	distance:  f64,
	// A delta light (point/spot/hard distant) is sampled with certainty: its
	// `emission` already folds in any distance falloff, `pdf` is 1, and NEE must
	// not apply an MIS weight or a BSDF-sampling counterpart to it.
	delta:     bool,
}

// Equirectangular HDRI environment (dome light). `pixels` is linear RGB, row
// major, top row (v=1, +Y pole) first, matching stbi.loadf with a top-left
// origin. Importance sampling uses a Pharr/PBRT 2D piecewise-constant
// distribution built from luminance weighted by sin(theta).
Environment :: struct {
	width, height: i32,
	pixels:        []f32, // linear RGB, len = width*height*3
	rotation:      f64,   // radians about +Y applied when looking up a direction
	intensity:     f64,
	has_data:      bool,
	// 2D sampling tables. `marginal_cdf` has `height+1` entries selecting a row;
	// `conditional_cdf` has `height*(width+1)` entries, one CDF per row over
	// columns. `func_int` is the overall integral used to normalize the pdf.
	marginal_cdf:    []f32,
	conditional_cdf: []f32,
	func_int:        f64,
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
	meshes:      []Mesh,
	spheres:     []Sphere,
	nodes:       []SceneNode,
	materials:   []Material,
	lights:      []Light,
	environment: Environment,
	camera:      Camera,
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
	exr_compress:      b32,
	// Stage 5: OpenImageDenoise HDR ray-tracing denoiser (post-process).
	denoise_enabled: b32,
	// Render sequence: inclusive [start, end]. Both <= 0 means
	// "render a single frame using the current --output path."
	frame_start:      i32,
	frame_end:        i32,
	frame_padding:    i32, // digits for the output filename (e.g. 4 -> "0001")
	// HDRI environment (dome light).
	hdri_file:       cstring,
	hdri_rotation:   f64, // degrees about +Y
	hdri_intensity:  f64,
	// Directional / sun light.
	sun_enabled:   bool,
	sun_dir:       Vec3,  // direction the light travels (points away from the sun)
	sun_color:     Color, // radiance
	sun_angle:     f64,   // angular diameter in degrees (0 = hard shadow)
	// USD scene authoring (plans/USD_SCENE_FORMAT.md). Empty means "use the
	// first Camera prim found in the stage."
	usd_camera_name: cstring,
	// Uniform subdivision level for USD catmullClark/loop/bilinear meshes.
	// USD meshes default to catmullClark, so their authored faces are a
	// control cage; 2 is the common render default, 0 renders the raw cage.
	usd_subdiv_level: i32,
	// Diagnostic: force `anisotropic` on every Principled material (no importer
	// authors it yet). 0 = off. See plans/PRINCIPLED_BSDF.md Phase B.
	force_anisotropic: f64,
	// Diagnostic: force `spec_trans` (glass) on every Principled material.
	// 0 = off. See plans/PRINCIPLED_BSDF.md Phase C.
	force_spec_trans: f64,
}

// CPU-side texture map. Pixels are stored as RGBA8 in row-major order.
// `width * height * 4` bytes total when `has_data` is true.
TextureMap :: struct {
	width:    i32,
	height:   i32,
	pixels:   []u8, // RGBA8, length = width * height * 4
	has_data: bool,
	// True when the RGB channels are sRGB-encoded and must be converted to
	// linear before shading. Color maps (base color / diffuse) are sRGB;
	// data maps (normal, metallic-roughness) are linear.
	srgb:     bool,
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

clone_texture :: proc(tex: TextureMap, allocator := context.allocator) -> TextureMap {
	if !tex.has_data || len(tex.pixels) == 0 {
		return TextureMap{}
	}
	out := TextureMap{
		width    = tex.width,
		height   = tex.height,
		pixels   = make([]u8, len(tex.pixels), allocator),
		has_data = true,
		srgb     = tex.srgb,
	}
	copy(out.pixels, tex.pixels)
	return out
}
