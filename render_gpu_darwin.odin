package main

import "core:c"
import "core:fmt"
import "core:strings"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"
import stbi "vendor:stb/image"
import m "core:math/linalg/glsl"
import "core:slice"
import "core:time"
import "output"

// ── GPU data structs (packed for Metal) ──────────────────────────────────────

GPUMaterial :: struct {
	albedo:    [4]f32,
	emission:  [4]f32,
	params0:   [4]f32, // kind, fuzz, ir, roughness
	params1:   [4]f32, // metallic, emission_strength, specular, clearcoat
	params2:   [4]f32, // clearcoat_roughness, sheen, normal_scale, unused
	spec_tint: [4]f32, // rgb = specular_tint
	sheen_tint: [4]f32, // rgb = sheen_tint
	// Each *_info is {pixel_offset, width, height, has_tex} into `tex_pixels`.
	tex_info:  [4]f32, // base color (sRGB)
	mr_info:   [4]f32, // metallic-roughness: G = rough, B = metal (linear)
	nrm_info:  [4]f32, // tangent-space normal map (linear)
	emis_info: [4]f32, // emissive (sRGB)
}

GPULightTriangle :: struct {
	p0:       [4]f32,
	p1:       [4]f32,
	p2:       [4]f32,
	emission: [4]f32,
}

GPUQuadLight :: struct {
	position:  [4]f32,
	u:         [4]f32,
	v:         [4]f32,
	emission:  [4]f32,
}

GPUSphereLight :: struct {
	position:  [4]f32,
	emission:  [4]f32,
	radius:    f32,
	_pad:      [3]f32,
}

GICachePoint :: struct {
	position:  [4]f32,
	normal:    [4]f32,
	irradiance: [4]f32,
}

Photon :: struct {
	position: [4]f32,
	incident: [4]f32,
	power:    [4]f32,
}

GPUSceneData :: struct {
	origin:            [4]f32,
	lower_left:        [4]f32,
	horizontal:        [4]f32,
	vertical:          [4]f32,
	u:                 [4]f32,
	v:                 [4]f32,
	lens_radius:       f32,
	image_width:       i32,
	image_height:      i32,
	samples_per_pixel: i32,
	max_depth:         i32,
	max_radiance:      f32,
	debug_mode:        i32,
	tri_light_count:   i32,
	primitive_count:   i32,
	seed:              u32,
	quad_light_count:   i32,
	sphere_light_count: i32,
	roughness_cutoff:   f32,
	glossy_bias:        f32,
	gi_cache_enabled:   i32,
	gi_cache_distance:  f32,
	gi_cache_normal_angle: f32,
	gi_cache_num_points: i32,
	photon_enabled:     i32,
	photon_count:       i32,
	photon_radius:      f32,
	photon_max_bounces: i32,
}

// Mirrors DenoiseParams in shaders/denoise.metal.
GPUDenoiseParams :: struct {
	width:        i32,
	height:       i32,
	step_width:   i32,
	sigma_color:  f32,
	sigma_normal: f32,
	sigma_depth:  f32,
}

GPUSphere :: struct {
	center:        [4]f32,
	radius:        f32,
	material_kind: i32,
	_fuzz:         f32,
	_ir:           f32,
	albedo:        [4]f32,
}

AxisAlignedBoundingBox :: struct {
	min: [3]f32,
	max: [3]f32,
}

// ── Sphere → mesh converter ─────────────────────────────────────────────────

// Picks a sensible photon-search / GI-cache radius from photon *density*
// rather than geometry. Photons land on surfaces, so the mean spacing between
// stored photons is ~sqrt(total_surface_area / photon_count). Keying off that
// (a) is independent of tessellation — a Cornell wall is one huge quad, a
// helmet panel is thousands of tiny tris, but both want the same gather scale;
// and (b) shrugs off a lone giant primitive (the test scene's radius-1000
// ground "sphere") far better than a bounding-box extent, which would inflate
// the radius into the hundreds and gather the whole map every lookup.
//
// Too-small a radius is as bad as too-large: the hash grid's cells shrink,
// buckets collide, and every gather scans a bloated bucket list — so we clamp
// the result to a fraction of the scene extent on both ends.
auto_gather_radius :: proc(tris: []Triangle, scene_extent: f64, photon_count: int) -> f64 {
	if len(tris) == 0 || photon_count <= 0 {
		return 0.05
	}
	area := 0.0
	for t in tris {
		area += 0.5 * m.length(m.cross(t.v1 - t.v0, t.v2 - t.v0))
	}
	spacing := m.sqrt(area / f64(photon_count))
	// A few multiples of the spacing gathers enough neighbours to smooth
	// indirect light without bleeding across features.
	K :: 4.0
	return clamp(K * spacing, 0.02, scene_extent * 0.1)
}

build_icosphere :: proc(center: Vec3, radius: f64, material: Material, allocator := context.allocator) -> []Triangle {
	// Simple UV sphere with 32×16 segments — good enough for display
	segments_u := 32
	segments_v := 16
	tri_count := segments_u * segments_v * 2
	tris := make([]Triangle, tri_count, allocator)
	idx := 0

	for j in 0 ..< segments_v {
		lat0 := m.PI * f64(j) / f64(segments_v)
		lat1 := m.PI * f64(j + 1) / f64(segments_v)
		for i in 0 ..< segments_u {
			lon0 := 2.0 * m.PI * f64(i) / f64(segments_u)
			lon1 := 2.0 * m.PI * f64(i + 1) / f64(segments_u)

			v0 := center + radius * Vec3{m.sin(lat0) * m.cos(lon0), m.cos(lat0), m.sin(lat0) * m.sin(lon0)}
			v1 := center + radius * Vec3{m.sin(lat1) * m.cos(lon0), m.cos(lat1), m.sin(lat1) * m.sin(lon0)}
			v2 := center + radius * Vec3{m.sin(lat1) * m.cos(lon1), m.cos(lat1), m.sin(lat1) * m.sin(lon1)}
			v3 := center + radius * Vec3{m.sin(lat0) * m.cos(lon1), m.cos(lat0), m.sin(lat0) * m.sin(lon1)}

			n0 := m.normalize(v0 - center)
			n1 := m.normalize(v1 - center)
			n2 := m.normalize(v2 - center)
			n3 := m.normalize(v3 - center)

			tris[idx] = Triangle{v0, v2, v1, n0, n2, n1, {}, {}, {}, false, 0}
			idx += 1
			tris[idx] = Triangle{v0, v3, v2, n0, n3, n2, {}, {}, {}, false, 0}
			idx += 1
		}
	}
	return tris
}

// ── Main GPU render function ────────────────────────────────────────────────

Render_GPU_Buffers :: struct {
	device:             ^MTL.Device,
	cmd_queue:          ^MTL.CommandQueue,
	vertex_buffer:      ^MTL.Buffer,
	index_buffer:       ^MTL.Buffer,
	material_buffer:    ^MTL.Buffer,
	scene_buffer:       ^MTL.Buffer,
	output_buffer:      ^MTL.Buffer,
	accel_struct:       ^MTL.AccelerationStructure,
	pipeline:           ^MTL.ComputePipelineState,
	ift:                ^MTL.IntersectionFunctionTable,
	num_triangles:      i32,
}

render_gpu :: proc(
	scene: ^Scene,
	image_width, image_height: i32,
	samples_per_pixel, max_depth: i32,
	max_radiance: f64,
	file_output: cstring,
	debug_mode: i32 = 0,
	roughness_cutoff: f64 = 0.95,
	glossy_bias: f64 = 0.0,
	gi_cache_enabled: b32 = true,
	gi_cache_distance: f32 = 0.0,
	gi_cache_normal_angle: f32 = 0.5,
	photon_enabled: b32 = true,
	photon_count: i32 = 200000,
	photon_radius: f32 = 0.0,
	photon_bounces: i32 = 8,
	enable_aovs: b32 = false,
	exr_compress: b32 = false,
	denoise_enabled: b32 = false,
	denoise_iterations: i32 = 5,
	denoise_c_sigma: f32 = 0.5,
	denoise_n_sigma: f32 = 0.1,
	denoise_d_sigma: f32 = 0.5,
) {
	total_start := time.tick_now()
	device := MTL.CreateSystemDefaultDevice()
	assert(device != nil, "Metal device required")
	assert(bool(device->supportsRaytracing()), "Raytracing required")
	fmt.println("Device:", device->name()->odinString())

	cmd_queue := device->newCommandQueue()

	// Flatten scene graph to world-space geometry
	flattened := flatten_scene_graph(scene)
	defer destroy_flattened_scene(flattened)

	all_triangles := make([dynamic]Triangle)
	materials := make([dynamic]Material)
	defer delete(all_triangles)
	defer delete(materials)

	// Add flattened triangles
	for tri in flattened.triangles {
		append(&all_triangles, tri)
	}

	// Add flattened materials
	for mat in flattened.materials {
		append(&materials, mat)
	}

	// Process spheres — convert to icosphere meshes (local space, appended after scene graph)
	for sphere in scene.spheres {
		sphere_tris := build_icosphere(sphere.center, sphere.radius, sphere.material)
		tri_mat_idx := i32(len(materials))
		for i in 0 ..< len(sphere_tris) {
			sphere_tris[i].mat_idx = tri_mat_idx
			append(&all_triangles, sphere_tris[i])
		}
		delete(sphere_tris)
		append(&materials, sphere.material)
	}

	num_tris := i32(len(all_triangles))
	fmt.println("Triangles:", num_tris)

	if num_tris == 0 {
		fmt.eprintln("No geometry to render")
		return
	}

	bounds_min := Vec3{1.0e30, 1.0e30, 1.0e30}
	bounds_max := Vec3{-1.0e30, -1.0e30, -1.0e30}
	for tri in all_triangles {
		bounds_min = m.min(bounds_min, tri.v0)
		bounds_min = m.min(bounds_min, tri.v1)
		bounds_min = m.min(bounds_min, tri.v2)
		bounds_max = m.max(bounds_max, tri.v0)
		bounds_max = m.max(bounds_max, tri.v1)
		bounds_max = m.max(bounds_max, tri.v2)
	}
	scene_size := bounds_max - bounds_min
	scene_extent := m.max(m.max(scene_size.x, scene_size.y), scene_size.z)
	auto_radius := f32(auto_gather_radius(all_triangles[:], f64(scene_extent), int(photon_count)))
	effective_gi_cache_distance := gi_cache_distance
	effective_photon_radius := photon_radius
	if effective_gi_cache_distance <= 0.0 {
		effective_gi_cache_distance = auto_radius
		fmt.println("Auto GI cache distance:", effective_gi_cache_distance, "(photon-density based; scene extent:", scene_extent, ")")
	}
	if effective_photon_radius <= 0.0 {
		effective_photon_radius = auto_radius
		fmt.println("Auto photon radius:", effective_photon_radius, "(photon-density based; scene extent:", scene_extent, ")")
	}

	// Build indexed material array (one per unique material)
	GPUTriVertex :: struct {
		pos:    [4]f32,
		normal: [4]f32,
		uv:     [4]f32, // x, y, has_uv (0/1), unused
	}
	vertices := make([]GPUTriVertex, num_tris * 3)
	indices := make([]u32, num_tris * 3)
	mat_indices := make([]i32, num_tris)
	defer delete(vertices)
	defer delete(indices)
	defer delete(mat_indices)

	// Build a single combined texture buffer from every per-material map.
	// Each map's pixels are appended in order and its start offset plus
	// dimensions recorded in the matching `*_info` field. Texture data is
	// RGBA8; the GPU decodes to floats during sampling. The color space is
	// baked into which shader path reads the map, not into the bytes.
	tex_pixels := make([dynamic]u8)
	defer delete(tex_pixels)
	total_tex_bytes: i32 = 0
	for mat in materials {
		for tex in ([]TextureMap{mat.albedo_tex, mat.metallic_roughness_tex, mat.normal_tex, mat.emissive_tex}) {
			if tex.has_data {
				total_tex_bytes += i32(len(tex.pixels))
			}
		}
	}
	if total_tex_bytes > 0 {
		reserve(&tex_pixels, total_tex_bytes)
	}

	// Append `tex` to the shared pixel buffer and return its descriptor:
	// {pixel_offset, width, height, has_tex}.
	pack_texture :: proc(buf: ^[dynamic]u8, tex: TextureMap) -> [4]f32 {
		if !tex.has_data || len(tex.pixels) == 0 {
			return {0, 0, 0, 0}
		}
		offset := i32(len(buf)) / 4 // pixel index, not byte
		append(buf, ..tex.pixels)
		return {f32(offset), f32(tex.width), f32(tex.height), 1.0}
	}

	gpu_materials := make([]GPUMaterial, len(materials))
	defer delete(gpu_materials)
	for mat, i in materials {
		kind_val := i32(0)
		switch mat.kind {
		case .Lambertian: kind_val = 0
		case .Metal: kind_val = 1
		case .Dielectric: kind_val = 2
		case .Principled: kind_val = 3
		case .Emissive: kind_val = 4
		}
		gpu_materials[i] = GPUMaterial {
			albedo   = {f32(mat.albedo.x), f32(mat.albedo.y), f32(mat.albedo.z), 0},
			emission = {f32(mat.emission.x), f32(mat.emission.y), f32(mat.emission.z), 0},
			params0  = {f32(kind_val), f32(mat.fuzz), f32(mat.ir), f32(mat.roughness)},
			params1  = {f32(mat.metallic), f32(mat.emission_strength), f32(mat.specular), f32(mat.clearcoat)},
			params2  = {f32(mat.clearcoat_roughness), f32(mat.sheen), f32(mat.normal_scale), 0},
			spec_tint = {f32(mat.specular_tint.x), f32(mat.specular_tint.y), f32(mat.specular_tint.z), 0},
			sheen_tint = {f32(mat.sheen_tint.x), f32(mat.sheen_tint.y), f32(mat.sheen_tint.z), 0},
			tex_info  = pack_texture(&tex_pixels, mat.albedo_tex),
			mr_info   = pack_texture(&tex_pixels, mat.metallic_roughness_tex),
			nrm_info  = pack_texture(&tex_pixels, mat.normal_tex),
			emis_info = pack_texture(&tex_pixels, mat.emissive_tex),
		}
	}

	// Build vertex buffers — one contiguous triangle soup
	for i in 0 ..< num_tris {
		tri := all_triangles[i]
		base := i * 3
		face_n := m.normalize(m.cross(tri.v1 - tri.v0, tri.v2 - tri.v0))
		n0 := tri.n0 if m.length(tri.n0) > 0 else face_n
		n1 := tri.n1 if m.length(tri.n1) > 0 else face_n
		n2 := tri.n2 if m.length(tri.n2) > 0 else face_n

		// Use uv0/uv1/uv2 from the OBJ/glTF parser; mark `has_uv` as 1 when
		// the triangle carries UVs and the assigned material has at least
		// one map. The shader uses this flag to decide between solid
		// material parameters and texture sampling.
		midx := tri.mat_idx
		if midx < 0 || i32(midx) >= i32(len(materials)) {
			midx = 0
		}
		has_uv_a: f32 = 0.0
		has_uv_b: f32 = 0.0
		has_uv_c: f32 = 0.0
		if tri.has_uv && len(materials) > 0 && material_needs_uv(materials[midx]) {
			has_uv_a = 1.0
			has_uv_b = 1.0
			has_uv_c = 1.0
		}

		vertices[base + 0] = GPUTriVertex{
			pos    = {f32(tri.v0.x), f32(tri.v0.y), f32(tri.v0.z), 0},
			normal = {f32(n0.x), f32(n0.y), f32(n0.z), 0},
			uv     = {f32(tri.uv0.x), f32(tri.uv0.y), has_uv_a, 0},
		}
		vertices[base + 1] = GPUTriVertex{
			pos    = {f32(tri.v1.x), f32(tri.v1.y), f32(tri.v1.z), 0},
			normal = {f32(n1.x), f32(n1.y), f32(n1.z), 0},
			uv     = {f32(tri.uv1.x), f32(tri.uv1.y), has_uv_b, 0},
		}
		vertices[base + 2] = GPUTriVertex{
			pos    = {f32(tri.v2.x), f32(tri.v2.y), f32(tri.v2.z), 0},
			normal = {f32(n2.x), f32(n2.y), f32(n2.z), 0},
			uv     = {f32(tri.uv2.x), f32(tri.uv2.y), has_uv_c, 0},
		}
		indices[base + 0] = u32(base)
		indices[base + 1] = u32(base + 1)
		indices[base + 2] = u32(base + 2)

		mat_indices[i] = midx
	}

	// Build explicit emissive triangle data for direct light sampling.
	gpu_lights := make([dynamic]GPULightTriangle)
	defer delete(gpu_lights)
	for i in 0 ..< num_tris {
		midx := mat_indices[i]
		if midx < 0 || i32(midx) >= i32(len(gpu_materials)) {
			continue
		}
		mat := gpu_materials[midx]
		if i32(mat.params0[0]) != 4 {
			continue
		}
		tri := all_triangles[i]
		emission := mat.emission
		if emission[0] <= 0 && emission[1] <= 0 && emission[2] <= 0 {
			emission = mat.albedo
		}
		strength := mat.params1[1]
		if strength <= 0 {
			strength = 20.0
		}
		append(&gpu_lights, GPULightTriangle {
			p0       = {f32(tri.v0.x), f32(tri.v0.y), f32(tri.v0.z), 0},
			p1       = {f32(tri.v1.x), f32(tri.v1.y), f32(tri.v1.z), 0},
			p2       = {f32(tri.v2.x), f32(tri.v2.y), f32(tri.v2.z), 0},
			emission = {emission[0] * strength, emission[1] * strength, emission[2] * strength, 0},
		})
	}
	// Build explicit quad and sphere lights from scene lights
	gpu_quad_lights := make([dynamic]GPUQuadLight)
	gpu_sphere_lights := make([dynamic]GPUSphereLight)
	defer delete(gpu_quad_lights)
	defer delete(gpu_sphere_lights)

	for l in flattened.lights {
		switch l.kind {
		case .Quad:
			intensity := l.intensity
			append(&gpu_quad_lights, GPUQuadLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				u        = {f32(l.u.x), f32(l.u.y), f32(l.u.z), 0},
				v        = {f32(l.v.x), f32(l.v.y), f32(l.v.z), 0},
				emission = {f32(intensity.x), f32(intensity.y), f32(intensity.z), 0},
			})
		case .Sphere:
			intensity := l.intensity
			append(&gpu_sphere_lights, GPUSphereLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				emission = {f32(intensity.x), f32(intensity.y), f32(intensity.z), 0},
				radius   = f32(l.radius),
			})
		case .Mesh:
			continue
		}
	}

	fmt.println("Light triangles:", len(gpu_lights))
	fmt.println("Quad lights:", len(gpu_quad_lights))
	fmt.println("Sphere lights:", len(gpu_sphere_lights))

	// Irradiance cache buffer + hash grid
	GI_CACHE_MAX_POINTS :: 262144
	GI_GRID_SIZE     :: 32768
	GI_MAX_PER_CELL  :: 16
	gi_cache := make([]GICachePoint, GI_CACHE_MAX_POINTS)
	defer delete(gi_cache)
	gi_counter: i32 = 0

	gi_grid_cells := make([]i32, GI_GRID_SIZE * GI_MAX_PER_CELL)
	defer delete(gi_grid_cells)
	gi_grid_counts := make([]i32, GI_GRID_SIZE)
	defer delete(gi_grid_counts)

	// Scene data
	cam := scene.camera
	scene_data := GPUSceneData {
		origin            = {f32(cam.origin.x), f32(cam.origin.y), f32(cam.origin.z), 0},
		lower_left        = {f32(cam.lower_left_corner.x), f32(cam.lower_left_corner.y), f32(cam.lower_left_corner.z), 0},
		horizontal        = {f32(cam.horizontal.x), f32(cam.horizontal.y), f32(cam.horizontal.z), 0},
		vertical          = {f32(cam.vertical.x), f32(cam.vertical.y), f32(cam.vertical.z), 0},
		u                 = {f32(cam.u.x), f32(cam.u.y), f32(cam.u.z), 0},
		v                 = {f32(cam.v.x), f32(cam.v.y), f32(cam.v.z), 0},
		lens_radius       = f32(cam.lens_radius),
		image_width       = image_width,
		image_height      = image_height,
		samples_per_pixel = samples_per_pixel,
		max_depth         = max_depth,
		max_radiance      = f32(max_radiance),
		debug_mode        = debug_mode,
		tri_light_count   = i32(len(gpu_lights)),
		primitive_count   = num_tris,
		seed              = 42,
		quad_light_count  = i32(len(gpu_quad_lights)),
		sphere_light_count = i32(len(gpu_sphere_lights)),
		roughness_cutoff   = f32(roughness_cutoff),
		glossy_bias        = f32(glossy_bias),
		gi_cache_enabled   = i32(gi_cache_enabled),
		gi_cache_distance  = effective_gi_cache_distance,
		gi_cache_normal_angle = gi_cache_normal_angle,
		photon_enabled     = i32(photon_enabled),
		photon_count       = photon_count,
		photon_radius      = effective_photon_radius,
		photon_max_bounces = photon_bounces,
		gi_cache_num_points = GI_CACHE_MAX_POINTS,
	}
	fmt.println("scene_data.tri_light_count:", scene_data.tri_light_count)
	fmt.println("Material buffer size:", len(gpu_materials) * size_of(GPUMaterial), "bytes, count:", len(gpu_materials))

	// Create Metal buffers
	vertex_buffer := device->newBufferWithSlice(vertices[:], MTL.ResourceStorageModeShared)
	index_buffer := device->newBufferWithSlice(indices[:], MTL.ResourceStorageModeShared)
	material_buffer := device->newBufferWithSlice(gpu_materials[:], MTL.ResourceStorageModeShared)
	mat_index_buffer := device->newBufferWithSlice(mat_indices[:], MTL.ResourceStorageModeShared)
	tri_light_buffer := device->newBufferWithSlice(gpu_lights[:], MTL.ResourceStorageModeShared)
	quad_light_buffer := device->newBufferWithSlice(gpu_quad_lights[:], MTL.ResourceStorageModeShared)
	sphere_light_buffer := device->newBufferWithSlice(gpu_sphere_lights[:], MTL.ResourceStorageModeShared)
	// Texture buffer: RGBA8 pixel data for all material albedo textures.
	// May be empty if no scene material has a map_Kd.
	tex_buffer: ^MTL.Buffer
	if len(tex_pixels) > 0 {
		tex_buffer = device->newBufferWithSlice(tex_pixels[:], MTL.ResourceStorageModeShared)
	} else {
		// Allocate a single dummy byte so the buffer pointer is valid.
		dummy: [1]u8 = {0}
		dummy_slice := dummy[:]
		tex_buffer = device->newBufferWithBytes(dummy_slice, MTL.ResourceStorageModeShared)
	}
	gi_cache_buffer := device->newBufferWithSlice(gi_cache[:], MTL.ResourceStorageModeShared)
	gi_counter_slice := ([^]byte)(&gi_counter)[:size_of(i32)]
	gi_counter_buffer := device->newBufferWithBytes(gi_counter_slice, MTL.ResourceStorageModeShared)
	gi_grid_cells_buffer := device->newBufferWithSlice(gi_grid_cells[:], MTL.ResourceStorageModeShared)
	gi_grid_counts_buffer := device->newBufferWithSlice(gi_grid_counts[:], MTL.ResourceStorageModeShared)

	// Photon mapping buffers. The grid is a counting-sort hash grid:
	//   photon_cell        : hash bucket per photon (count pass output)
	//   photon_grid_counts : photons per bucket
	//   photon_grid_offsets: exclusive prefix sum of counts (CPU-built)
	//   photon_grid_fill   : per-bucket scatter cursor (zeroed before scatter)
	//   photon_grid_sorted : photon indices grouped by bucket
	PHOTON_MAX_COUNT :: 1048576
	PHOTON_GRID_SIZE :: 16384
	photons := make([]Photon, PHOTON_MAX_COUNT)
	defer delete(photons)
	photon_counter: i32 = 0
	photon_cell := make([]i32, PHOTON_MAX_COUNT)
	defer delete(photon_cell)
	photon_grid_counts := make([]i32, PHOTON_GRID_SIZE)
	defer delete(photon_grid_counts)
	photon_grid_offsets := make([]i32, PHOTON_GRID_SIZE)
	defer delete(photon_grid_offsets)
	photon_grid_fill := make([]i32, PHOTON_GRID_SIZE)
	defer delete(photon_grid_fill)
	photon_grid_sorted := make([]i32, PHOTON_MAX_COUNT)
	defer delete(photon_grid_sorted)

	photons_buffer := device->newBufferWithSlice(photons[:], MTL.ResourceStorageModeShared)
	photon_counter_slice := ([^]byte)(&photon_counter)[:size_of(i32)]
	photon_counter_buffer := device->newBufferWithBytes(photon_counter_slice, MTL.ResourceStorageModeShared)
	photon_cell_buffer := device->newBufferWithSlice(photon_cell[:], MTL.ResourceStorageModeShared)
	photon_grid_counts_buffer := device->newBufferWithSlice(photon_grid_counts[:], MTL.ResourceStorageModeShared)
	photon_grid_offsets_buffer := device->newBufferWithSlice(photon_grid_offsets[:], MTL.ResourceStorageModeShared)
	photon_grid_fill_buffer := device->newBufferWithSlice(photon_grid_fill[:], MTL.ResourceStorageModeShared)
	photon_grid_sorted_buffer := device->newBufferWithSlice(photon_grid_sorted[:], MTL.ResourceStorageModeShared)

	scene_slice := ([^]byte)(&scene_data)[:size_of(GPUSceneData)]
	scene_buffer := device->newBufferWithBytes(scene_slice, MTL.ResourceStorageModeShared)

	pixel_count := int(image_width) * int(image_height)
	output_buffer := device->newBufferWithLength(
		NS.UInteger(pixel_count * size_of([4]f32)),
		MTL.ResourceStorageModeShared,
	)

	// Load shader
	msl_source := #load("shaders/raytrace.metal", string)
	src := NS.String.alloc()->initWithOdinString(msl_source)
	opts := MTL.CompileOptions.alloc()->init()
	opts->setFastMathEnabled(true)
	opts->setLanguageVersion(.Version3_0)

	library, err := device->newLibraryWithSource(src, opts)
	if err != nil {
		error_str := err->localizedDescription()->odinString()
		fmt.eprintln("Shader compilation failed:", error_str)
		return
	}

	kernel_func := library->newFunctionWithName(NS.AT("raytraceKernel"))
	assert(kernel_func != nil, "kernel function not found")

	desc := MTL.ComputePipelineDescriptor.alloc()->init()
	desc->setComputeFunction(kernel_func)

	pipeline, p_err := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
		device, desc, MTL.PipelineOption{}, nil,
	)
	if p_err != nil {
		fmt.eprintln("Pipeline creation failed:", p_err->localizedDescription()->odinString())
		return
	}

	// Denoise pipeline (À-Trous edge-avoiding wavelet). Compiled from its own
	// library so the post-process filter stays separate from the tracer.
	denoise_pipeline: ^MTL.ComputePipelineState
	if denoise_enabled {
		dn_source := #load("shaders/denoise.metal", string)
		dn_src := NS.String.alloc()->initWithOdinString(dn_source)
		dn_opts := MTL.CompileOptions.alloc()->init()
		dn_opts->setFastMathEnabled(true)
		dn_opts->setLanguageVersion(.Version3_0)
		dn_library, dn_err := device->newLibraryWithSource(dn_src, dn_opts)
		if dn_err != nil {
			fmt.eprintln("Denoise shader compilation failed:", dn_err->localizedDescription()->odinString())
			return
		}
		dn_func := dn_library->newFunctionWithName(NS.AT("atrousDenoiseKernel"))
		assert(dn_func != nil, "atrousDenoiseKernel not found")
		dn_desc := MTL.ComputePipelineDescriptor.alloc()->init()
		dn_desc->setComputeFunction(dn_func)
		dn_pipe, dn_perr := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
			device, dn_desc, MTL.PipelineOption{}, nil,
		)
		if dn_perr != nil {
			fmt.eprintln("Denoise pipeline creation failed:", dn_perr->localizedDescription()->odinString())
			return
		}
		denoise_pipeline = dn_pipe
	}

	// Photon emission pipeline
	photon_kernel_func := library->newFunctionWithName(NS.AT("photonEmitKernel"))
	assert(photon_kernel_func != nil, "photonEmitKernel not found")
	photon_desc := MTL.ComputePipelineDescriptor.alloc()->init()
	photon_desc->setComputeFunction(photon_kernel_func)
	photon_pipeline, pp_err := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
		device, photon_desc, MTL.PipelineOption{}, nil,
	)
	if pp_err != nil {
		fmt.eprintln("Photon pipeline failed:", pp_err->localizedDescription()->odinString())
		return
	}

	// Photon grid count pipeline
	photon_count_func := library->newFunctionWithName(NS.AT("photonCountKernel"))
	assert(photon_count_func != nil, "photonCountKernel not found")
	photon_count_desc := MTL.ComputePipelineDescriptor.alloc()->init()
	photon_count_desc->setComputeFunction(photon_count_func)
	photon_count_pipeline, pc_err := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
		device, photon_count_desc, MTL.PipelineOption{}, nil,
	)
	if pc_err != nil {
		fmt.eprintln("Photon count pipeline failed:", pc_err->localizedDescription()->odinString())
		return
	}

	// Photon grid scatter pipeline
	photon_scatter_func := library->newFunctionWithName(NS.AT("photonScatterKernel"))
	assert(photon_scatter_func != nil, "photonScatterKernel not found")
	photon_scatter_desc := MTL.ComputePipelineDescriptor.alloc()->init()
	photon_scatter_desc->setComputeFunction(photon_scatter_func)
	photon_scatter_pipeline, ps_err := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
		device, photon_scatter_desc, MTL.PipelineOption{}, nil,
	)
	if ps_err != nil {
		fmt.eprintln("Photon scatter pipeline failed:", ps_err->localizedDescription()->odinString())
		return
	}

	// Build triangle acceleration structure
	fmt.println("Building acceleration structure...")

	tri_geom := MTL.AccelerationStructureTriangleGeometryDescriptor.alloc()->init()
	tri_geom->setVertexBuffer(vertex_buffer)
	tri_geom->setVertexStride(48) // float4 position + float4 normal + float4 uv
	tri_geom->setIndexBuffer(index_buffer)
	tri_geom->setIndexType(.UInt32)
	tri_geom->setTriangleCount(NS.UInteger(num_tris))

	prim_desc := MTL.PrimitiveAccelerationStructureDescriptor.alloc()->init()
	geometries := [?]^NS.Object{auto_cast tri_geom}
	geom_array := NS.Array.alloc()->initWithObjects(raw_data(geometries[:]), 1)
	prim_desc->setGeometryDescriptors(geom_array)

	sizes := device->accelerationStructureSizesWithDescriptor(prim_desc)
	fmt.println("  AS size:", sizes.accelerationStructureSize)

	as := device->newAccelerationStructureWithSize(NS.UInteger(sizes.accelerationStructureSize))
	scratch := device->newBufferWithLength(
		NS.UInteger(sizes.buildScratchBufferSize),
		MTL.ResourceStorageModeShared,
	)

	as_start := time.tick_now()
	cmd_buf := cmd_queue->commandBuffer()
	as_encoder := cmd_buf->accelerationStructureCommandEncoder()
	as_encoder->buildAccelerationStructure(as, prim_desc, scratch, 0)
	as_encoder->endEncoding()
	cmd_buf->commit()
	cmd_buf->waitUntilCompleted()
	fmt.printfln("  Done. [%.3f s]", time.duration_seconds(time.tick_since(as_start)))

	// Dispatch compute
	fmt.println("Rendering...")

	// Photon map build: emit → count → [CPU prefix sum] → scatter.
	// The count and scatter passes are split across two command buffers so
	// the CPU can compute the exclusive prefix sum of the per-bucket counts
	// in between (16384 buckets — trivial on the CPU, and it avoids a GPU
	// scan kernel).
	if scene_data.photon_enabled != 0 && scene_data.photon_count > 0 {
		photon_start := time.tick_now()
		photon_n := min(scene_data.photon_count, PHOTON_MAX_COUNT)
		ph_tg := MTL.Size{width = 64, height = 1, depth = 1}
		ph_gs := MTL.Size{width = NS.Integer(photon_n), height = 1, depth = 1}

		build_buf := cmd_queue->commandBuffer()

		// Emit
		emit_enc := build_buf->computeCommandEncoder()
		emit_enc->setComputePipelineState(photon_pipeline)
		emit_enc->setBuffer(scene_buffer, 0, 0)
		emit_enc->setBuffer(material_buffer, 0, 1)
		emit_enc->setAccelerationStructure(as, 3)
		emit_enc->setBuffer(vertex_buffer, 0, 4)
		emit_enc->setBuffer(index_buffer, 0, 5)
		emit_enc->setBuffer(tri_light_buffer, 0, 6)
		emit_enc->setBuffer(quad_light_buffer, 0, 7)
		emit_enc->setBuffer(sphere_light_buffer, 0, 8)
		emit_enc->setBuffer(mat_index_buffer, 0, 9)
		emit_enc->setBuffer(photons_buffer, 0, 14)
		emit_enc->setBuffer(photon_counter_buffer, 0, 15)
		emit_enc->dispatchThreads(ph_gs, ph_tg)
		emit_enc->endEncoding()

		// Count photons per bucket
		cnt_enc := build_buf->computeCommandEncoder()
		cnt_enc->setComputePipelineState(photon_count_pipeline)
		cnt_enc->setBuffer(photons_buffer, 0, 0)
		cnt_enc->setBuffer(photon_counter_buffer, 0, 1)
		cnt_enc->setBuffer(photon_cell_buffer, 0, 2)
		cnt_enc->setBuffer(photon_grid_counts_buffer, 0, 3)
		cnt_enc->setBuffer(scene_buffer, 0, 4)
		cnt_enc->dispatchThreads(ph_gs, ph_tg)
		cnt_enc->endEncoding()

		build_buf->commit()
		build_buf->waitUntilCompleted()

		// CPU exclusive prefix sum: counts → offsets, zero the fill cursors.
		counts := photon_grid_counts_buffer->contentsAsSlice([]i32)
		offsets := photon_grid_offsets_buffer->contentsAsSlice([]i32)
		fill := photon_grid_fill_buffer->contentsAsSlice([]i32)
		running: i32 = 0
		for i in 0 ..< PHOTON_GRID_SIZE {
			offsets[i] = running
			running += counts[i]
			fill[i] = 0
		}
		fmt.println("  Photons stored:", running)

		// Scatter photon indices into the sorted array.
		scatter_buf := cmd_queue->commandBuffer()
		sc_enc := scatter_buf->computeCommandEncoder()
		sc_enc->setComputePipelineState(photon_scatter_pipeline)
		sc_enc->setBuffer(photon_counter_buffer, 0, 1)
		sc_enc->setBuffer(photon_cell_buffer, 0, 2)
		sc_enc->setBuffer(photon_grid_offsets_buffer, 0, 3)
		sc_enc->setBuffer(photon_grid_fill_buffer, 0, 4)
		sc_enc->setBuffer(photon_grid_sorted_buffer, 0, 5)
		sc_enc->setBuffer(scene_buffer, 0, 6)
		sc_enc->dispatchThreads(ph_gs, ph_tg)
		sc_enc->endEncoding()
		scatter_buf->commit()
		scatter_buf->waitUntilCompleted()
		fmt.printfln("  Photon map build: [%.3f s]", time.duration_seconds(time.tick_since(photon_start)))
	}

	// Main raytrace pass
	trace_start := time.tick_now()
	dispatch_buf := cmd_queue->commandBuffer()
	enc := dispatch_buf->computeCommandEncoder()
	enc->setComputePipelineState(pipeline)
	enc->setBuffer(scene_buffer, 0, 0)
	enc->setBuffer(material_buffer, 0, 1)
	enc->setBuffer(output_buffer, 0, 2)
	enc->setAccelerationStructure(as, 3)
	enc->setBuffer(vertex_buffer, 0, 4)
	enc->setBuffer(index_buffer, 0, 5)
	enc->setBuffer(tri_light_buffer, 0, 6)
	enc->setBuffer(quad_light_buffer, 0, 7)
	enc->setBuffer(sphere_light_buffer, 0, 8)
	enc->setBuffer(mat_index_buffer, 0, 9)
	enc->setBuffer(gi_cache_buffer, 0, 10)
	enc->setBuffer(gi_counter_buffer, 0, 11)
	enc->setBuffer(gi_grid_cells_buffer, 0, 12)
	enc->setBuffer(gi_grid_counts_buffer, 0, 13)
	enc->setBuffer(photons_buffer, 0, 14)
	enc->setBuffer(photon_counter_buffer, 0, 15)
	enc->setBuffer(photon_grid_offsets_buffer, 0, 16)
	enc->setBuffer(photon_grid_counts_buffer, 0, 17)
	enc->setBuffer(tex_buffer, 0, 18)
	enc->setBuffer(photon_grid_sorted_buffer, 0, 19)

	tg_size := MTL.Size{width = 16, height = 8, depth = 1}
	grid_size := MTL.Size{
		width  = NS.Integer(image_width),
		height = NS.Integer(image_height),
		depth  = 1,
	}
	enc->dispatchThreads(grid_size, tg_size)
	enc->endEncoding()

	dispatch_buf->commit()
	dispatch_buf->waitUntilCompleted()
	fmt.printfln("  Done. [%.3f s]", time.duration_seconds(time.tick_since(trace_start)))

	// Snapshot the beauty result before the AOV/guide passes below overwrite
	// output_buffer. This is the pristine beauty image the final readback and
	// the denoiser both consume.
	beauty_snapshot := make([][4]f32, pixel_count)
	defer delete(beauty_snapshot)
	copy(beauty_snapshot, output_buffer->contentsAsSlice([][4]f32)[:pixel_count])

	// AOV passes: re-run the kernel with a different debug_mode to
	// capture each AOV. We re-use the photon and GI cache state from
	// the main pass — the only thing that changes is the kernel's
	// `debug_mode` field in scene_data.
	aov_passes: []int
	if enable_aovs {
		aov_passes = []int{1, 2, 3, 5, 9} // albedo, normal, depth, direct, indirect
	} else {
		aov_passes = []int{}
	}
	aov_results := make(map[int][dynamic][4]f32)
	defer {
		for _, v in aov_results {
			delete(v)
		}
		delete(aov_results)
	}
	aov_start := time.tick_now()
	if enable_aovs {
		fmt.println("AOV passes:", len(aov_passes))
		// The GI cache and photon map built during the beauty pass persist on
		// the GPU and are reused for every AOV pass — the indirect/beauty AOVs
		// see the same biased-GI state the beauty image did.

		for aov_debug in aov_passes {
			// Update scene_data.debug_mode in the GPU buffer
			scene_data.debug_mode = i32(aov_debug)
			scene_slice := ([^]byte)(&scene_data)[:size_of(GPUSceneData)]
			dst := ([^]byte)(raw_data(scene_buffer->contents()))[:size_of(GPUSceneData)]
			copy(dst, scene_slice)

			aov_cmd := cmd_queue->commandBuffer()
			aov_enc := aov_cmd->computeCommandEncoder()
			aov_enc->setComputePipelineState(pipeline)
			aov_enc->setBuffer(scene_buffer, 0, 0)
			aov_enc->setBuffer(material_buffer, 0, 1)
			aov_enc->setBuffer(output_buffer, 0, 2)
			aov_enc->setAccelerationStructure(as, 3)
			aov_enc->setBuffer(vertex_buffer, 0, 4)
			aov_enc->setBuffer(index_buffer, 0, 5)
			aov_enc->setBuffer(tri_light_buffer, 0, 6)
			aov_enc->setBuffer(quad_light_buffer, 0, 7)
			aov_enc->setBuffer(sphere_light_buffer, 0, 8)
			aov_enc->setBuffer(mat_index_buffer, 0, 9)
			aov_enc->setBuffer(gi_cache_buffer, 0, 10)
			aov_enc->setBuffer(gi_counter_buffer, 0, 11)
			aov_enc->setBuffer(gi_grid_cells_buffer, 0, 12)
			aov_enc->setBuffer(gi_grid_counts_buffer, 0, 13)
			aov_enc->setBuffer(photons_buffer, 0, 14)
			aov_enc->setBuffer(photon_counter_buffer, 0, 15)
			aov_enc->setBuffer(photon_grid_offsets_buffer, 0, 16)
			aov_enc->setBuffer(photon_grid_counts_buffer, 0, 17)
			aov_enc->setBuffer(tex_buffer, 0, 18)
			aov_enc->setBuffer(photon_grid_sorted_buffer, 0, 19)
			aov_enc->dispatchThreads(grid_size, tg_size)
			aov_enc->endEncoding()
			aov_cmd->commit()
			aov_cmd->waitUntilCompleted()

			// Read back the result for this AOV
			aov_data := output_buffer->contentsAsSlice([][4]f32)
			aov_results[aov_debug] = make([dynamic][4]f32, pixel_count)
			copy(aov_results[aov_debug][:], aov_data[:pixel_count])
		}
		// Reset debug_mode for the beauty readback
		scene_data.debug_mode = debug_mode
		fmt.printfln("  AOV passes: [%.3f s]", time.duration_seconds(time.tick_since(aov_start)))
	}

	// ── Stage 5: edge-avoiding À-Trous denoise ──────────────────────────
	// Filters the beauty snapshot in place, guided by first-hit normal/depth
	// passes. The result is written back into output_buffer so both the PNG
	// and EXR beauty consume the denoised image.
	denoise_start := time.tick_now()
	if denoise_enabled {
		fmt.println("Denoising (A-Trous,", denoise_iterations, "iterations)...")

		buf_len := NS.UInteger(pixel_count * size_of([4]f32))
		normal_guide := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		depth_guide  := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		albedo_guide := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		emission_guide := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		color_a      := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		color_b      := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)

		// Render one raw geometry guide (first-hit-only debug mode) into
		// output_buffer, then copy it into `dst`.
		render_guide :: proc(
			cmd_queue: ^MTL.CommandQueue, pipeline: ^MTL.ComputePipelineState,
			scene_data: ^GPUSceneData, scene_buffer, material_buffer, output_buffer: ^MTL.Buffer,
			as: ^MTL.AccelerationStructure,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer,
			gi_grid_counts_buffer, photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer,
			photon_grid_counts_buffer, tex_buffer, photon_grid_sorted_buffer: ^MTL.Buffer,
			grid_size, tg_size: MTL.Size, mode: i32, dst: ^MTL.Buffer, pixel_count: int,
		) {
			scene_data.debug_mode = mode
			// Guides are deterministic geometry passes: one sample per pixel
			// keeps them crisp (averaging specular-followed paths over many
			// samples would blur the guide) and makes them ~spp× cheaper.
			saved_spp := scene_data.samples_per_pixel
			scene_data.samples_per_pixel = 1
			scene_slice := ([^]byte)(scene_data)[:size_of(GPUSceneData)]
			copy(([^]byte)(raw_data(scene_buffer->contents()))[:size_of(GPUSceneData)], scene_slice)
			scene_data.samples_per_pixel = saved_spp

			cmd := cmd_queue->commandBuffer()
			e := cmd->computeCommandEncoder()
			e->setComputePipelineState(pipeline)
			e->setBuffer(scene_buffer, 0, 0)
			e->setBuffer(material_buffer, 0, 1)
			e->setBuffer(output_buffer, 0, 2)
			e->setAccelerationStructure(as, 3)
			e->setBuffer(vertex_buffer, 0, 4)
			e->setBuffer(index_buffer, 0, 5)
			e->setBuffer(tri_light_buffer, 0, 6)
			e->setBuffer(quad_light_buffer, 0, 7)
			e->setBuffer(sphere_light_buffer, 0, 8)
			e->setBuffer(mat_index_buffer, 0, 9)
			e->setBuffer(gi_cache_buffer, 0, 10)
			e->setBuffer(gi_counter_buffer, 0, 11)
			e->setBuffer(gi_grid_cells_buffer, 0, 12)
			e->setBuffer(gi_grid_counts_buffer, 0, 13)
			e->setBuffer(photons_buffer, 0, 14)
			e->setBuffer(photon_counter_buffer, 0, 15)
			e->setBuffer(photon_grid_offsets_buffer, 0, 16)
			e->setBuffer(photon_grid_counts_buffer, 0, 17)
			e->setBuffer(tex_buffer, 0, 18)
			e->setBuffer(photon_grid_sorted_buffer, 0, 19)
			e->dispatchThreads(grid_size, tg_size)
			e->endEncoding()
			cmd->commit()
			cmd->waitUntilCompleted()
			copy(dst->contentsAsSlice([][4]f32)[:pixel_count], output_buffer->contentsAsSlice([][4]f32)[:pixel_count])
		}

		render_guide(cmd_queue, pipeline, &scene_data, scene_buffer, material_buffer, output_buffer, as,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer, gi_grid_counts_buffer,
			photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer, photon_grid_counts_buffer,
			tex_buffer, photon_grid_sorted_buffer, grid_size, tg_size, 20, normal_guide, pixel_count)
		render_guide(cmd_queue, pipeline, &scene_data, scene_buffer, material_buffer, output_buffer, as,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer, gi_grid_counts_buffer,
			photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer, photon_grid_counts_buffer,
			tex_buffer, photon_grid_sorted_buffer, grid_size, tg_size, 21, depth_guide, pixel_count)
		render_guide(cmd_queue, pipeline, &scene_data, scene_buffer, material_buffer, output_buffer, as,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer, gi_grid_counts_buffer,
			photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer, photon_grid_counts_buffer,
			tex_buffer, photon_grid_sorted_buffer, grid_size, tg_size, 22, albedo_guide, pixel_count)
		render_guide(cmd_queue, pipeline, &scene_data, scene_buffer, material_buffer, output_buffer, as,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer, gi_grid_counts_buffer,
			photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer, photon_grid_counts_buffer,
			tex_buffer, photon_grid_sorted_buffer, grid_size, tg_size, 23, emission_guide, pixel_count)

		// Albedo demodulation: filter illumination, not texture detail. Divide
		// the beauty by a clamped first-hit albedo, filter that illumination
		// signal, then remultiply by the same clamped albedo. Using the *same*
		// clamped value for divide and multiply makes the transform an exact
		// inverse where albedo is near zero (background, emitters), so those
		// pixels pass through unharmed.
		// Albedo floor. Dividing by a near-zero albedo channel amplifies noise
		// into self-preserving outliers the wavelet can't remove. Texture
		// detail lives in the *bright* albedo channels (which stay above the
		// floor and demodulate fully); dark channels carry no visible detail,
		// so flooring them well above zero caps amplification with no loss.
		DEMOD_EPS :: f32(0.15)
		clamped_albedo := make([][4]f32, pixel_count)
		defer delete(clamped_albedo)
		albedo_src := albedo_guide->contentsAsSlice([][4]f32)
		emission_src := emission_guide->contentsAsSlice([][4]f32)
		color_a_dst := color_a->contentsAsSlice([][4]f32)
		for i in 0 ..< pixel_count {
			ar := max(albedo_src[i][0], DEMOD_EPS)
			ag := max(albedo_src[i][1], DEMOD_EPS)
			ab := max(albedo_src[i][2], DEMOD_EPS)
			clamped_albedo[i] = {ar, ag, ab, 1.0}
			// Remove the noise-free self-emission (HUD/screen glow, emissive
			// textures) before demodulating. Otherwise it lands in the
			// illumination signal and the wavelet filter smears it across the
			// surface. It is added back, untouched, after filtering.
			color_a_dst[i] = {
				(beauty_snapshot[i][0] - emission_src[i][0]) / ar,
				(beauty_snapshot[i][1] - emission_src[i][1]) / ag,
				(beauty_snapshot[i][2] - emission_src[i][2]) / ab,
				beauty_snapshot[i][3],
			}
		}

		dn_params := GPUDenoiseParams{
			width        = image_width,
			height       = image_height,
			step_width   = 1,
			sigma_color  = denoise_c_sigma,
			sigma_normal = denoise_n_sigma,
			sigma_depth  = denoise_d_sigma,
		}
		dn_params_buffer := device->newBufferWithLength(NS.UInteger(size_of(GPUDenoiseParams)), MTL.ResourceStorageModeShared)

		iters := int(denoise_iterations)
		if iters < 1 { iters = 1 }
		src, dst := color_a, color_b
		for it in 0 ..< iters {
			dn_params.step_width = i32(1 << uint(it))
			copy(([^]byte)(raw_data(dn_params_buffer->contents()))[:size_of(GPUDenoiseParams)],
				([^]byte)(&dn_params)[:size_of(GPUDenoiseParams)])

			dn_cmd := cmd_queue->commandBuffer()
			dn_enc := dn_cmd->computeCommandEncoder()
			dn_enc->setComputePipelineState(denoise_pipeline)
			dn_enc->setBuffer(dn_params_buffer, 0, 0)
			dn_enc->setBuffer(src, 0, 1)
			dn_enc->setBuffer(dst, 0, 2)
			dn_enc->setBuffer(normal_guide, 0, 3)
			dn_enc->setBuffer(depth_guide, 0, 4)
			dn_enc->dispatchThreads(grid_size, tg_size)
			dn_enc->endEncoding()
			dn_cmd->commit()
			dn_cmd->waitUntilCompleted()

			src, dst = dst, src
		}

		// Firefly pre-clamp on the demodulated illumination. Dividing a dark
		// beauty spike by a small albedo channel yields an extreme outlier,
		// and À-Trous self-preserves outliers (every neighbour reads as an
		// edge next to the spike), so they survive filtering. Clamp each
		// pixel's luminance to a multiple of its 3×3 neighbourhood mean before
		// the wavelet passes.
		{
			luminance :: proc(c: [4]f32) -> f32 {
				return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
			}
			FIREFLY_K :: f32(3.0)
			w := int(image_width)
			h := int(image_height)
			illum := color_a->contentsAsSlice([][4]f32)
			clamped := make([][4]f32, pixel_count)
			defer delete(clamped)
			copy(clamped, illum[:pixel_count])
			for y in 0 ..< h {
				for x in 0 ..< w {
					idx := y * w + x
					sum: f32 = 0
					n: f32 = 0
					for dy in -1 ..= 1 {
						for dx in -1 ..= 1 {
							sx := x + dx
							sy := y + dy
							if sx < 0 || sx >= w || sy < 0 || sy >= h {
								continue
							}
							sum += luminance(illum[sy * w + sx])
							n += 1
						}
					}
					mean := sum / n
					lum := luminance(illum[idx])
					limit := mean * FIREFLY_K
					if lum > limit && lum > 0 {
						scale := limit / lum
						clamped[idx] = {
							illum[idx][0] * scale,
							illum[idx][1] * scale,
							illum[idx][2] * scale,
							illum[idx][3],
						}
					}
				}
			}
			copy(illum[:pixel_count], clamped)
		}

		// `src` holds the final filtered illumination. Remodulate by the
		// clamped albedo to restore texture detail, and push it back into the
		// beauty snapshot and output_buffer.
		filtered := src->contentsAsSlice([][4]f32)
		for i in 0 ..< pixel_count {
			beauty_snapshot[i] = {
				filtered[i][0] * clamped_albedo[i][0] + emission_src[i][0],
				filtered[i][1] * clamped_albedo[i][1] + emission_src[i][1],
				filtered[i][2] * clamped_albedo[i][2] + emission_src[i][2],
				filtered[i][3],
			}
		}
		fmt.printfln("  Done. [%.3f s]", time.duration_seconds(time.tick_since(denoise_start)))
	}

	// Restore the beauty image (denoised or raw) into output_buffer — the AOV
	// passes above leave the last AOV in it.
	copy(output_buffer->contentsAsSlice([][4]f32)[:pixel_count], beauty_snapshot)

	// Readback + write
	fmt.printfln("Total render time: %.3f s", time.duration_seconds(time.tick_since(total_start)))
	fmt.println("Writing", file_output)

	output_data := output_buffer->contentsAsSlice([][4]f32)
	pixels := make([]u8, pixel_count * 3)
	defer delete(pixels)

	// The kernel writes linear radiance. Encode with the sRGB OETF for the
	// 8-bit PNG buffer; the EXR branch below reads `output_data` directly and
	// so stays linear. Clamp in linear space first -- the OETF is undefined
	// for negatives.
	for i in 0 ..< pixel_count {
		lr := linear_to_srgb(clamp(f64(output_data[i][0]), 0.0, 1.0))
		lg := linear_to_srgb(clamp(f64(output_data[i][1]), 0.0, 1.0))
		lb := linear_to_srgb(clamp(f64(output_data[i][2]), 0.0, 1.0))
		pixels[i * 3 + 0] = u8(clamp(lr * 255.0, 0.0, 255.0))
		pixels[i * 3 + 1] = u8(clamp(lg * 255.0, 0.0, 255.0))
		pixels[i * 3 + 2] = u8(clamp(lb * 255.0, 0.0, 255.0))
	}

	stbi.flip_vertically_on_write(true)
	path_str := string(file_output)
	if strings.has_suffix(path_str, ".exr") {
		// Build the EXR image. Beauty is always the first layer.
		// When AOVs are enabled, additional layers (albedo, normal,
		// depth, direct, indirect) are appended.
		img: output.EXR_Image
		output.exr_image_init(&img, image_width, image_height)
		img.compression = output.EXR_COMPRESSION_ZIP if exr_compress else output.EXR_COMPRESSION_NONE
		defer output.exr_destroy(&img)
		beauty_pixels := make([][4]f32, pixel_count)
		defer delete(beauty_pixels)
		for i in 0 ..< pixel_count {
			beauty_pixels[i] = [4]f32 {
				output_data[i][0],
				output_data[i][1],
				output_data[i][2],
				output_data[i][3],
			}
		}
		rgba_chans := []output.EXR_Channel{
			{name = "R", component = 0, pixel_type = 1, x_sampling = 1, y_sampling = 1},
			{name = "G", component = 1, pixel_type = 1, x_sampling = 1, y_sampling = 1},
			{name = "B", component = 2, pixel_type = 1, x_sampling = 1, y_sampling = 1},
			{name = "A", component = 3, pixel_type = 1, x_sampling = 1, y_sampling = 1},
		}
		output.exr_add_layer(&img, "", rgba_chans[:], beauty_pixels)

		if enable_aovs {
			// Iterate over the AOV debug modes that were actually rendered
			aov_passes_seen := []int{1, 2, 3, 5, 9}
			aov_layer_names := make(map[int]string)
			defer delete(aov_layer_names)
			aov_layer_names[1] = "albedo"
			aov_layer_names[2] = "normal"
			aov_layer_names[3] = "depth"
			aov_layer_names[5] = "direct"
			aov_layer_names[9] = "indirect"
			for debug in aov_passes_seen {
				_, ok := aov_results[debug]
				if !ok {
					continue
				}
				aov_chans := []output.EXR_Channel{
					{name = "R", component = 0, pixel_type = 1, x_sampling = 1, y_sampling = 1},
					{name = "G", component = 1, pixel_type = 1, x_sampling = 1, y_sampling = 1},
					{name = "B", component = 2, pixel_type = 1, x_sampling = 1, y_sampling = 1},
					{name = "A", component = 3, pixel_type = 1, x_sampling = 1, y_sampling = 1},
				}
				output.exr_add_layer(&img, aov_layer_names[debug], aov_chans[:], aov_results[debug][:])
			}
		}

		if !output.exr_write_file(&img, path_str) {
			fmt.eprintln("Failed to write EXR")
			return
		}
		fmt.println("Wrote", file_output, "(EXR,", len(img.layers), "layers)")
		return
	}
	ok := stbi.write_png(
		file_output,
		c.int(image_width),
		c.int(image_height),
		3,
		raw_data(pixels),
		c.int(image_width * 3),
	)
	if ok == 0 {
		fmt.eprintln("Failed to write PNG")
		return
	}
	fmt.println("Wrote", file_output)
}
