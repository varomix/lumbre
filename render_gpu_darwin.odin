package main

import "core:c"
import "core:fmt"
import "core:strings"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"
import stbi "vendor:stb/image"
import m "core:math/linalg/glsl"
import "output"

// ── GPU data structs (packed for Metal) ──────────────────────────────────────

GPUMaterial :: struct {
	albedo:    [4]f32,
	emission:  [4]f32,
	params0:   [4]f32, // kind, fuzz, ir, roughness
	params1:   [4]f32, // metallic, emission_strength, specular, clearcoat
	params2:   [4]f32, // clearcoat_roughness, sheen, unused, unused
	spec_tint: [4]f32, // rgb = specular_tint
	sheen_tint: [4]f32, // rgb = sheen_tint
	tex_info:  [4]f32, // x=tex_offset, y=tex_width, z=tex_height, w=has_tex
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
	photon_count: i32 = 1048576,
	photon_radius: f32 = 0.0,
	photon_bounces: i32 = 8,
	enable_aovs: b32 = false,
) {
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
	auto_radius := f32(max(scene_extent / 6.0, 0.05))
	effective_gi_cache_distance := gi_cache_distance
	effective_photon_radius := photon_radius
	if effective_gi_cache_distance <= 0.0 {
		effective_gi_cache_distance = auto_radius
		fmt.println("Auto GI cache distance:", effective_gi_cache_distance, "(scene extent:", scene_extent, ")")
	}
	if effective_photon_radius <= 0.0 {
		effective_photon_radius = auto_radius
		fmt.println("Auto photon radius:", effective_photon_radius, "(scene extent:", scene_extent, ")")
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

	// Build a single combined texture buffer from per-material albedo
	// textures. We pack each material's pixels in order and record the
	// starting offset and dimensions in `tex_info`. Texture data is
	// RGBA8; the GPU decodes to floats during sampling.
	tex_pixels := make([dynamic]u8)
	defer delete(tex_pixels)
	total_tex_bytes: i32 = 0
	for mat in materials {
		if mat.albedo_tex.has_data && len(mat.albedo_tex.pixels) > 0 {
			total_tex_bytes += i32(len(mat.albedo_tex.pixels))
		}
	}
	if total_tex_bytes > 0 {
		reserve(&tex_pixels, total_tex_bytes)
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
		// Pack this material's texture into the global buffer (if any).
		tex_offset: i32 = 0
		tex_w: i32 = 0
		tex_h: i32 = 0
		has_tex: f32 = 0.0
		if mat.albedo_tex.has_data && len(mat.albedo_tex.pixels) > 0 {
			tex_offset = i32(len(tex_pixels)) / 4  // pixel index, not byte
			tex_w = mat.albedo_tex.width
			tex_h = mat.albedo_tex.height
			has_tex = 1.0
			append(&tex_pixels, ..mat.albedo_tex.pixels)
		}
		gpu_materials[i] = GPUMaterial {
			albedo   = {f32(mat.albedo.x), f32(mat.albedo.y), f32(mat.albedo.z), 0},
			emission = {f32(mat.emission.x), f32(mat.emission.y), f32(mat.emission.z), 0},
			params0  = {f32(kind_val), f32(mat.fuzz), f32(mat.ir), f32(mat.roughness)},
			params1  = {f32(mat.metallic), f32(mat.emission_strength), f32(mat.specular), f32(mat.clearcoat)},
			params2  = {f32(mat.clearcoat_roughness), f32(mat.sheen), 0, 0},
			spec_tint = {f32(mat.specular_tint.x), f32(mat.specular_tint.y), f32(mat.specular_tint.z), 0},
			sheen_tint = {f32(mat.sheen_tint.x), f32(mat.sheen_tint.y), f32(mat.sheen_tint.z), 0},
			tex_info = {f32(tex_offset), f32(tex_w), f32(tex_h), has_tex},
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

		// Use uv0/uv1/uv2 from the OBJ parser; mark `has_uv` as 1 when
		// the triangle carries UVs and the assigned material has a
		// texture. The shader uses this flag to decide between solid
		// albedo and texture sampling.
		midx := tri.mat_idx
		if midx < 0 || i32(midx) >= i32(len(materials)) {
			midx = 0
		}
		has_uv_a: f32 = 0.0
		has_uv_b: f32 = 0.0
		has_uv_c: f32 = 0.0
		if tri.has_uv && len(materials) > 0 && materials[midx].albedo_tex.has_data {
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

	// Photon mapping buffers
	PHOTON_MAX_COUNT :: 1048576
	PHOTON_GRID_SIZE :: 16384
	PHOTON_MAX_PER_CELL :: 32
	photons := make([]Photon, PHOTON_MAX_COUNT)
	defer delete(photons)
	photon_counter: i32 = 0
	photon_grid_cells := make([]i32, PHOTON_GRID_SIZE * PHOTON_MAX_PER_CELL)
	defer delete(photon_grid_cells)
	photon_grid_counts := make([]i32, PHOTON_GRID_SIZE)
	defer delete(photon_grid_counts)

	photons_buffer := device->newBufferWithSlice(photons[:], MTL.ResourceStorageModeShared)
	photon_counter_slice := ([^]byte)(&photon_counter)[:size_of(i32)]
	photon_counter_buffer := device->newBufferWithBytes(photon_counter_slice, MTL.ResourceStorageModeShared)
	photon_grid_cells_buffer := device->newBufferWithSlice(photon_grid_cells[:], MTL.ResourceStorageModeShared)
	photon_grid_counts_buffer := device->newBufferWithSlice(photon_grid_counts[:], MTL.ResourceStorageModeShared)

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

	// Photon grid build pipeline
	photon_grid_func := library->newFunctionWithName(NS.AT("photonBuildGridKernel"))
	assert(photon_grid_func != nil, "photonBuildGridKernel not found")
	photon_grid_desc := MTL.ComputePipelineDescriptor.alloc()->init()
	photon_grid_desc->setComputeFunction(photon_grid_func)
	photon_grid_pipeline, pg_err := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
		device, photon_grid_desc, MTL.PipelineOption{}, nil,
	)
	if pg_err != nil {
		fmt.eprintln("Photon grid pipeline failed:", pg_err->localizedDescription()->odinString())
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

	cmd_buf := cmd_queue->commandBuffer()
	as_encoder := cmd_buf->accelerationStructureCommandEncoder()
	as_encoder->buildAccelerationStructure(as, prim_desc, scratch, 0)
	as_encoder->endEncoding()
	cmd_buf->commit()
	cmd_buf->waitUntilCompleted()
	fmt.println("  Done.")

	// Dispatch compute (three passes)
	fmt.println("Rendering...")

	dispatch_buf := cmd_queue->commandBuffer()

	// Pass 1: Photon emission
	if scene_data.photon_enabled != 0 && scene_data.photon_count > 0 {
		photon_enc := dispatch_buf->computeCommandEncoder()
		photon_enc->setComputePipelineState(photon_pipeline)
		photon_enc->setBuffer(scene_buffer, 0, 0)
		photon_enc->setBuffer(material_buffer, 0, 1)
		photon_enc->setAccelerationStructure(as, 3)
		photon_enc->setBuffer(vertex_buffer, 0, 4)
		photon_enc->setBuffer(index_buffer, 0, 5)
		photon_enc->setBuffer(tri_light_buffer, 0, 6)
		photon_enc->setBuffer(quad_light_buffer, 0, 7)
		photon_enc->setBuffer(sphere_light_buffer, 0, 8)
		photon_enc->setBuffer(mat_index_buffer, 0, 9)
		photon_enc->setBuffer(photons_buffer, 0, 14)
		photon_enc->setBuffer(photon_counter_buffer, 0, 15)
		photon_tg := MTL.Size{width = 64, height = 1, depth = 1}
		photon_gs := MTL.Size{
			width  = NS.Integer(min(scene_data.photon_count, PHOTON_MAX_COUNT)),
			height = 1,
			depth  = 1,
		}
		photon_enc->dispatchThreads(photon_gs, photon_tg)
		photon_enc->endEncoding()
	}

	// Pass 2: Photon grid build
	if scene_data.photon_enabled != 0 && scene_data.photon_count > 0 {
		pg_enc := dispatch_buf->computeCommandEncoder()
		pg_enc->setComputePipelineState(photon_grid_pipeline)
		pg_enc->setBuffer(photons_buffer, 0, 0)
		pg_enc->setBuffer(photon_counter_buffer, 0, 1)
		pg_enc->setBuffer(photon_grid_cells_buffer, 0, 2)
		pg_enc->setBuffer(photon_grid_counts_buffer, 0, 3)
		pg_enc->setBuffer(scene_buffer, 0, 4)
		pg_tg := MTL.Size{width = 64, height = 1, depth = 1}
		pg_gs := MTL.Size{
			width  = NS.Integer(min(scene_data.photon_count, PHOTON_MAX_COUNT)),
			height = 1,
			depth  = 1,
		}
		pg_enc->dispatchThreads(pg_gs, pg_tg)
		pg_enc->endEncoding()
	}

	// Pass 3: Main raytrace
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
	enc->setBuffer(photon_grid_cells_buffer, 0, 16)
	enc->setBuffer(photon_grid_counts_buffer, 0, 17)
	enc->setBuffer(tex_buffer, 0, 18)

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
	fmt.println("  Done.")

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
	if enable_aovs {
		fmt.println("AOV passes:", len(aov_passes))
		// Reset the GI cache and photon maps so each AOV pass sees
		// a fresh biased-GI state — the cache and photon contributions
		// are themselves path-trace results and would otherwise
		// leak between passes.
		gi_counter = 0
		for i in 0 ..< len(gi_grid_cells) { gi_grid_cells[i] = 0 }
		for i in 0 ..< len(gi_grid_counts) { gi_grid_counts[i] = 0 }
		photon_counter = 0
		for i in 0 ..< len(photon_grid_cells) { photon_grid_cells[i] = 0 }
		for i in 0 ..< len(photon_grid_counts) { photon_grid_counts[i] = 0 }

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
			aov_enc->setBuffer(photon_grid_cells_buffer, 0, 16)
			aov_enc->setBuffer(photon_grid_counts_buffer, 0, 17)
			aov_enc->setBuffer(tex_buffer, 0, 18)
			aov_enc->dispatchThreads(grid_size, tg_size)
			aov_enc->endEncoding()
			aov_cmd->commit()
			aov_cmd->waitUntilCompleted()

			// Read back the result for this AOV
			aov_data := output_buffer->contentsAsSlice([][4]f32)
			aov_results[aov_debug] = make([dynamic][4]f32, pixel_count)
			copy(aov_results[aov_debug][:], aov_data[:pixel_count])

			// Reset GI cache / photons for the next pass
			gi_counter = 0
			for i in 0 ..< len(gi_grid_cells) { gi_grid_cells[i] = 0 }
			for i in 0 ..< len(gi_grid_counts) { gi_grid_counts[i] = 0 }
			photon_counter = 0
			for i in 0 ..< len(photon_grid_cells) { photon_grid_cells[i] = 0 }
			for i in 0 ..< len(photon_grid_counts) { photon_grid_counts[i] = 0 }
		}
		// Reset debug_mode for the beauty readback
		scene_data.debug_mode = debug_mode
	}

	// Readback + write
	fmt.println("Writing", file_output)

	output_data := output_buffer->contentsAsSlice([][4]f32)
	pixels := make([]u8, pixel_count * 3)
	defer delete(pixels)

	for i in 0 ..< pixel_count {
		r := u8(clamp(output_data[i][0] * 255.0, 0.0, 255.0))
		g := u8(clamp(output_data[i][1] * 255.0, 0.0, 255.0))
		b := u8(clamp(output_data[i][2] * 255.0, 0.0, 255.0))
		pixels[i * 3 + 0] = r
		pixels[i * 3 + 1] = g
		pixels[i * 3 + 2] = b
	}

	stbi.flip_vertically_on_write(true)
	path_str := string(file_output)
	if strings.has_suffix(path_str, ".exr") {
		// Build the EXR image. Beauty is always the first layer.
		// When AOVs are enabled, additional layers (albedo, normal,
		// depth, direct, indirect) are appended.
		img: output.EXR_Image
		output.exr_image_init(&img, image_width, image_height)
		img.compression = output.EXR_COMPRESSION_ZIP
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
