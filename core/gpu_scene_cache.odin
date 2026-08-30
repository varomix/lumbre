package lumbre_core

// Scene-dependent GPU resources, cached on the renderer.
//
// Everything here is derived from the scene and is independent of the camera,
// the sample count, and the resolution. `gpu_render_frame` used to rebuild all
// of it on every call, which is invisible for a one-shot CLI render and fatal
// for an interactive viewport: every mouse movement paid the full cost.
//
// Measured per batch at 983x643 before this cache existed:
//
//     cornell   32 tris     37 ms
//     suzanne  ~1k tris     88 ms
//     helmet    15k tris   104 ms
//     guitar   157k tris   584 ms
//
// against 1-3.5 ms of actual sampling. The photon buffers alone are ~40 MB of
// allocation per call (1M photons plus grid), paid even when photon mapping is
// switched off.
//
// The photon map and irradiance cache live here too. Both are world-space and
// camera-independent, so they survive navigation: the viewport can show biased
// GI without rebuilding it every time the camera moves.

import "core:fmt"
import "core:time"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"
import m "core:math/linalg/glsl"

// Fixed capacities for the biased-GI structures. Package scope because the
// cache allocates them and the render pass dispatches against them.
GI_CACHE_MAX_POINTS :: 262144
PHOTON_MAX_COUNT    :: 1048576
PHOTON_GRID_SIZE    :: 16384

GPU_Scene_Cache :: struct {
	valid: bool,
	// Identifies the scene these resources were built from. The frontend bumps
	// this whenever geometry, materials, or lights change; camera moves must not.
	key:   u64,
	// Auto-derived radii depend on it, so a change forces a rebuild.
	photon_count_key: i32,

	// Geometry and shading.
	vertex_buffer:     ^MTL.Buffer,
	index_buffer:      ^MTL.Buffer,
	material_buffer:   ^MTL.Buffer,
	mat_index_buffer:  ^MTL.Buffer,
	tex_buffer:        ^MTL.Buffer,
	as:                ^MTL.AccelerationStructure,

	// Lights.
	tri_light_buffer:      ^MTL.Buffer,
	quad_light_buffer:     ^MTL.Buffer,
	sphere_light_buffer:   ^MTL.Buffer,
	disc_light_buffer:     ^MTL.Buffer,
	cylinder_light_buffer: ^MTL.Buffer,
	punctual_light_buffer: ^MTL.Buffer,

	// Environment.
	env_pixels_buffer:      ^MTL.Buffer,
	env_marginal_buffer:    ^MTL.Buffer,
	env_conditional_buffer: ^MTL.Buffer,

	// Biased-GI state. World-space and camera-independent, so it is reused
	// across camera moves; `gpu_scene_cache_reset_gi` clears it when the
	// lighting it was built from changes.
	gi_cache_buffer:      ^MTL.Buffer,
	gi_counter_buffer:    ^MTL.Buffer,
	gi_grid_cells_buffer: ^MTL.Buffer,
	gi_grid_counts_buffer: ^MTL.Buffer,

	photons_buffer:             ^MTL.Buffer,
	photon_counter_buffer:      ^MTL.Buffer,
	photon_cell_buffer:         ^MTL.Buffer,
	photon_grid_counts_buffer:  ^MTL.Buffer,
	photon_grid_offsets_buffer: ^MTL.Buffer,
	photon_grid_fill_buffer:    ^MTL.Buffer,
	photon_grid_sorted_buffer:  ^MTL.Buffer,
	// True once the photon map has been built for this scene; the build is
	// camera-independent so it happens once, not once per batch.
	photons_built: bool,

	// Scalars the per-render GPUSceneData needs.
	num_tris:             i32,
	tri_light_count:      i32,
	quad_light_count:     i32,
	sphere_light_count:   i32,
	disc_light_count:     i32,
	cylinder_light_count: i32,
	punctual_light_count: i32,
	has_env:              bool,
	env_width:            i32,
	env_height:           i32,
	env_rotation:         f32,
	env_intensity:        f32,
	env_func_int:         f32,
	effective_gi_cache_distance: f32,
	effective_photon_radius:     f32,
}

// Returns the cache, rebuilding it if `key` or `photon_count` no longer match.
gpu_scene_cache_ensure :: proc(
	rnd: ^GPU_Renderer,
	scene: ^Scene,
	key: u64,
	photon_count: i32,
	gi_cache_distance: f32,
	photon_radius: f32,
) -> bool {
	c := &rnd.cache
	if c.valid && c.key == key && c.photon_count_key == photon_count {
		return true
	}
	c^ = {}
	if !gpu_build_scene_cache(rnd, scene, photon_count, gi_cache_distance, photon_radius) {
		return false
	}
	c.key = key
	c.photon_count_key = photon_count
	c.valid = true
	return true
}

// Drops the accumulated irradiance cache and photon map without discarding
// geometry, so relighting does not pay for a full rebuild.
gpu_scene_cache_reset_gi :: proc(rnd: ^GPU_Renderer) {
	c := &rnd.cache
	if !c.valid {
		return
	}
	c.photons_built = false
	if c.gi_counter_buffer != nil {
		zero: i32 = 0
		src := ([^]byte)(&zero)[:size_of(i32)]
		copy(c.gi_counter_buffer->contents()[:size_of(i32)], src)
	}
	if c.gi_grid_counts_buffer != nil {
		counts := c.gi_grid_counts_buffer->contentsAsSlice([]i32)
		for i in 0 ..< len(counts) {
			counts[i] = 0
		}
	}
}

@(private = "file")
gpu_build_scene_cache :: proc(
	rnd: ^GPU_Renderer,
	scene: ^Scene,
	photon_count: i32,
	gi_cache_distance: f32,
	photon_radius: f32,
) -> bool {
	build_start := time.tick_now()
	device := rnd.device
	cmd_queue := rnd.queue

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
		return false	}

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
			params2  = {f32(mat.clearcoat_roughness), f32(mat.sheen), f32(mat.normal_scale), f32(mat.anisotropic)},
			spec_tint = {f32(mat.specular_tint.x), f32(mat.specular_tint.y), f32(mat.specular_tint.z), 0},
			sheen_tint = {f32(mat.sheen_tint.x), f32(mat.sheen_tint.y), f32(mat.sheen_tint.z), 0},
			tex_info  = pack_texture(&tex_pixels, mat.albedo_tex),
			mr_info   = pack_texture(&tex_pixels, mat.metallic_roughness_tex),
			nrm_info  = pack_texture(&tex_pixels, mat.normal_tex),
			emis_info = pack_texture(&tex_pixels, mat.emissive_tex),
			params3   = {f32(mat.spec_trans), 0, 0, 0},
			params4   = {f32(mat.subsurface_color.x), f32(mat.subsurface_color.y), f32(mat.subsurface_color.z), f32(mat.subsurface)},
			params5   = {f32(mat.subsurface_radius.x * mat.subsurface_scale), f32(mat.subsurface_radius.y * mat.subsurface_scale), f32(mat.subsurface_radius.z * mat.subsurface_scale), 0},
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
	// Build explicit analytic lights from scene lights
	gpu_quad_lights := make([dynamic]GPUQuadLight)
	gpu_sphere_lights := make([dynamic]GPUSphereLight)
	gpu_disc_lights := make([dynamic]GPUDiscLight)
	gpu_cylinder_lights := make([dynamic]GPUCylinderLight)
	gpu_punctual_lights := make([dynamic]GPUPunctualLight)
	defer delete(gpu_quad_lights)
	defer delete(gpu_sphere_lights)
	defer delete(gpu_disc_lights)
	defer delete(gpu_cylinder_lights)
	defer delete(gpu_punctual_lights)

	for l in flattened.lights {
		intensity := l.intensity
		emis := [4]f32{f32(intensity.x), f32(intensity.y), f32(intensity.z), 0}
		switch l.kind {
		case .Quad:
			append(&gpu_quad_lights, GPUQuadLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				u        = {f32(l.u.x), f32(l.u.y), f32(l.u.z), 0},
				v        = {f32(l.v.x), f32(l.v.y), f32(l.v.z), 0},
				emission = emis,
			})
		case .Sphere:
			append(&gpu_sphere_lights, GPUSphereLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				emission = emis,
				radius   = f32(l.radius),
			})
		case .Disc:
			append(&gpu_disc_lights, GPUDiscLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), f32(l.radius)},
				normal   = {f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), 0},
				emission = emis,
			})
		case .Cylinder:
			append(&gpu_cylinder_lights, GPUCylinderLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), f32(l.radius)},
				axis     = {f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), f32(l.height)},
				emission = emis,
			})
		case .Point:
			append(&gpu_punctual_lights, GPUPunctualLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				emission = emis,
				params   = {0, 0, 0, 0},
			})
		case .Spot:
			append(&gpu_punctual_lights, GPUPunctualLight{
				position  = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				direction = {f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), 0},
				emission  = emis,
				params    = {1, f32(l.cos_inner), f32(l.cos_outer), 0},
			})
		case .Distant:
			append(&gpu_punctual_lights, GPUPunctualLight{
				direction = {f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), 0},
				emission  = emis,
				params    = {2, 0, 0, f32(l.angular_radius)},
			})
		case .Mesh, .Dome:
			continue
		}
	}

	fmt.println("Light triangles:", len(gpu_lights))
	fmt.println("Quad lights:", len(gpu_quad_lights))
	fmt.println("Sphere lights:", len(gpu_sphere_lights))
	fmt.println("Disc lights:", len(gpu_disc_lights))
	fmt.println("Cylinder lights:", len(gpu_cylinder_lights))
	fmt.println("Punctual lights:", len(gpu_punctual_lights))

	// HDRI environment data. When absent, bind 1-element dummies so the buffer
	// pointers are valid (same pattern as `tex_buffer`).
	env := &scene.environment
	has_env := env.has_data
	env_pixels_slice := env.pixels
	env_marginal_slice := env.marginal_cdf
	env_conditional_slice := env.conditional_cdf
	dummy_f32 := [1]f32{0}
	if !has_env {
		env_pixels_slice = dummy_f32[:]
		env_marginal_slice = dummy_f32[:]
		env_conditional_slice = dummy_f32[:]
	}

	// Irradiance cache buffer + hash grid
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
	fmt.println("Material buffer size:", len(gpu_materials) * size_of(GPUMaterial), "bytes, count:", len(gpu_materials))

	// Create Metal buffers
	vertex_buffer := device->newBufferWithSlice(vertices[:], MTL.ResourceStorageModeShared)
	index_buffer := device->newBufferWithSlice(indices[:], MTL.ResourceStorageModeShared)
	material_buffer := device->newBufferWithSlice(gpu_materials[:], MTL.ResourceStorageModeShared)
	mat_index_buffer := device->newBufferWithSlice(mat_indices[:], MTL.ResourceStorageModeShared)
	tri_light_buffer := device->newBufferWithSlice(gpu_lights[:], MTL.ResourceStorageModeShared)
	quad_light_buffer := device->newBufferWithSlice(gpu_quad_lights[:], MTL.ResourceStorageModeShared)
	sphere_light_buffer := device->newBufferWithSlice(gpu_sphere_lights[:], MTL.ResourceStorageModeShared)
	disc_light_buffer := device->newBufferWithSlice(gpu_disc_lights[:], MTL.ResourceStorageModeShared)
	cylinder_light_buffer := device->newBufferWithSlice(gpu_cylinder_lights[:], MTL.ResourceStorageModeShared)
	punctual_light_buffer := device->newBufferWithSlice(gpu_punctual_lights[:], MTL.ResourceStorageModeShared)
	env_pixels_buffer := device->newBufferWithSlice(env_pixels_slice, MTL.ResourceStorageModeShared)
	env_marginal_buffer := device->newBufferWithSlice(env_marginal_slice, MTL.ResourceStorageModeShared)
	env_conditional_buffer := device->newBufferWithSlice(env_conditional_slice, MTL.ResourceStorageModeShared)
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

	c := &rnd.cache
	c.vertex_buffer = vertex_buffer
	c.index_buffer = index_buffer
	c.material_buffer = material_buffer
	c.mat_index_buffer = mat_index_buffer
	c.tex_buffer = tex_buffer
	c.as = as

	c.tri_light_buffer = tri_light_buffer
	c.quad_light_buffer = quad_light_buffer
	c.sphere_light_buffer = sphere_light_buffer
	c.disc_light_buffer = disc_light_buffer
	c.cylinder_light_buffer = cylinder_light_buffer
	c.punctual_light_buffer = punctual_light_buffer

	c.env_pixels_buffer = env_pixels_buffer
	c.env_marginal_buffer = env_marginal_buffer
	c.env_conditional_buffer = env_conditional_buffer

	c.gi_cache_buffer = gi_cache_buffer
	c.gi_counter_buffer = gi_counter_buffer
	c.gi_grid_cells_buffer = gi_grid_cells_buffer
	c.gi_grid_counts_buffer = gi_grid_counts_buffer

	c.photons_buffer = photons_buffer
	c.photon_counter_buffer = photon_counter_buffer
	c.photon_cell_buffer = photon_cell_buffer
	c.photon_grid_counts_buffer = photon_grid_counts_buffer
	c.photon_grid_offsets_buffer = photon_grid_offsets_buffer
	c.photon_grid_fill_buffer = photon_grid_fill_buffer
	c.photon_grid_sorted_buffer = photon_grid_sorted_buffer
	c.photons_built = false

	c.num_tris = num_tris
	c.tri_light_count = i32(len(gpu_lights))
	c.quad_light_count = i32(len(gpu_quad_lights))
	c.sphere_light_count = i32(len(gpu_sphere_lights))
	c.disc_light_count = i32(len(gpu_disc_lights))
	c.cylinder_light_count = i32(len(gpu_cylinder_lights))
	c.punctual_light_count = i32(len(gpu_punctual_lights))
	c.has_env = has_env
	c.env_width = env.width
	c.env_height = env.height
	c.env_rotation = f32(env.rotation)
	c.env_intensity = f32(env.intensity)
	c.env_func_int = f32(env.func_int)
	c.effective_gi_cache_distance = effective_gi_cache_distance
	c.effective_photon_radius = effective_photon_radius

	fmt.printfln("Scene GPU cache built: %d triangles [%.3f s]",
		num_tris, time.duration_seconds(time.tick_since(build_start)))
	return true
}

// Rewrites just the material buffer from `scene.materials`, leaving geometry,
// textures, lights and the acceleration structure alone.
//
// A material edit must not bump `scene_key`: that would rebuild the whole
// cache, which is ~0.6 s on a 157k-triangle scene and makes dragging a
// roughness slider unusable. The texture descriptors (`tex_info` and friends)
// carry offsets into the shared texture buffer computed during the cache build,
// so they are preserved rather than recomputed — only the scalar shading
// parameters are written.
//
// Returns false if there is no cache yet, in which case the caller can ignore
// it: the pending build will pick up the current materials anyway.
gpu_scene_cache_update_materials :: proc(rnd: ^GPU_Renderer, scene: ^Scene) -> bool {
	c := &rnd.cache
	if !c.valid || c.material_buffer == nil {
		return false
	}

	gpu_mats := c.material_buffer->contentsAsSlice([]GPUMaterial)
	n := min(len(gpu_mats), len(scene.materials))

	for i in 0 ..< n {
		mat := scene.materials[i]
		dst := &gpu_mats[i]

		kind_val := i32(0)
		switch mat.kind {
		case .Lambertian: kind_val = 0
		case .Metal:      kind_val = 1
		case .Dielectric: kind_val = 2
		case .Principled: kind_val = 3
		case .Emissive:   kind_val = 4
		}

		dst.albedo = {f32(mat.albedo.x), f32(mat.albedo.y), f32(mat.albedo.z), 0}
		dst.emission = {f32(mat.emission.x), f32(mat.emission.y), f32(mat.emission.z), 0}
		dst.params0 = {f32(kind_val), f32(mat.fuzz), f32(mat.ir), f32(mat.roughness)}
		dst.params1 = {f32(mat.metallic), f32(mat.emission_strength), f32(mat.specular), f32(mat.clearcoat)}
		dst.params2 = {f32(mat.clearcoat_roughness), f32(mat.sheen), f32(mat.normal_scale), f32(mat.anisotropic)}
		dst.spec_tint = {f32(mat.specular_tint.x), f32(mat.specular_tint.y), f32(mat.specular_tint.z), 0}
		dst.sheen_tint = {f32(mat.sheen_tint.x), f32(mat.sheen_tint.y), f32(mat.sheen_tint.z), 0}
		dst.params3 = {f32(mat.spec_trans), 0, 0, 0}
		dst.params4 = {
			f32(mat.subsurface_color.x),
			f32(mat.subsurface_color.y),
			f32(mat.subsurface_color.z),
			f32(mat.subsurface),
		}
		dst.params5 = {
			f32(mat.subsurface_radius.x * mat.subsurface_scale),
			f32(mat.subsurface_radius.y * mat.subsurface_scale),
			f32(mat.subsurface_radius.z * mat.subsurface_scale),
			0,
		}
		// tex_info / mr_info / nrm_info / emis_info deliberately untouched.
	}

	// Emissive materials feed the light lists, which are baked into the cache,
	// so a change of emission does not relight until the scene is rebuilt. The
	// photon map is rebuilt though, since it is cheap relative to geometry.
	c.photons_built = false
	return true
}
