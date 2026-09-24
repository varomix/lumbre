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
import "core:hash/xxhash"
import "core:slice"
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

	// Light selection for next-event estimation: a CDF over every light in
	// the kernel's order -- emissive triangles, quads, spheres, discs,
	// cylinders, punctual lights, then the dome -- by estimated power.
	light_cdf_buffer:      ^MTL.Buffer,
	light_count:           i32,
	light_inv_total_power: f32,
	env_select_prob:       f32,
	scene_radius:          f32, // sizes the power of a distant light and the dome
}

// One path tracer vertex; the kernel reads it as float4 position, normal, uv.
@(private = "file")
GPUTriVertex :: struct {
	pos:    [4]f32,
	normal: [4]f32,
	uv:     [4]f32, // x, y, has_uv (0/1), unused
}

// An open-addressed table from vertex to index, reused across weld runs.
// Slots hold the vertex's index within the run plus one; zero is empty.
@(private = "file")
gpu_weld_reset :: proc(table: ^[dynamic]u32, corners: int) {
	size := 16
	for size < corners * 2 {
		size *= 2
	}
	resize(table, size)
	slice.zero(table[:])
}

// The global index of `v`, appending it to `verts` if this run has not seen
// it. Exact bitwise equality, so welding never changes what is rendered.
@(private = "file")
gpu_weld_insert :: proc(table: []u32, verts: ^[dynamic]GPUTriVertex, first: int, v: GPUTriVertex) -> u32 {
	v := v
	mask := len(table) - 1
	slot := int(xxhash.XXH3_64_default(slice.bytes_from_ptr(&v, size_of(GPUTriVertex)))) & mask
	for {
		entry := table[slot]
		if entry == 0 {
			local := u32(len(verts) - first)
			table[slot] = local + 1
			append(verts, v)
			return u32(first) + local
		}
		if verts[first + int(entry - 1)] == v {
			return u32(first) + entry - 1
		}
		slot = (slot + 1) & mask
	}
}

// Releases every Metal object the cache owns and clears it. Called before a
// rebuild, which used to drop the handles and leak the whole previous scene:
// geometry, acceleration structure, textures and GI buffers.
gpu_scene_cache_release :: proc(c: ^GPU_Scene_Cache) {
	for buf in ([]^MTL.Buffer{
		c.vertex_buffer, c.index_buffer, c.material_buffer, c.mat_index_buffer, c.tex_buffer,
		c.tri_light_buffer, c.quad_light_buffer, c.sphere_light_buffer, c.disc_light_buffer,
		c.cylinder_light_buffer, c.punctual_light_buffer,
		c.env_pixels_buffer, c.env_marginal_buffer, c.env_conditional_buffer,
		c.gi_cache_buffer, c.gi_counter_buffer, c.gi_grid_cells_buffer, c.gi_grid_counts_buffer,
		c.photons_buffer, c.photon_counter_buffer, c.photon_cell_buffer, c.photon_grid_counts_buffer,
		c.photon_grid_offsets_buffer, c.photon_grid_fill_buffer, c.photon_grid_sorted_buffer,
		c.light_cdf_buffer,
	}) {
		gpu_release(buf)
	}
	gpu_release(c.as)
	c^ = {}
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
	gpu_scene_cache_release(c)
	if !gpu_build_scene_cache(rnd, scene, photon_count, gi_cache_distance, photon_radius) {
		return false
	}
	c.key = key
	c.photon_count_key = photon_count
	c.valid = true
	return true
}

// Drops the photon map alone, leaving the irradiance cache standing. The photon
// map is re-emitted in one GPU pass, whereas the irradiance cache is gathered
// over many batches — so for an ordinary relight this is the cheap half of
// gpu_scene_cache_reset_gi and buys most of the correctness.
gpu_scene_cache_reset_photons :: proc(rnd: ^GPU_Renderer) {
	if !rnd.cache.valid {
		return
	}
	rnd.cache.photons_built = false
}

// Drops the accumulated irradiance cache and photon map without discarding
// geometry, so relighting does not pay for a full rebuild.
//
// The irradiance cache is what makes the viewport look converged, and refilling
// it takes many batches, so reach for this only when the cached irradiance
// answers a question that is no longer being asked — the gather settings
// themselves changed. A material or light edit takes reset_photons instead.
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

	// Flattened triangles and materials, in one copy each.
	sphere_tri_count := 0
	for _ in scene.spheres {
		sphere_tri_count += ICOSPHERE_TRIANGLES
	}
	reserve(&all_triangles, len(flattened.triangles) + sphere_tri_count)
	append(&all_triangles, ..flattened.triangles)
	append(&materials, ..flattened.materials)

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

	// Corners are welded (see gpu_weld_insert), so the vertex count is not
	// known up front; a mesh typically needs about one vertex per two corners.
	vertices := make([dynamic]GPUTriVertex, 0, int(num_tris) * 3 / 2)
	indices := make([]u32, num_tris * 3)
	mat_indices := make([]i32, num_tris)
	defer delete(vertices)
	weld: [dynamic]u32
	defer delete(weld)
	run_key := i32(min(i32))
	run_first_vertex := 0
	// Which weld run a triangle belongs to: its scene node, or for the
	// spheres appended after the flattened scene, which sphere.
	run_of :: proc(i: int, node_idx: []i32) -> i32 {
		if i < len(node_idx) {
			return node_idx[i]
		}
		return -1 - i32((i - len(node_idx)) / ICOSPHERE_TRIANGLES)
	}
	defer delete(indices)
	defer delete(mat_indices)

	// Build a single combined texture buffer from every per-material map.
	// Each map's pixels are appended in order and its start offset plus
	// dimensions recorded in the matching `*_info` field. Texture data is
	// RGBA8; the GPU decodes to floats during sampling. The color space is
	// baked into which shader path reads the map, not into the bytes.
	tex_pixels := make([dynamic]u8)
	defer delete(tex_pixels)
	// Maps shared between materials -- one texture set on several materials,
	// or equal pixels from separate loads -- are stored once.
	packed := make(map[Texture_Content]u32)
	defer delete(packed)
	total_tex_bytes := 0
	for mat in materials {
		for tex in ([]TextureMap{mat.albedo_tex, mat.metallic_roughness_tex, mat.normal_tex, mat.emissive_tex}) {
			if tex.has_data {
				total_tex_bytes += len(tex.pixels)
			}
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
		gpu_materials[i] = GPUMaterial {
			albedo   = {f32(mat.albedo.x), f32(mat.albedo.y), f32(mat.albedo.z), 0},
			emission = {f32(mat.emission.x), f32(mat.emission.y), f32(mat.emission.z), 0},
			params0  = {f32(kind_val), f32(mat.fuzz), f32(mat.ir), f32(mat.roughness)},
			params1  = {f32(mat.metallic), f32(mat.emission_strength), f32(mat.specular), f32(mat.clearcoat)},
			params2  = {f32(mat.clearcoat_roughness), f32(mat.sheen), f32(mat.normal_scale), f32(mat.anisotropic)},
			spec_tint = {f32(mat.specular_tint.x), f32(mat.specular_tint.y), f32(mat.specular_tint.z), 0},
			sheen_tint = {f32(mat.sheen_tint.x), f32(mat.sheen_tint.y), f32(mat.sheen_tint.z), 0},
			tex_info  = pack_texture(&tex_pixels, &packed, mat.albedo_tex),
			mr_info   = pack_texture(&tex_pixels, &packed, mat.metallic_roughness_tex),
			nrm_info  = pack_texture(&tex_pixels, &packed, mat.normal_tex),
			emis_info = pack_texture(&tex_pixels, &packed, mat.emissive_tex),
			params3   = {f32(mat.spec_trans), 0, 0, 0},
			params4   = {f32(mat.subsurface_color.x), f32(mat.subsurface_color.y), f32(mat.subsurface_color.z), f32(mat.subsurface)},
			params5   = {f32(mat.subsurface_radius.x * mat.subsurface_scale), f32(mat.subsurface_radius.y * mat.subsurface_scale), f32(mat.subsurface_radius.z * mat.subsurface_scale), 0},
		}
	}

	// Build the vertex and index buffers. Corners are welded one scene node
	// at a time: a smooth mesh repeats each vertex across about six triangles,
	// and storing every corner separately cost ~4x the memory -- 1.25 GB of
	// vertices for Kitchen_set. Welding across nodes would find nothing a
	// node's own mesh does not, and would need a table the size of the scene.
	for i in 0 ..< num_tris {
		tri := all_triangles[i]
		base := i * 3
		if key := run_of(int(i), flattened.node_idx); key != run_key {
			run_key = key
			run_first_vertex = len(vertices)
			run_end := int(i)
			for run_end < int(num_tris) && run_of(run_end, flattened.node_idx) == key {
				run_end += 1
			}
			gpu_weld_reset(&weld, (run_end - int(i)) * 3)
		}
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

		corners := [3]GPUTriVertex{
			{
				pos    = {f32(tri.v0.x), f32(tri.v0.y), f32(tri.v0.z), 0},
				normal = {f32(n0.x), f32(n0.y), f32(n0.z), 0},
				uv     = {f32(tri.uv0.x), f32(tri.uv0.y), has_uv_a, 0},
			},
			{
				pos    = {f32(tri.v1.x), f32(tri.v1.y), f32(tri.v1.z), 0},
				normal = {f32(n1.x), f32(n1.y), f32(n1.z), 0},
				uv     = {f32(tri.uv1.x), f32(tri.uv1.y), has_uv_b, 0},
			},
			{
				pos    = {f32(tri.v2.x), f32(tri.v2.y), f32(tri.v2.z), 0},
				normal = {f32(n2.x), f32(n2.y), f32(n2.z), 0},
				uv     = {f32(tri.uv2.x), f32(tri.uv2.y), has_uv_c, 0},
			},
		}
		for corner, k in corners {
			indices[int(base) + k] = gpu_weld_insert(weld[:], &vertices, run_first_vertex, corner)
		}

		mat_indices[i] = midx
	}

	fmt.printfln("Vertices: %d, welded from %d corners", len(vertices), num_tris * 3)

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
		// p0.w carries the material index, so a material edit can refresh
		// the emission without rebuilding; see gpu_scene_cache_update_materials.
		append(&gpu_lights, GPULightTriangle {
			p0       = {f32(tri.v0.x), f32(tri.v0.y), f32(tri.v0.z), f32(midx)},
			p1       = {f32(tri.v1.x), f32(tri.v1.y), f32(tri.v1.z), 0},
			p2       = {f32(tri.v2.x), f32(tri.v2.y), f32(tri.v2.z), 0},
			emission = gpu_emissive_radiance(mat),
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

	gpu_convert_lights(
		flattened.lights,
		&gpu_quad_lights, &gpu_sphere_lights, &gpu_disc_lights,
		&gpu_cylinder_lights, &gpu_punctual_lights,
	)
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
	scratch->release()
	geom_array->release()
	prim_desc->release()
	tri_geom->release()

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
	c.scene_radius = f32(m.length(scene_size)) * 0.5

	// One entry per light, plus the dome. Counts are fixed until the next
	// build, so the updates below rewrite this buffer in place.
	c.light_count = c.tri_light_count + c.quad_light_count + c.sphere_light_count +
		c.disc_light_count + c.cylinder_light_count + c.punctual_light_count + (has_env ? 1 : 0)
	c.light_cdf_buffer = device->newBufferWithLength(
		NS.UInteger(size_of(f32) * (c.light_count + 1)), MTL.ResourceStorageModeShared,
	)
	gpu_scene_cache_build_light_cdf(c)

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

	// Emissive triangles cache their material's radiance for light sampling.
	// Leaving it stale made light samples and hits on the emitter disagree
	// about how bright it is until the next full rebuild. (A material that
	// starts or stops emitting changes the light list itself, which still
	// needs the rebuild.)
	if c.tri_light_count > 0 {
		for &lt in c.tri_light_buffer->contentsAsSlice([]GPULightTriangle)[:c.tri_light_count] {
			midx := int(lt.p0.w)
			if midx >= 0 && midx < len(gpu_mats) && i32(gpu_mats[midx].params0[0]) == 4 {
				lt.emission = gpu_emissive_radiance(gpu_mats[midx])
			}
		}
	}
	gpu_scene_cache_build_light_cdf(c)

	// The photon map and irradiance cache are *not* dropped here. They are
	// emitted from the lighting, so a material edit does invalidate them, but
	// rebuilding costs a full GPU round trip and dropping them costs every
	// sample already cached — far too much to pay on each frame of a slider
	// drag. The caller decides when the edit has settled and calls
	// gpu_scene_cache_reset_gi then.
	return true
}

// Converts the scene's analytic lights into their GPU layouts. Shared by the
// cache build and by gpu_scene_cache_update_lights, so the two cannot drift.
gpu_convert_lights :: proc(
	lights: []Light,
	quad: ^[dynamic]GPUQuadLight,
	sphere: ^[dynamic]GPUSphereLight,
	disc: ^[dynamic]GPUDiscLight,
	cylinder: ^[dynamic]GPUCylinderLight,
	punctual: ^[dynamic]GPUPunctualLight,
) {
	clear(quad); clear(sphere); clear(disc); clear(cylinder); clear(punctual)
	for l in lights {
		intensity := l.intensity
		emis := [4]f32{f32(intensity.x), f32(intensity.y), f32(intensity.z), 0}
		switch l.kind {
		case .Quad:
			append(quad, GPUQuadLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				u        = {f32(l.u.x), f32(l.u.y), f32(l.u.z), 0},
				v        = {f32(l.v.x), f32(l.v.y), f32(l.v.z), 0},
				emission = emis,
			})
		case .Sphere:
			append(sphere, GPUSphereLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				emission = emis,
				radius   = f32(l.radius),
			})
		case .Disc:
			append(disc, GPUDiscLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), f32(l.radius)},
				normal   = {f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), 0},
				emission = emis,
			})
		case .Cylinder:
			append(cylinder, GPUCylinderLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), f32(l.radius)},
				axis     = {f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), f32(l.height)},
				emission = emis,
			})
		case .Point:
			append(punctual, GPUPunctualLight{
				position = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				emission = emis,
				params   = {0, 0, 0, 0},
			})
		case .Spot:
			append(punctual, GPUPunctualLight{
				position  = {f32(l.position.x), f32(l.position.y), f32(l.position.z), 0},
				direction = {f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), 0},
				emission  = emis,
				params    = {1, f32(l.cos_inner), f32(l.cos_outer), 0},
			})
		case .Distant:
			append(punctual, GPUPunctualLight{
				direction = {f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), 0},
				emission  = emis,
				params    = {2, 0, 0, f32(l.angular_radius)},
			})
		case .Mesh, .Dome:
			continue
		}
	}
}

// Rewrites the analytic light buffers from `scene.lights` without rebuilding
// geometry, textures or the acceleration structure — the same reasoning as
// gpu_scene_cache_update_materials: a full rebuild is ~0.6 s on a large scene
// and would make dragging a light slider useless.
//
// Returns false when the per-kind light counts have changed, which happens if a
// light's *type* changed or lights were added or removed. The buffers are sized
// at build time, so the caller must bump the scene key and rebuild in that case.
gpu_scene_cache_update_lights :: proc(rnd: ^GPU_Renderer, scene: ^Scene) -> bool {
	c := &rnd.cache
	if !c.valid {
		return false
	}

	quad := make([dynamic]GPUQuadLight, context.temp_allocator)
	sphere := make([dynamic]GPUSphereLight, context.temp_allocator)
	disc := make([dynamic]GPUDiscLight, context.temp_allocator)
	cylinder := make([dynamic]GPUCylinderLight, context.temp_allocator)
	punctual := make([dynamic]GPUPunctualLight, context.temp_allocator)

	// Lights live on the scene graph, so they are flattened the same way the
	// build does it — a light's world transform matters.
	flat := flatten_scene_graph(scene, context.temp_allocator)
	gpu_convert_lights(flat.lights, &quad, &sphere, &disc, &cylinder, &punctual)

	if i32(len(quad)) != c.quad_light_count ||
	   i32(len(sphere)) != c.sphere_light_count ||
	   i32(len(disc)) != c.disc_light_count ||
	   i32(len(cylinder)) != c.cylinder_light_count ||
	   i32(len(punctual)) != c.punctual_light_count {
		return false
	}

	copy(c.quad_light_buffer->contentsAsSlice([]GPUQuadLight)[:len(quad)], quad[:])
	copy(c.sphere_light_buffer->contentsAsSlice([]GPUSphereLight)[:len(sphere)], sphere[:])
	copy(c.disc_light_buffer->contentsAsSlice([]GPUDiscLight)[:len(disc)], disc[:])
	copy(c.cylinder_light_buffer->contentsAsSlice([]GPUCylinderLight)[:len(cylinder)], cylinder[:])
	copy(c.punctual_light_buffer->contentsAsSlice([]GPUPunctualLight)[:len(punctual)], punctual[:])
	gpu_scene_cache_build_light_cdf(c)

	// The photon map was emitted from the old lighting, but dropping it is the
	// caller's call — see the note in gpu_scene_cache_update_materials.
	return true
}

// What identifies a map's pixels for sharing: its size and a hash of its bytes.
@(private = "file")
Texture_Content :: struct {
	hash:          u64,
	width, height: i32,
}

// Appends `tex` to the shared pixel buffer, or finds an equal map already
// there, and returns its descriptor: {pixel_offset, width, height, has_tex}.
//
// The offset travels as the raw BITS of a u32 in the float slot, read back
// with `as_type<uint>` in the kernel. As a float VALUE it was exact only to
// 2^24 pixels -- four 2K maps -- and every map past that sampled from the
// wrong place.
@(private = "file")
pack_texture :: proc(buf: ^[dynamic]u8, packed: ^map[Texture_Content]u32, tex: TextureMap) -> [4]f32 {
	if !tex.has_data || len(tex.pixels) == 0 {
		return {0, 0, 0, 0}
	}
	key := Texture_Content{xxhash.XXH3_64_default(tex.pixels), tex.width, tex.height}
	offset, found := packed[key]
	if !found {
		pixel_index := len(buf) / 4
		if pixel_index + len(tex.pixels) / 4 > int(max(u32)) {
			fmt.eprintln("Texture buffer exceeds 2^32 pixels; dropping a map")
			return {0, 0, 0, 0}
		}
		offset = u32(pixel_index)
		append(buf, ..tex.pixels)
		packed[key] = offset
	}
	return {transmute(f32)offset, f32(tex.width), f32(tex.height), 1.0}
}

// An emissive material's radiance, exactly as the kernel's `emissive_radiance`
// computes it: the emission colour, or the albedo when that is black, times a
// strength that defaults to 20. Light samples and hits on the emitter must
// agree on this, or their MIS weights stop summing to one.
gpu_emissive_radiance :: proc(mat: GPUMaterial) -> [4]f32 {
	color := mat.emission
	if gpu_luminance(color) <= 0 {
		color = mat.albedo
	}
	strength := mat.params1[1]
	if strength <= 0 {
		strength = 20.0
	}
	return {color[0] * strength, color[1] * strength, color[2] * strength, 0}
}

// Mirrors `luminance` in shaders/raytrace.metal.
gpu_luminance :: proc(c: [4]f32) -> f32 {
	return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

// Rebuilds the light-selection CDF from the light buffers, which are shared
// memory the CPU can read back.
//
// Each light is chosen in proportion to an estimate of its power, so a small
// bright lamp is not drowned out by a large dim panel, or by the thousand
// triangles of one emissive mesh. Power ignores distance and visibility: it
// steers variance only, never the expected value, since each sample is divided
// by the probability it was chosen with.
//
// Area lights emit pi * L * A; a point light 4 pi I; a spot the same over its
// cone; a distant light and the dome deliver their irradiance over the
// scene's cross-section.
gpu_scene_cache_build_light_cdf :: proc(c: ^GPU_Scene_Cache) {
	if c.light_cdf_buffer == nil {
		return
	}
	cdf := c.light_cdf_buffer->contentsAsSlice([]f32)[:c.light_count + 1]
	PI :: f32(3.14159265358979)
	cross_section := PI * c.scene_radius * c.scene_radius

	powers := make([]f64, c.light_count, context.temp_allocator)
	i := 0
	for lt in c.tri_light_buffer->contentsAsSlice([]GPULightTriangle)[:c.tri_light_count] {
		e1 := m.vec3{lt.p1.x - lt.p0.x, lt.p1.y - lt.p0.y, lt.p1.z - lt.p0.z}
		e2 := m.vec3{lt.p2.x - lt.p0.x, lt.p2.y - lt.p0.y, lt.p2.z - lt.p0.z}
		area := 0.5 * m.length(m.cross(e1, e2))
		powers[i] = f64(gpu_tri_light_power(gpu_luminance(lt.emission), area)); i += 1
	}
	for q in c.quad_light_buffer->contentsAsSlice([]GPUQuadLight)[:c.quad_light_count] {
		area := m.length(m.cross(m.vec3{q.u.x, q.u.y, q.u.z}, m.vec3{q.v.x, q.v.y, q.v.z}))
		powers[i] = f64(PI * gpu_luminance(q.emission) * area); i += 1
	}
	for s in c.sphere_light_buffer->contentsAsSlice([]GPUSphereLight)[:c.sphere_light_count] {
		powers[i] = f64(PI * gpu_luminance(s.emission) * 4 * PI * s.radius * s.radius); i += 1
	}
	for d in c.disc_light_buffer->contentsAsSlice([]GPUDiscLight)[:c.disc_light_count] {
		powers[i] = f64(PI * gpu_luminance(d.emission) * PI * d.position.w * d.position.w); i += 1
	}
	for cy in c.cylinder_light_buffer->contentsAsSlice([]GPUCylinderLight)[:c.cylinder_light_count] {
		powers[i] = f64(PI * gpu_luminance(cy.emission) * 2 * PI * cy.position.w * cy.axis.w); i += 1
	}
	for p in c.punctual_light_buffer->contentsAsSlice([]GPUPunctualLight)[:c.punctual_light_count] {
		lum := gpu_luminance(p.emission)
		switch i32(p.params.x) {
		case 0: powers[i] = f64(4 * PI * lum)
		case 1: powers[i] = f64(2 * PI * (1 - 0.5 * (p.params.y + p.params.z)) * lum)
		case:   powers[i] = f64(lum * cross_section)
		}
		i += 1
	}
	env_power: f64
	if c.has_env {
		// `env_func_int` is the mean of luminance * sin(theta) over the map, so
		// the mean radiance over the sphere is (pi / 2) * env_func_int.
		mean_radiance := (PI / 2) * c.env_func_int * c.env_intensity
		env_power = f64(PI * mean_radiance * cross_section)
		powers[i] = env_power; i += 1
	}

	total: f64
	for p in powers { total += max(p, 0) }
	cdf[0] = 0
	if total <= 0 {
		// Nothing has measurable power (all black, say): choose uniformly.
		for k in 0 ..< int(c.light_count) {
			cdf[k + 1] = f32(k + 1) / f32(c.light_count)
		}
		c.light_inv_total_power = 0
		c.env_select_prob = c.has_env ? 1 / f32(c.light_count) : 0
		return
	}
	acc: f64
	for p, k in powers {
		acc += max(p, 0)
		cdf[k + 1] = f32(acc / total)
	}
	cdf[c.light_count] = 1
	c.light_inv_total_power = f32(1 / total)
	c.env_select_prob = f32(env_power / total)
}

// An emissive triangle's power. The kernel computes the same thing when a BSDF
// ray hits the triangle, to weight the hit against light sampling.
gpu_tri_light_power :: proc(luminance, area: f32) -> f32 {
	return 3.14159265358979 * luminance * area
}
