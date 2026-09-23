package lumbre_realtime

// Turns a `core.Scene` into buffers a rasterizer can draw.
//
// Geometry is uploaded once per distinct mesh, in the mesh's own space, and each
// scene node that draws it becomes an INSTANCE carrying its world transform and
// instance id. A prototype placed a thousand times costs one copy of its
// vertices and a thousand small records -- baking transforms into the vertices,
// as the path tracer's flattening does, would cost a thousand copies. Both modes
// still read the same scene graph and tessellate spheres with the same
// `build_icosphere`, so they draw the same geometry.
//
// The layout differs from the path tracer's because the two renderers want
// opposite things:
//
//   - The path tracer wants one flat buffer it can index by triangle id from
//     anywhere in a kernel. Material is a per-triangle lookup.
//   - The rasterizer wants triangles GROUPED BY MATERIAL, because binding a
//     pipeline's textures is per-draw. A soup in arbitrary order would mean one
//     draw call per triangle.
//
// So each mesh's triangles are sorted by material and emitted as contiguous
// runs, one draw per material per mesh, instanced over that mesh's nodes.
//
// The source `Triangle` is unshared, but most of its corners are not: a smooth
// mesh repeats each vertex across about six triangles. Each mesh's corners are
// welded on exact equality and drawn through an index buffer, which cuts vertex
// memory and vertex shading several times over. A draw is an index range plus
// the mesh's base vertex.
//
// Textures are the other divergence. The path tracer concatenates every map
// into one mip-less byte buffer indexed by offset, because a compute kernel has
// no sampler. Here each map becomes a real `sdl.GPUTexture` with a generated
// mip chain, sampled by hardware — which is most of why the raster mode can
// afford to be cheap.

import "core:fmt"
import "core:hash/xxhash"
import "core:math"
import "core:slice"

import lc "../core"
import m "core:math/linalg/glsl"
import sdl "vendor:sdl3"

// A triangle's material and its position before sorting.
@(private = "file")
Sort_Key :: struct {
	mat_idx: i32,
	index:   i32,
}

// One vertex, interleaved, in its mesh's own space. 48 bytes; the layout is
// mirrored by `VERTEX_ATTRIBUTES` below and by the shaders' input structs.
Vertex :: struct {
	pos:     [3]f32,
	normal:  [3]f32,
	uv:      [2]f32,
	// xyz = tangent, w = bitangent handedness. Per-face, matching the path
	// tracer's derivation, and only for materials with a normal map -- nothing
	// else reads it. Everything else carries NO_TANGENT, so the tangent never
	// stops a corner welding with its neighbours.
	tangent: [4]f32,
}

NO_TANGENT :: [4]f32{1, 0, 0, 1}

// One placement of a mesh: a scene node, or a tessellated sphere. Read per
// instance by the vertex stages, from the second vertex buffer.
Instance_GPU :: struct {
	// World transform, as four columns.
	world:  [4][4]f32,
	// Inverse-transpose of the transform's linear part, as three columns, so
	// normals stay perpendicular under non-uniform scale.
	normal: [3][3]f32,
	// Scene node index, the label pass's instance id; spheres continue past the
	// node range. An integer end to end: f32 is exact only to 2^24, and a
	// PointInstancer expands to one node per point.
	id:     u32,
}

// World-space box of one instance, kept on the CPU for culling.
Instance_Bounds :: struct {
	lo, hi: [3]f32,
}

// Everything the fragment shader needs about a material that is not a texture.
// Vectors only, so MSL and SPIR-V agree on the layout without padding games.
Material_Uniforms :: struct {
	// rgb = base colour, w = 1 when an albedo map is bound.
	base_color: [4]f32,
	// rgb = emission * strength, w = 1 when an emissive map is bound.
	emission:   [4]f32,
	// roughness, metallic, specular, normal_scale
	params:     [4]f32,
	// has_metallic_roughness_map, has_normal_map, spec_trans, ior
	flags:      [4]f32,
}

// Materials at or above this transmission are drawn by the forward pass
// instead of the G-buffer. Below it the transmission is not worth a second
// pass, and the surface shades as an opaque dielectric.
TRANSPARENT_THRESHOLD :: 0.01

// One material run of one mesh, drawn once per instance of that mesh.
Draw_Batch :: struct {
	// Index range into the scene's index buffer. Indices are local to the
	// mesh; `base_vertex` places them in the shared vertex buffer.
	first_index:    u32,
	index_count:    u32,
	base_vertex:    i32,
	// The mesh's instances. Every batch cut from one mesh shares this range.
	first_instance: u32,
	instance_count: u32,
	// Index into the scene's materials, continuing into its spheres. Carried so
	// a batch can be traced back to its material when a render looks wrong.
	material_index: i32,
	material:       Material_Uniforms,
	// Drawn by the forward pass rather than the G-buffer; see
	// TRANSPARENT_THRESHOLD.
	transparent:    bool,
	// Borrowed from the renderer texture cache, or from scene fallbacks. Never nil:
	// SDL binds a fixed number of samplers per pipeline, so an absent map is a
	// 1x1 default rather than a hole.
	albedo:         ^sdl.GPUTexture,
	mr:             ^sdl.GPUTexture,
	normal:         ^sdl.GPUTexture,
	emissive:       ^sdl.GPUTexture,
}

Scene_GPU :: struct {
	// Matches the IPR's `scene_key`. Bumped only when the scene itself
	// changes, so navigating never rebuilds any of this.
	key:             u64,
	vertices:        ^sdl.GPUBuffer,
	vertex_count:    u32,
	indices:         ^sdl.GPUBuffer, // u32, local to each mesh
	index_count:     u32,
	instances:       ^sdl.GPUBuffer,
	instance_count:  u32,
	batches:         []Draw_Batch,
	instance_bounds: []Instance_Bounds,
	// Which instances the current pass keeps, parallel to `instance_bounds`.
	// Reused across passes rather than allocated per frame.
	visible:         [dynamic]bool,

	// Owned sampler and fallback textures. Material textures live in the renderer cache.
	sampler:         ^sdl.GPUSampler,
	white:           ^sdl.GPUTexture, // stands in for a missing colour map
	flat_normal:     ^sdl.GPUTexture, // stands in for a missing normal map

	// World-space bounds, kept for the shadow cascades and for framing.
	bounds_min:      [3]f32,
	bounds_max:      [3]f32,
}

// Slot 0 is per vertex; slot 1 is per instance. Offsets into `Instance_GPU`
// step one column at a time.
VERTEX_ATTRIBUTES := [12]sdl.GPUVertexAttribute {
	{location = 0, buffer_slot = 0, format = .FLOAT3, offset = u32(offset_of(Vertex, pos))},
	{location = 1, buffer_slot = 0, format = .FLOAT3, offset = u32(offset_of(Vertex, normal))},
	{location = 2, buffer_slot = 0, format = .FLOAT2, offset = u32(offset_of(Vertex, uv))},
	{location = 3, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(Vertex, tangent))},
	{location = 4, buffer_slot = 1, format = .FLOAT4, offset = u32(offset_of(Instance_GPU, world))},
	{location = 5, buffer_slot = 1, format = .FLOAT4, offset = u32(offset_of(Instance_GPU, world)) + 4 * size_of(f32)},
	{location = 6, buffer_slot = 1, format = .FLOAT4, offset = u32(offset_of(Instance_GPU, world)) + 8 * size_of(f32)},
	{location = 7, buffer_slot = 1, format = .FLOAT4, offset = u32(offset_of(Instance_GPU, world)) + 12 * size_of(f32)},
	{location = 8, buffer_slot = 1, format = .FLOAT3, offset = u32(offset_of(Instance_GPU, normal))},
	{location = 9, buffer_slot = 1, format = .FLOAT3, offset = u32(offset_of(Instance_GPU, normal)) + 3 * size_of(f32)},
	{location = 10, buffer_slot = 1, format = .FLOAT3, offset = u32(offset_of(Instance_GPU, normal)) + 6 * size_of(f32)},
	{location = 11, buffer_slot = 1, format = .UINT, offset = u32(offset_of(Instance_GPU, id))},
}

VERTEX_BUFFER_DESCRIPTION := [2]sdl.GPUVertexBufferDescription {
	{slot = 0, pitch = u32(size_of(Vertex)), input_rate = .VERTEX},
	{slot = 1, pitch = u32(size_of(Instance_GPU)), input_rate = .INSTANCE},
}

vertex_input_state :: proc() -> sdl.GPUVertexInputState {
	return sdl.GPUVertexInputState {
		vertex_buffer_descriptions = raw_data(&VERTEX_BUFFER_DESCRIPTION),
		num_vertex_buffers = len(VERTEX_BUFFER_DESCRIPTION),
		vertex_attributes = raw_data(&VERTEX_ATTRIBUTES),
		num_vertex_attributes = len(VERTEX_ATTRIBUTES),
	}
}

// ── build ────────────────────────────────────────────────────────────────────

// The CPU half of an upload. Free with `scene_free_cpu`.
Scene_CPU :: struct {
	batches:         []Draw_Batch,
	verts:           []Vertex,
	indices:         []u32,
	instances:       []Instance_GPU,
	instance_bounds: []Instance_Bounds,
	bounds_min:      [3]f32,
	bounds_max:      [3]f32,
}

// Nodes draw the same vertices when they share a triangle array -- copies of
// one instanced prototype do (see Mesh.borrowed_triangles) -- and the same
// material override, which is baked into the batches.
@(private = "file")
Geometry_Key :: struct {
	triangles: rawptr,
	count:     int,
	override:  i32,
}

@(private = "file")
Geometry :: struct {
	triangles: []lc.Triangle,
	override:  i32, // material for every triangle, or -1
	sphere:    bool,
	ids:       [dynamic]i32, // one per instance: node index, or a sphere's id
}

// The CPU half: group nodes by the geometry they draw, sort each group's
// triangles by material, build local-space vertices, batch ranges and one
// instance record per node. Split out from the GPU upload so it can be tested
// without a device — the batching is where the bugs that survive a screenshot
// live.
//
// The returned batches have no textures bound yet; `scene_upload` fills those in.
scene_build_cpu :: proc(scene: ^lc.Scene) -> (cpu: Scene_CPU, ok: bool) {
	lc.compute_world_transforms(scene.nodes)

	groups := make([dynamic]Geometry)
	defer {
		for g in groups {
			delete(g.ids)
		}
		delete(groups)
	}
	group_of := make(map[Geometry_Key]int)
	defer delete(group_of)

	for node, ni in scene.nodes {
		if node.mesh_idx < 0 || int(node.mesh_idx) >= len(scene.meshes) {
			continue
		}
		tris := scene.meshes[node.mesh_idx].triangles
		if len(tris) == 0 {
			continue
		}
		override := i32(-1)
		if node.material_override_idx >= 0 && int(node.material_override_idx) < len(scene.materials) {
			override = node.material_override_idx
		}
		key := Geometry_Key{raw_data(tris), len(tris), override}
		gi, found := group_of[key]
		if !found {
			gi = len(groups)
			group_of[key] = gi
			append(&groups, Geometry{triangles = tris, override = override})
		}
		append(&groups[gi].ids, i32(ni))
	}

	// Spheres are analytic for the path tracer and must be tessellated here,
	// already in world space. `build_icosphere` is the same one the GPU cache
	// uses. They are not scene nodes, so their ids continue past the node range
	// rather than colliding with node 0, and their materials continue past the
	// scene's.
	sphere_tris := make([dynamic][]lc.Triangle)
	defer {
		for t in sphere_tris {
			delete(t)
		}
		delete(sphere_tris)
	}
	for sphere, si in scene.spheres {
		t := lc.build_icosphere(sphere.center, sphere.radius, sphere.material)
		append(&sphere_tris, t)
		g := Geometry{triangles = t, override = i32(len(scene.materials) + si), sphere = true}
		append(&g.ids, i32(len(scene.nodes) + si))
		append(&groups, g)
	}

	index_total, instance_total := 0, 0
	for g in groups {
		index_total += len(g.triangles) * 3
		instance_total += len(g.ids)
	}
	if index_total == 0 {
		fmt.eprintln("realtime: no geometry to draw")
		return {}, false
	}

	// Welding leaves fewer vertices than corners; how many fewer is not known
	// until it is done.
	verts := make([dynamic]Vertex, 0, index_total / 2)
	cpu.indices = make([]u32, index_total)
	weld: [dynamic]u32
	defer delete(weld)
	cpu.instances = make([]Instance_GPU, instance_total)
	cpu.instance_bounds = make([]Instance_Bounds, instance_total)
	batches := make([dynamic]Draw_Batch)
	cpu.bounds_min = {max(f32), max(f32), max(f32)}
	cpu.bounds_max = {min(f32), min(f32), min(f32)}

	order := make([dynamic]Sort_Key)
	defer delete(order)

	ix, inst := 0, 0
	for g in groups {
		// Sort by material so each material's triangles form one contiguous
		// run. Stable, so geometry order within a material stays as authored,
		// which keeps z-fighting on coplanar faces consistent between runs.
		//
		// The key carries the material so the comparator needs no context --
		// Odin procedure literals capture nothing.
		clear(&order)
		for tri, i in g.triangles {
			append(&order, Sort_Key{mat_idx = resolve_material(tri.mat_idx, g.override, len(scene.materials)), index = i32(i)})
		}
		slice.stable_sort_by(order[:], proc(a, b: Sort_Key) -> bool {
			return a.mat_idx < b.mat_idx
		})

		// Weld within this mesh only: indices are local to it, and a corner
		// shared with a different mesh is a coincidence, not topology.
		first_vertex := len(verts)
		first_index := ix
		weld_reset(&weld, len(g.triangles) * 3)
		for key in order {
			tri := g.triangles[key.index]
			tangent := NO_TANGENT
			if batch_material(scene, key.mat_idx).normal_tex.has_data {
				tangent = triangle_tangent(tri)
			}
			positions := [3]lc.Vec3{tri.v0, tri.v1, tri.v2}
			normals := [3]lc.Vec3{tri.n0, tri.n1, tri.n2}
			uvs := [3]lc.Vec3{tri.uv0, tri.uv1, tri.uv2}
			for k in 0 ..< 3 {
				vert := Vertex {
					pos     = vec3f(positions[k]),
					normal  = vec3f(normals[k]),
					uv      = tri.has_uv ? [2]f32{f32(uvs[k].x), f32(uvs[k].y)} : {0, 0},
					tangent = tangent,
				}
				cpu.indices[ix] = weld_insert(weld[:], &verts, first_vertex, vert)
				ix += 1
			}
		}
		local := verts[first_vertex:]

		first_instance := inst
		for id in g.ids {
			world := g.sphere ? m.mat4(1) : scene.nodes[id].world_transform
			cpu.instances[inst] = instance_record(world, u32(id))

			// Exact world bounds, from every vertex: the cascades fit to the
			// scene bounds, and a box grown from a rotated local box would
			// loosen them.
			lo := [3]f32{max(f32), max(f32), max(f32)}
			hi := [3]f32{min(f32), min(f32), min(f32)}
			for p in local {
				w := world * [4]f32{p.pos.x, p.pos.y, p.pos.z, 1}
				for k in 0 ..< 3 {
					lo[k] = min(lo[k], w[k])
					hi[k] = max(hi[k], w[k])
				}
			}
			cpu.instance_bounds[inst] = {lo, hi}
			for k in 0 ..< 3 {
				cpu.bounds_min[k] = min(cpu.bounds_min[k], lo[k])
				cpu.bounds_max[k] = max(cpu.bounds_max[k], hi[k])
			}
			inst += 1
		}

		run_start := 0
		for i := 1; i <= len(order); i += 1 {
			if i < len(order) && order[i].mat_idx == order[run_start].mat_idx {
				continue
			}
			mat_idx := order[run_start].mat_idx
			mat := batch_material(scene, mat_idx)
			append(&batches, Draw_Batch {
				first_index    = u32(first_index + run_start * 3),
				index_count    = u32((i - run_start) * 3),
				base_vertex    = i32(first_vertex),
				first_instance = u32(first_instance),
				instance_count = u32(len(g.ids)),
				material_index = mat_idx,
				material       = material_uniforms(mat),
				transparent    = mat.spec_trans >= TRANSPARENT_THRESHOLD,
			})
			run_start = i
		}
	}

	cpu.batches = batches[:]
	cpu.verts = verts[:]
	return cpu, true
}

// An open-addressed table from vertex to its index within one mesh. Slots hold
// index + 1, so zero is empty. A `map[Vertex]u32` did the same job at a quarter
// of the speed, which on a 26-million-corner scene is ten seconds of upload.
@(private = "file")
weld_reset :: proc(table: ^[dynamic]u32, corners: int) {
	// At most half full, so probe runs stay short.
	size := 16
	for size < corners * 2 {
		size *= 2
	}
	resize(table, size)
	slice.zero(table[:])
}

// The index of `v` within the mesh starting at `first_vertex`, appending it to
// `verts` first if it is new. Exact bitwise equality: welding two corners that
// differ in any attribute would change what is drawn.
@(private = "file")
weld_insert :: proc(table: []u32, verts: ^[dynamic]Vertex, first_vertex: int, v: Vertex) -> u32 {
	v := v
	mask := len(table) - 1
	slot := int(xxhash.XXH3_64_default(slice.bytes_from_ptr(&v, size_of(Vertex)))) & mask
	for {
		entry := table[slot]
		if entry == 0 {
			local := u32(len(verts) - first_vertex)
			table[slot] = local + 1
			append(verts, v)
			return local
		}
		if verts[first_vertex + int(entry - 1)] == v {
			return entry - 1
		}
		slot = (slot + 1) & mask
	}
}

// A triangle's material as the path tracer's flattening resolves it: the
// node's override, else the triangle's own index, else material 0.
@(private = "file")
resolve_material :: proc(tri_mat, override: i32, material_count: int) -> i32 {
	if override >= 0 {
		return override
	}
	if tri_mat >= 0 && int(tri_mat) < material_count {
		return tri_mat
	}
	return 0
}

// The material a batch index names: the scene's, then its spheres'.
@(private = "file")
batch_material :: proc(scene: ^lc.Scene, index: i32) -> lc.Material {
	if index >= 0 && int(index) < len(scene.materials) {
		return scene.materials[index]
	}
	if si := int(index) - len(scene.materials); si >= 0 && si < len(scene.spheres) {
		return scene.spheres[si].material
	}
	return {}
}

@(private = "file")
instance_record :: proc(world: m.mat4, id: u32) -> Instance_GPU {
	rec := Instance_GPU{id = id}
	for c in 0 ..< 4 {
		for r in 0 ..< 4 {
			rec.world[c][r] = world[r, c]
		}
	}
	linear := m.mat3(world)
	normal := m.mat3(1)
	// A degenerate transform flattens the mesh to nothing visible; any normal
	// matrix will do rather than one full of infinities.
	if abs(m.determinant(linear)) > 1e-30 {
		normal = m.transpose(m.inverse(linear))
	}
	for c in 0 ..< 3 {
		for r in 0 ..< 3 {
			rec.normal[c][r] = normal[r, c]
		}
	}
	return rec
}

// Re-reads material values into the batches, leaving geometry and textures
// alone. Materials reach the shader as per-draw uniforms rather than a buffer,
// so an edit costs this loop and no upload at all — the same reasoning behind
// the path tracer updating its material buffer in place instead of rebuilding
// the cache.
batches_refresh_materials :: proc(batches: []Draw_Batch, mats: []lc.Material) {
	for &b in batches {
		if b.material_index >= 0 && int(b.material_index) < len(mats) {
			b.material = material_uniforms(mats[b.material_index])
			b.transparent = mats[b.material_index].spec_trans >= TRANSPARENT_THRESHOLD
		}
	}
}

scene_free_cpu :: proc(cpu: ^Scene_CPU) {
	delete(cpu.batches)
	delete(cpu.verts)
	delete(cpu.indices)
	delete(cpu.instances)
	delete(cpu.instance_bounds)
	cpu^ = {}
}

scene_upload :: proc(gpu: ^sdl.GPUDevice, scene: ^lc.Scene, key: u64, cache: ^Texture_Cache) -> (result: Scene_GPU, ok: bool) {
	s: Scene_GPU
	defer if !ok { scene_destroy(gpu, &s) }
	cpu, built := scene_build_cpu(scene)
	if !built {
		return {}, false
	}
	defer delete(cpu.verts)
	defer delete(cpu.indices)
	defer delete(cpu.instances)

	// Batches and bounds move into the scene, which frees them.
	s.key = key
	s.batches = cpu.batches
	s.instance_bounds = cpu.instance_bounds
	s.bounds_min = cpu.bounds_min
	s.bounds_max = cpu.bounds_max

	s.white = make_solid_texture(gpu, {255, 255, 255, 255}, srgb = false) or_return
	s.flat_normal = make_solid_texture(gpu, {128, 128, 255, 255}, srgb = false) or_return
	if !scene_bind_textures(gpu, &s, scene, cache) {
		return {}, false
	}

	s.vertices = make_gpu_buffer(gpu, slice.to_bytes(cpu.verts), {.VERTEX})
	if s.vertices == nil {
		return {}, false
	}
	s.vertex_count = u32(len(cpu.verts))
	s.indices = make_gpu_buffer(gpu, slice.to_bytes(cpu.indices), {.INDEX})
	if s.indices == nil {
		return {}, false
	}
	s.index_count = u32(len(cpu.indices))
	s.instances = make_gpu_buffer(gpu, slice.to_bytes(cpu.instances), {.VERTEX})
	if s.instances == nil {
		return {}, false
	}
	s.instance_count = u32(len(cpu.instances))

	s.sampler = sdl.CreateGPUSampler(
		gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .LINEAR,
			mag_filter = .LINEAR,
			mipmap_mode = .LINEAR,
			address_mode_u = .REPEAT,
			address_mode_v = .REPEAT,
			address_mode_w = .REPEAT,
			// Textures come from film and photogrammetry sources seen at
			// grazing angles; trilinear alone reads as a smeared floor.
			enable_anisotropy = true,
			max_anisotropy = 8,
			max_lod = 1000,
		},
	)
	if s.sampler == nil {
		fmt.eprintln("realtime: CreateGPUSampler failed:", sdl.GetError())
		return {}, false
	}

	fmt.printfln(
		"Realtime scene: %d vertices (%d indices), %d instances, %d draw batches, %d textures",
		s.vertex_count, s.index_count, s.instance_count, len(s.batches), len(cache^),
	)
	return s, true
}

// Replaces instance transforms and bounds after a placement-only edit, leaving
// vertices, batches and textures alone. The caller has checked the geometry
// revision is unchanged, so the rebuilt batches match the uploaded ones.
// Returns false when they do not, and the caller uploads the scene instead.
//
// The CPU build is rerun whole: it is the one place that computes exact world
// bounds, and it costs far less than the vertex upload it avoids.
scene_update_instances :: proc(gpu: ^sdl.GPUDevice, s: ^Scene_GPU, scene: ^lc.Scene) -> bool {
	cpu, built := scene_build_cpu(scene)
	if !built {
		return false
	}
	defer scene_free_cpu(&cpu)
	if len(cpu.instances) != int(s.instance_count) || len(cpu.batches) != len(s.batches) {
		return false
	}
	if !upload_bytes(gpu, s.instances, slice.to_bytes(cpu.instances)) {
		return false
	}
	copy(s.instance_bounds, cpu.instance_bounds)
	s.bounds_min = cpu.bounds_min
	s.bounds_max = cpu.bounds_max
	return true
}

scene_destroy :: proc(gpu: ^sdl.GPUDevice, s: ^Scene_GPU) {
	if gpu == nil {
		return
	}
	if s.vertices != nil {
		sdl.ReleaseGPUBuffer(gpu, s.vertices)
	}
	if s.indices != nil {
		sdl.ReleaseGPUBuffer(gpu, s.indices)
	}
	if s.instances != nil {
		sdl.ReleaseGPUBuffer(gpu, s.instances)
	}
	if s.sampler != nil {
		sdl.ReleaseGPUSampler(gpu, s.sampler)
	}
	if s.white != nil {
		sdl.ReleaseGPUTexture(gpu, s.white)
	}
	if s.flat_normal != nil {
		sdl.ReleaseGPUTexture(gpu, s.flat_normal)
	}
	delete(s.batches)
	delete(s.instance_bounds)
	delete(s.visible)
	s^ = {}
}

// ── helpers ──────────────────────────────────────────────────────────────────

@(private = "file")
make_gpu_buffer :: proc(gpu: ^sdl.GPUDevice, data: []u8, usage: sdl.GPUBufferUsageFlags) -> ^sdl.GPUBuffer {
	buffer := sdl.CreateGPUBuffer(gpu, sdl.GPUBufferCreateInfo{usage = usage, size = u32(len(data))})
	if buffer == nil {
		fmt.eprintln("realtime: CreateGPUBuffer failed:", sdl.GetError())
		return nil
	}
	if !upload_bytes(gpu, buffer, data) {
		sdl.ReleaseGPUBuffer(gpu, buffer)
		return nil
	}
	return buffer
}

material_uniforms :: proc(mat: lc.Material) -> Material_Uniforms {
	// Emission mirrors the path tracer's two distinct paths, which do NOT use
	// the same inputs (shaders/raytrace.metal):
	//
	//   - An emissive MAP modulates `emission` alone, with no strength factor.
	//     glTF leaves `emission_strength` at zero for these, so folding it in
	//     multiplies the whole map away -- which is exactly what happened to
	//     the damaged helmet's HUD graphics.
	//   - A pure emitter (`.Emissive`) uses `emissive_radiance`: emission, or
	//     albedo when emission is black, times strength, which defaults to 20
	//     rather than to zero.
	//
	// Any other material emits nothing, map or not.
	emission_rgb: [3]f32
	if mat.emissive_tex.has_data {
		emission_rgb = {f32(mat.emission.x), f32(mat.emission.y), f32(mat.emission.z)}
	} else if mat.kind == .Emissive {
		color := mat.emission
		if color.x + color.y + color.z <= 0 {
			color = mat.albedo
		}
		strength := mat.emission_strength
		if strength <= 0 {
			strength = 20.0
		}
		emission_rgb = {
			f32(color.x * strength), f32(color.y * strength), f32(color.z * strength),
		}
	}

	return Material_Uniforms {
		base_color = {
			f32(mat.albedo.x), f32(mat.albedo.y), f32(mat.albedo.z),
			mat.albedo_tex.has_data ? 1 : 0,
		},
		emission = {
			emission_rgb.x, emission_rgb.y, emission_rgb.z,
			mat.emissive_tex.has_data ? 1 : 0,
		},
		params = {
			f32(mat.roughness), f32(mat.metallic), f32(mat.specular), f32(mat.normal_scale),
		},
		flags = {
			mat.metallic_roughness_tex.has_data ? 1 : 0,
			mat.normal_tex.has_data ? 1 : 0,
			f32(clamp(mat.spec_trans, 0, 1)),
			f32(max(mat.ir, 1)),
		},
	}
}

// Per-face tangent from the UV gradient, the same basis the path tracer's
// `perturb_normal` derives per hit. Not averaged across faces, even though the
// mesh is now welded: smoothing it would make the two renderers disagree about
// every normal-mapped surface.
@(private = "file")
triangle_tangent :: proc(tri: lc.Triangle) -> [4]f32 {
	e1 := vec3f(tri.v1 - tri.v0)
	e2 := vec3f(tri.v2 - tri.v0)

	if tri.has_uv {
		du1 := f32(tri.uv1.x - tri.uv0.x)
		dv1 := f32(tri.uv1.y - tri.uv0.y)
		du2 := f32(tri.uv2.x - tri.uv0.x)
		dv2 := f32(tri.uv2.y - tri.uv0.y)

		det := du1 * dv2 - du2 * dv1
		if abs(det) > 1e-12 {
			r := 1.0 / det
			t := [3]f32{
				(e1.x * dv2 - e2.x * dv1) * r,
				(e1.y * dv2 - e2.y * dv1) * r,
				(e1.z * dv2 - e2.z * dv1) * r,
			}
			t = normalize3(t)
			return {t.x, t.y, t.z, 1}
		}
	}

	// Degenerate or missing UVs: any vector in the triangle's plane will do,
	// since without UVs there is no normal map to orient.
	t := normalize3(e1)
	return {t.x, t.y, t.z, 1}
}

// Uploads a map, reusing the texture when several materials point at the same
// pixels. Returns `fallback` when the material has no such map, so every batch
// binds a full set and the pipeline's sampler count stays fixed.
// sRGB is expressed in the texture FORMAT rather than decoded in the shader.
// The path tracer has to decode by hand because a compute kernel reads raw
// bytes; here the sampler does it for free, and does it before filtering, which
// is also the correct order.
@(private)
upload_texture :: proc(gpu: ^sdl.GPUDevice, tex: lc.TextureMap) -> ^sdl.GPUTexture {
	levels := mip_levels(tex.width, tex.height)

	texture := sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = .D2,
			format = tex.srgb ? .R8G8B8A8_UNORM_SRGB : .R8G8B8A8_UNORM,
			// COLOR_TARGET is required by SDL to generate mips into the chain.
			usage = {.SAMPLER, .COLOR_TARGET},
			width = u32(tex.width),
			height = u32(tex.height),
			layer_count_or_depth = 1,
			num_levels = levels,
			sample_count = ._1,
		},
	)
	if texture == nil {
		fmt.eprintln("realtime: CreateGPUTexture failed:", sdl.GetError())
		return nil
	}

	transfer := sdl.CreateGPUTransferBuffer(
		gpu,
		sdl.GPUTransferBufferCreateInfo{usage = .UPLOAD, size = u32(len(tex.pixels))},
	)
	if transfer == nil {
		fmt.eprintln("realtime: texture transfer buffer failed:", sdl.GetError())
		sdl.ReleaseGPUTexture(gpu, texture)
		return nil
	}
	defer sdl.ReleaseGPUTransferBuffer(gpu, transfer)

	dst := sdl.MapGPUTransferBuffer(gpu, transfer, false)
	if dst == nil {
		sdl.ReleaseGPUTexture(gpu, texture)
		return nil
	}
	copy(([^]u8)(dst)[:len(tex.pixels)], tex.pixels)
	sdl.UnmapGPUTransferBuffer(gpu, transfer)

	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	if cmd == nil {
		sdl.ReleaseGPUTexture(gpu, texture)
		return nil
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUTexture(
		pass,
		sdl.GPUTextureTransferInfo {
			transfer_buffer = transfer,
			pixels_per_row = u32(tex.width),
			rows_per_layer = u32(tex.height),
		},
		sdl.GPUTextureRegion {
			texture = texture,
			w = u32(tex.width),
			h = u32(tex.height),
			d = 1,
		},
		false,
	)
	sdl.EndGPUCopyPass(pass)
	if levels > 1 {
		sdl.GenerateMipmapsForGPUTexture(cmd, texture)
	}
	_ = sdl.SubmitGPUCommandBuffer(cmd)

	return texture
}

@(private = "file")
make_solid_texture :: proc(gpu: ^sdl.GPUDevice, rgba: [4]u8, srgb: bool) -> (^sdl.GPUTexture, bool) {
	pixels := rgba
	tex := lc.TextureMap {
		width = 1,
		height = 1,
		pixels = pixels[:],
		has_data = true,
		srgb = srgb,
	}
	texture := upload_texture(gpu, tex)
	return texture, texture != nil
}

mip_levels :: proc(width, height: i32) -> u32 {
	levels: u32 = 1
	size := max(width, height)
	for size > 1 {
		size /= 2
		levels += 1
	}
	return levels
}

upload_bytes :: proc(gpu: ^sdl.GPUDevice, buffer: ^sdl.GPUBuffer, data: []u8) -> bool {
	transfer := sdl.CreateGPUTransferBuffer(
		gpu,
		sdl.GPUTransferBufferCreateInfo{usage = .UPLOAD, size = u32(len(data))},
	)
	if transfer == nil {
		fmt.eprintln("realtime: CreateGPUTransferBuffer failed:", sdl.GetError())
		return false
	}
	defer sdl.ReleaseGPUTransferBuffer(gpu, transfer)

	dst := sdl.MapGPUTransferBuffer(gpu, transfer, false)
	if dst == nil {
		return false
	}
	copy(([^]u8)(dst)[:len(data)], data)
	sdl.UnmapGPUTransferBuffer(gpu, transfer)

	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	if cmd == nil {
		return false
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUBuffer(
		pass,
		sdl.GPUTransferBufferLocation{transfer_buffer = transfer},
		sdl.GPUBufferRegion{buffer = buffer, size = u32(len(data))},
		false,
	)
	sdl.EndGPUCopyPass(pass)
	return bool(sdl.SubmitGPUCommandBuffer(cmd))
}

vec3f :: proc(v: lc.Vec3) -> [3]f32 {
	return {f32(v.x), f32(v.y), f32(v.z)}
}

normalize3 :: proc(v: [3]f32) -> [3]f32 {
	l := v.x * v.x + v.y * v.y + v.z * v.z
	if l <= 1e-20 {
		return {0, 0, 1}
	}
	inv := 1.0 / math.sqrt(l)
	return {v.x * inv, v.y * inv, v.z * inv}
}
