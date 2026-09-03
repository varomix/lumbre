package lumbre_realtime

// Turns a `core.Scene` into buffers a rasterizer can draw.
//
// The path tracer's GPU cache (core/gpu_scene_cache_darwin.odin) already
// flattens the scene graph to a world-space triangle soup, and this mirrors
// that flattening rather than inventing a second one — same
// `flatten_scene_graph`, same icosphere tessellation for spheres, so both modes
// draw the same geometry. What it does NOT share is the layout, because the two
// renderers want opposite things:
//
//   - The path tracer wants one flat buffer it can index by triangle id from
//     anywhere in a kernel. Material is a per-triangle lookup.
//   - The rasterizer wants triangles GROUPED BY MATERIAL, because binding a
//     pipeline's textures is per-draw. A soup in arbitrary order would mean one
//     draw call per triangle.
//
// So triangles are sorted by material and emitted as contiguous runs, one draw
// per material. Vertices are not shared between triangles (they carry per-face
// data and the source `Triangle` is already unshared), so there is no index
// buffer: a draw is a vertex range, which is the honest shape of this data.
//
// Textures are the other divergence. The path tracer concatenates every map
// into one mip-less byte buffer indexed by offset, because a compute kernel has
// no sampler. Here each map becomes a real `sdl.GPUTexture` with a generated
// mip chain, sampled by hardware — which is most of why the raster mode can
// afford to be cheap.

import "core:fmt"
import "core:math"
import "core:slice"

import lc "../core"
import sdl "vendor:sdl3"

// One vertex, interleaved. 48 bytes; the layout is mirrored by
// `VERTEX_ATTRIBUTES` below and by the shader's input struct.
Vertex :: struct {
	pos:     [3]f32,
	normal:  [3]f32,
	uv:      [2]f32,
	// xyz = tangent, w = bitangent handedness. Per-face, since nothing is
	// shared; a normal map needs no more than that.
	tangent: [4]f32,
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
	// has_metallic_roughness_map, has_normal_map, unused, unused
	flags:      [4]f32,
}

// One draw: a contiguous vertex range sharing a material.
Draw_Batch :: struct {
	first_vertex: u32,
	vertex_count: u32,
	// Index into the flattened material list. Carried so a batch can be traced
	// back to its material when a render looks wrong.
	material_index: i32,
	material:     Material_Uniforms,
	// Borrowed from `Scene_GPU.textures`, or from its fallbacks. Never nil:
	// SDL binds a fixed number of samplers per pipeline, so an absent map is a
	// 1x1 default rather than a hole.
	albedo:       ^sdl.GPUTexture,
	mr:           ^sdl.GPUTexture,
	normal:       ^sdl.GPUTexture,
	emissive:     ^sdl.GPUTexture,
}

Scene_GPU :: struct {
	// Matches the IPR's `scene_key`. Bumped only when the scene itself
	// changes, so navigating never rebuilds any of this.
	key:           u64,
	vertices:      ^sdl.GPUBuffer,
	vertex_count:  u32,
	batches:       []Draw_Batch,

	// Owned. `textures` holds one entry per distinct map in the scene.
	textures:      []^sdl.GPUTexture,
	sampler:       ^sdl.GPUSampler,
	white:         ^sdl.GPUTexture, // stands in for a missing colour map
	flat_normal:   ^sdl.GPUTexture, // stands in for a missing normal map

	// World-space bounds, kept for the shadow cascades and for framing.
	bounds_min:    [3]f32,
	bounds_max:    [3]f32,
}

VERTEX_ATTRIBUTES := [4]sdl.GPUVertexAttribute {
	{location = 0, buffer_slot = 0, format = .FLOAT3, offset = u32(offset_of(Vertex, pos))},
	{location = 1, buffer_slot = 0, format = .FLOAT3, offset = u32(offset_of(Vertex, normal))},
	{location = 2, buffer_slot = 0, format = .FLOAT2, offset = u32(offset_of(Vertex, uv))},
	{location = 3, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(Vertex, tangent))},
}

VERTEX_BUFFER_DESCRIPTION := [1]sdl.GPUVertexBufferDescription {
	{slot = 0, pitch = u32(size_of(Vertex)), input_rate = .VERTEX},
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

// The CPU half: flatten, sort by material, build vertices and batch ranges.
// Split out from the GPU upload so it can be tested without a device — the
// batching is where the bugs that survive a screenshot live.
//
// The returned batches have no textures bound yet; `scene_upload` fills those
// in. Free with `scene_free_cpu`.
scene_build_cpu :: proc(
	scene: ^lc.Scene,
) -> (
	batches: []Draw_Batch,
	verts: []Vertex,
	bounds_min: [3]f32,
	bounds_max: [3]f32,
	ok: bool,
) {
	// Same flattening the path tracer does, so both modes draw identical
	// geometry rather than two interpretations of the scene graph.
	flat := lc.flatten_scene_graph(scene)
	defer lc.destroy_flattened_scene(flat)

	tris := make([dynamic]lc.Triangle)
	defer delete(tris)
	mats := make([dynamic]lc.Material)
	defer delete(mats)

	append(&tris, ..flat.triangles)
	append(&mats, ..flat.materials)

	// Spheres are analytic for the path tracer and must be tessellated here.
	// `build_icosphere` is the same one the GPU cache uses.
	for sphere in scene.spheres {
		sphere_tris := lc.build_icosphere(sphere.center, sphere.radius, sphere.material)
		defer delete(sphere_tris)
		mat_idx := i32(len(mats))
		for t in sphere_tris {
			tri := t
			tri.mat_idx = mat_idx
			append(&tris, tri)
		}
		append(&mats, sphere.material)
	}

	if len(tris) == 0 {
		fmt.eprintln("realtime: no geometry to draw")
		return nil, nil, {}, {}, false
	}

	// Sort by material so each material's triangles form one contiguous run.
	// Stable, so geometry order within a material stays as authored, which
	// keeps z-fighting on coplanar faces consistent between runs.
	slice.stable_sort_by(tris[:], proc(a, b: lc.Triangle) -> bool {
		return a.mat_idx < b.mat_idx
	})

	verts = make([]Vertex, len(tris) * 3)

	bounds_min = {max(f32), max(f32), max(f32)}
	bounds_max = {min(f32), min(f32), min(f32)}

	for tri, i in tris {
		tangent := triangle_tangent(tri)
		positions := [3]lc.Vec3{tri.v0, tri.v1, tri.v2}
		normals := [3]lc.Vec3{tri.n0, tri.n1, tri.n2}
		uvs := [3]lc.Vec3{tri.uv0, tri.uv1, tri.uv2}

		for k in 0 ..< 3 {
			p := vec3f(positions[k])
			verts[i * 3 + k] = Vertex {
				pos     = p,
				normal  = vec3f(normals[k]),
				uv      = tri.has_uv ? [2]f32{f32(uvs[k].x), f32(uvs[k].y)} : {0, 0},
				tangent = tangent,
			}
			bounds_min = {min(bounds_min.x, p.x), min(bounds_min.y, p.y), min(bounds_min.z, p.z)}
			bounds_max = {max(bounds_max.x, p.x), max(bounds_max.y, p.y), max(bounds_max.z, p.z)}
		}
	}

	// One batch per run of equal material index.
	out := make([dynamic]Draw_Batch)
	run_start := 0
	for i := 1; i <= len(tris); i += 1 {
		if i < len(tris) && tris[i].mat_idx == tris[run_start].mat_idx {
			continue
		}

		mat_idx := tris[run_start].mat_idx
		mat: lc.Material
		if mat_idx >= 0 && int(mat_idx) < len(mats) {
			mat = mats[mat_idx]
		}

		append(&out, Draw_Batch {
			first_vertex   = u32(run_start * 3),
			vertex_count   = u32((i - run_start) * 3),
			material_index = mat_idx,
			material       = material_uniforms(mat),
		})
		run_start = i
	}

	return out[:], verts, bounds_min, bounds_max, true
}

scene_free_cpu :: proc(batches: []Draw_Batch, verts: []Vertex) {
	delete(batches)
	delete(verts)
}

scene_upload :: proc(gpu: ^sdl.GPUDevice, scene: ^lc.Scene, key: u64) -> (s: Scene_GPU, ok: bool) {
	batches, verts, lo, hi, built := scene_build_cpu(scene)
	if !built {
		return {}, false
	}
	defer delete(verts)

	s.key = key
	s.batches = batches
	s.bounds_min = lo
	s.bounds_max = hi

	// Re-flatten only to recover the material list the batches index into.
	// Cheap next to the geometry work, and it keeps `scene_build_cpu` free of
	// GPU-shaped return values.
	flat := lc.flatten_scene_graph(scene)
	defer lc.destroy_flattened_scene(flat)
	mats := make([dynamic]lc.Material)
	defer delete(mats)
	append(&mats, ..flat.materials)
	for sphere in scene.spheres {
		append(&mats, sphere.material)
	}

	s.white = make_solid_texture(gpu, {255, 255, 255, 255}, srgb = false) or_return
	// A tangent-space normal of (0,0,1) encodes as (128,128,255).
	s.flat_normal = make_solid_texture(gpu, {128, 128, 255, 255}, srgb = false) or_return

	tex_cache := make(map[rawptr]^sdl.GPUTexture)
	defer delete(tex_cache)
	owned := make([dynamic]^sdl.GPUTexture)

	for &b in s.batches {
		mat: lc.Material
		if b.material_index >= 0 && int(b.material_index) < len(mats) {
			mat = mats[b.material_index]
		}
		b.albedo = upload_map(gpu, mat.albedo_tex, &tex_cache, &owned, s.white)
		b.mr = upload_map(gpu, mat.metallic_roughness_tex, &tex_cache, &owned, s.white)
		b.normal = upload_map(gpu, mat.normal_tex, &tex_cache, &owned, s.flat_normal)
		b.emissive = upload_map(gpu, mat.emissive_tex, &tex_cache, &owned, s.white)
	}
	s.textures = owned[:]

	byte_size := u32(len(verts) * size_of(Vertex))
	s.vertices = sdl.CreateGPUBuffer(gpu, sdl.GPUBufferCreateInfo{usage = {.VERTEX}, size = byte_size})
	if s.vertices == nil {
		fmt.eprintln("realtime: CreateGPUBuffer failed:", sdl.GetError())
		scene_destroy(gpu, &s)
		return {}, false
	}
	s.vertex_count = u32(len(verts))

	if !upload_bytes(gpu, s.vertices, slice.to_bytes(verts)) {
		scene_destroy(gpu, &s)
		return {}, false
	}

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
		scene_destroy(gpu, &s)
		return {}, false
	}

	fmt.printfln(
		"Realtime scene: %d vertices, %d draw batches, %d textures",
		s.vertex_count, len(s.batches), len(s.textures),
	)
	return s, true
}

scene_destroy :: proc(gpu: ^sdl.GPUDevice, s: ^Scene_GPU) {
	if gpu == nil {
		return
	}
	if s.vertices != nil {
		sdl.ReleaseGPUBuffer(gpu, s.vertices)
	}
	if s.sampler != nil {
		sdl.ReleaseGPUSampler(gpu, s.sampler)
	}
	for tex in s.textures {
		sdl.ReleaseGPUTexture(gpu, tex)
	}
	if s.white != nil {
		sdl.ReleaseGPUTexture(gpu, s.white)
	}
	if s.flat_normal != nil {
		sdl.ReleaseGPUTexture(gpu, s.flat_normal)
	}
	delete(s.textures)
	delete(s.batches)
	s^ = {}
}

// ── helpers ──────────────────────────────────────────────────────────────────

@(private = "file")
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
			0, 0,
		},
	}
}

// Per-face tangent from the UV gradient. Vertices are unshared, so there is
// nothing to average across faces — a smoothed tangent basis would need a
// welded mesh, which this geometry is not.
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
@(private = "file")
upload_map :: proc(
	gpu: ^sdl.GPUDevice,
	tex: lc.TextureMap,
	cache: ^map[rawptr]^sdl.GPUTexture,
	owned: ^[dynamic]^sdl.GPUTexture,
	fallback: ^sdl.GPUTexture,
) -> ^sdl.GPUTexture {
	if !tex.has_data || len(tex.pixels) == 0 || tex.width <= 0 || tex.height <= 0 {
		return fallback
	}

	pixels_key := rawptr(raw_data(tex.pixels))
	if existing, found := cache[pixels_key]; found {
		return existing
	}

	uploaded := upload_texture(gpu, tex)
	if uploaded == nil {
		return fallback
	}
	cache[pixels_key] = uploaded
	append(owned, uploaded)
	return uploaded
}

// sRGB is expressed in the texture FORMAT rather than decoded in the shader.
// The path tracer has to decode by hand because a compute kernel reads raw
// bytes; here the sampler does it for free, and does it before filtering, which
// is also the correct order.
@(private = "file")
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
	_ = sdl.SubmitGPUCommandBuffer(cmd)
	return true
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
