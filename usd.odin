package main

import "core:c"
import "core:fmt"
import "core:strings"
import m "core:math/linalg/glsl"

foreign import usd_shim "lib/darwin/libusd_shim.dylib"

Usd_Interp :: enum i32 {
	None         = 0,
	Vertex       = 1,
	Face_Varying = 2,
	Uniform      = 3,
}

Usd_Shim_Mesh_Data :: struct {
	points:               [^]f32,
	point_count:          c.int,
	face_vertex_indices:  [^]c.int,
	index_count:          c.int,
	face_vertex_counts:   [^]c.int,
	face_count:           c.int,
	normals:              [^]f32,
	normal_count:         c.int,
	normal_interp:        Usd_Interp,
	uvs:                  [^]f32,
	uv_count:             c.int,
	uv_interp:            Usd_Interp,
}

Usd_Shim_Material_Data :: struct {
	base_color:          [3]f32,
	has_base_color_tex:  c.int,
	base_color_tex:      [1024]u8,
	roughness:           f32,
	has_roughness_tex:   c.int,
	roughness_tex:       [1024]u8,
	metallic:            f32,
	has_metallic_tex:    c.int,
	metallic_tex:        [1024]u8,
	opacity:             f32,
	emissive_color:      [3]f32,
	has_emissive_tex:    c.int,
	emissive_tex:        [1024]u8,
	has_normal_tex:      c.int,
	normal_tex:          [1024]u8,
}

// A "materialBind" face subset: a set of faces on the mesh carrying their
// own material. How a DCC gives one mesh several texture sets; such a mesh
// usually has no material bound at the mesh level.
Usd_Shim_Subset_Data :: struct {
	face_indices:     [^]c.int,
	face_index_count: c.int,
	has_material:     c.int,
	material:         Usd_Shim_Material_Data,
}

Usd_Shim_Stage :: distinct rawptr
Usd_Shim_Prim :: distinct rawptr

@(default_calling_convention = "c")
foreign usd_shim {
	usd_shim_open_flattened :: proc(path: cstring, err_buf: [^]u8, err_buf_len: c.int) -> Usd_Shim_Stage ---
	usd_shim_close :: proc(stage: Usd_Shim_Stage) ---
	usd_shim_get_pseudo_root :: proc(stage: Usd_Shim_Stage) -> Usd_Shim_Prim ---
	usd_shim_get_children :: proc(prim: Usd_Shim_Prim, out: [^]Usd_Shim_Prim, max: c.int) -> c.int ---
	usd_shim_prim_type_name :: proc(prim: Usd_Shim_Prim) -> cstring ---
	usd_shim_prim_name :: proc(prim: Usd_Shim_Prim) -> cstring ---
	usd_shim_get_local_transform :: proc(prim: Usd_Shim_Prim, out_mat4x4: [^]f64) -> c.int ---
	usd_shim_get_mesh_data :: proc(prim: Usd_Shim_Prim, out: ^Usd_Shim_Mesh_Data) -> c.int ---
	usd_shim_free_mesh_data :: proc(data: ^Usd_Shim_Mesh_Data) ---
	usd_shim_get_bound_material :: proc(prim: Usd_Shim_Prim, out: ^Usd_Shim_Material_Data) -> c.int ---
	usd_shim_resolve_asset_path :: proc(stage: Usd_Shim_Stage, asset_path: cstring) -> cstring ---
	usd_shim_read_asset :: proc(resolved_path: cstring, out_size: ^c.size_t) -> [^]u8 ---
	usd_shim_free_asset :: proc(data: [^]u8) ---
	usd_shim_get_subset_count :: proc(mesh: Usd_Shim_Prim) -> c.int ---
	usd_shim_get_subsets :: proc(mesh: Usd_Shim_Prim, out: [^]Usd_Shim_Subset_Data, max: c.int) -> c.int ---
	usd_shim_free_subsets :: proc(subsets: [^]Usd_Shim_Subset_Data, count: c.int) ---
}

usd_load_state :: struct {
	stage:       Usd_Shim_Stage,
	base_dir:    string,
	materials:   [dynamic]Material,
	// Cache from a resolved texture path to its index in `materials`'s
	// texture slots isn't needed here: USD materials are read per-mesh
	// (no shared material array like glTF's, since UsdShadeMaterial
	// binding is resolved per-prim). image_cache avoids reloading the
	// same texture file for meshes that share a bound material.
	image_cache: map[string]TextureMap,
}

// Loads a USD stage (.usd/.usda/.usdc/.usdz) into Lumbre's common ObjData
// intermediate, mirroring load_gltf/load_obj so it plugs into the existing
// build_default_scene_graph / flatten_scene_graph pipeline unchanged.
//
// Composition (references/layers/variants) is flattened up front via
// UsdStage::Flatten() in the shim -- v1 does not support live variant
// switching or unflattened composition arcs.
load_usd :: proc(path: string, allocator := context.allocator) -> (ObjData, bool) {
	cpath := strings.clone_to_cstring(path, allocator)
	defer delete(cpath, allocator)

	err_buf: [512]u8
	stage := usd_shim_open_flattened(cpath, &err_buf[0], c.int(len(err_buf)))
	if stage == nil {
		fmt.eprintln("usd: failed to open", path, "err=", cstring(&err_buf[0]))
		return {}, false
	}
	defer usd_shim_close(stage)

	state := usd_load_state{
		stage       = stage,
		image_cache = make(map[string]TextureMap),
	}
	if idx := strings.last_index(path, "/"); idx >= 0 {
		state.base_dir = strings.clone(path[:idx + 1], allocator)
	}
	defer {
		for _, &v in state.image_cache {
			destroy_texture(&v)
		}
		delete(state.image_cache)
		if state.base_dir != "" {
			delete(state.base_dir, allocator)
		}
	}

	all_meshes: [dynamic]Mesh
	root := usd_shim_get_pseudo_root(stage)
	usd_collect_meshes(root, m.mat4(1), &all_meshes, &state)

	live_count := 0
	for msh in all_meshes {
		if len(msh.triangles) > 0 {
			live_count += 1
		} else {
			delete(msh.triangles)
			if msh.name != "" { delete(msh.name) }
		}
	}

	result_meshes := make([]Mesh, live_count, allocator)
	write_idx := 0
	for msh in all_meshes {
		if len(msh.triangles) > 0 {
			result_meshes[write_idx] = msh
			write_idx += 1
		}
	}
	delete(all_meshes)

	result_mats := make([]Material, len(state.materials), allocator)
	for mat, i in state.materials {
		result_mats[i] = mat
	}
	delete(state.materials)

	fmt.println("usd: loaded", len(result_meshes), "meshes,", len(result_mats), "materials from", path)
	return ObjData{meshes = result_meshes, materials = result_mats}, true
}

usd_collect_meshes :: proc(
	prim: Usd_Shim_Prim,
	parent_transform: m.mat4,
	meshes: ^[dynamic]Mesh,
	state: ^usd_load_state,
) {
	local := m.mat4(1)
	mat4_raw: [16]f64
	if usd_shim_get_local_transform(prim, &mat4_raw[0]) != 0 {
		// GfMatrix4d is row-major with a row-vector convention (v' = v*M,
		// translation in row 3). Odin's mat4 is column-major with a
		// column-vector convention (v' = M*v). Converting the convention
		// transposes, and converting the storage transposes again, so the
		// two cancel: copy the 16 values straight across, exactly as the
		// glTF path does with cgltf's column-major matrices.
		dst := ([^]f32)(&local)[:16]
		for i in 0 ..< 16 {
			dst[i] = f32(mat4_raw[i])
		}
	}
	world := parent_transform * local

	type_name := string(usd_shim_prim_type_name(prim))
	if type_name == "Mesh" {
		usd_emit_mesh(prim, world, meshes, state)
	}

	children: [256]Usd_Shim_Prim
	n := int(usd_shim_get_children(prim, &children[0], 256))
	for i in 0 ..< min(n, 256) {
		usd_collect_meshes(children[i], world, meshes, state)
	}
}

usd_emit_mesh :: proc(
	prim: Usd_Shim_Prim,
	transform: m.mat4,
	meshes: ^[dynamic]Mesh,
	state: ^usd_load_state,
) {
	mesh_data: Usd_Shim_Mesh_Data
	if usd_shim_get_mesh_data(prim, &mesh_data) == 0 {
		return
	}
	defer usd_shim_free_mesh_data(&mesh_data)

	// Material bound to the mesh as a whole. -1 when the mesh carries its
	// materials on face subsets instead, which is how one mesh gets
	// several texture sets.
	mat_idx := usd_build_material(prim, state)

	name := string(usd_shim_prim_name(prim))

	triangles, tri_faces := usd_triangulate_mesh(mesh_data)
	defer delete(triangles)
	defer delete(tri_faces)
	if len(triangles) == 0 {
		return
	}

	// Per-face material table, defaulting to the mesh-level binding.
	face_mats := make([]i32, int(mesh_data.face_count))
	defer delete(face_mats)
	for i in 0 ..< len(face_mats) {
		face_mats[i] = mat_idx
	}
	usd_apply_subset_materials(prim, face_mats, state)

	for &t, i in triangles {
		face := tri_faces[i]
		t.mat_idx = face_mats[face] if face >= 0 && face < len(face_mats) else mat_idx
	}

	mm := Mesh{
		name      = strings.clone(name, context.allocator),
		transform = transform,
	}
	mm.triangles = make([]Triangle, len(triangles), context.allocator)
	copy(mm.triangles, triangles[:])
	// Mesh.material is only a representative value -- shading resolves per
	// triangle through mat_idx, which differs across faces when the mesh
	// has subsets. Take the first triangle's.
	repr := triangles[0].mat_idx
	mm.material = state.materials[repr] if repr >= 0 && int(repr) < len(state.materials) else Material{}
	append(meshes, mm)
}

// Fan-triangulates every polygon (faceVertexCounts may be 3+, quads/ngons
// are common from DCC exports, unlike glTF's implicit triangle lists).
// Correct for convex faces; concave ngons are a known, unaddressed v1 gap.
// Returns the triangles plus, for each, the index of the source face it was
// cut from. Callers need the face index to resolve per-face materials from
// GeomSubsets.
usd_triangulate_mesh :: proc(data: Usd_Shim_Mesh_Data) -> ([dynamic]Triangle, [dynamic]int) {
	tris: [dynamic]Triangle
	tri_faces: [dynamic]int

	points := data.points[:data.point_count * 3]
	indices := data.face_vertex_indices[:data.index_count]
	counts := data.face_vertex_counts[:data.face_count]

	normals: []f32
	if data.normal_count > 0 {
		normals = data.normals[:data.normal_count * 3]
	}
	uvs: []f32
	if data.uv_count > 0 {
		uvs = data.uvs[:data.uv_count * 2]
	}

	get_point :: proc(points: []f32, idx: int) -> Vec3 {
		o := idx * 3
		return Vec3{f64(points[o]), f64(points[o + 1]), f64(points[o + 2])}
	}

	// `corner` is the position within face_vertex_indices (i.e. the
	// faceVarying index); `vertex_idx` is what it points at in points[].
	get_normal :: proc(normals: []f32, interp: Usd_Interp, corner, face, vertex_idx: int) -> (Vec3, bool) {
		if len(normals) == 0 { return {}, false }
		idx := 0
		#partial switch interp {
		case .Face_Varying: idx = corner
		case .Uniform:      idx = face
		case .Vertex:       idx = vertex_idx
		case:               return {}, false
		}
		o := idx * 3
		if o + 2 >= len(normals) { return {}, false }
		return Vec3{f64(normals[o]), f64(normals[o + 1]), f64(normals[o + 2])}, true
	}

	// USD's `st` puts v=0 at the bottom of the image; glTF puts it at the
	// top. Lumbre's texture pipeline (stb's flip-on-load plus the sampler
	// in texture.odin) is calibrated for the glTF convention, so flip v
	// here. Verified on head_of_osiris, which ships as both .usdz and
	// .glb: for the same vertex, v_usd == 1 - v_gltf exactly.
	get_uv :: proc(uvs: []f32, interp: Usd_Interp, corner, face, vertex_idx: int) -> (Vec3, bool) {
		if len(uvs) == 0 { return {}, false }
		idx := 0
		#partial switch interp {
		case .Face_Varying: idx = corner
		case .Uniform:      idx = face
		case .Vertex:       idx = vertex_idx
		case:               return {}, false
		}
		o := idx * 2
		if o + 1 >= len(uvs) { return {}, false }
		return Vec3{f64(uvs[o]), 1.0 - f64(uvs[o + 1]), 0}, true
	}

	corner_start := 0
	for face_i in 0 ..< len(counts) {
		face_vert_count := int(counts[face_i])
		if face_vert_count < 3 {
			corner_start += face_vert_count
			continue
		}

		// Fan triangulation: (0,1,2), (0,2,3), (0,3,4), ...
		for k in 1 ..< face_vert_count - 1 {
			c0 := corner_start
			c1 := corner_start + k
			c2 := corner_start + k + 1

			vi0 := int(indices[c0])
			vi1 := int(indices[c1])
			vi2 := int(indices[c2])

			p0 := get_point(points, vi0)
			p1 := get_point(points, vi1)
			p2 := get_point(points, vi2)

			face_n := m.normalize(m.cross(p1 - p0, p2 - p0))

			n0, ok0 := get_normal(normals, data.normal_interp, c0, face_i, vi0)
			n1, ok1 := get_normal(normals, data.normal_interp, c1, face_i, vi1)
			n2, ok2 := get_normal(normals, data.normal_interp, c2, face_i, vi2)
			if !ok0 { n0 = face_n }
			if !ok1 { n1 = face_n }
			if !ok2 { n2 = face_n }

			uv0, uok0 := get_uv(uvs, data.uv_interp, c0, face_i, vi0)
			uv1, uok1 := get_uv(uvs, data.uv_interp, c1, face_i, vi1)
			uv2, uok2 := get_uv(uvs, data.uv_interp, c2, face_i, vi2)
			has_uv := uok0 && uok1 && uok2

			append(&tris, Triangle{
				v0 = p0, v1 = p1, v2 = p2,
				n0 = n0, n1 = n1, n2 = n2,
				uv0 = uv0, uv1 = uv1, uv2 = uv2,
				has_uv = b32(has_uv),
			})
			append(&tri_faces, face_i)
		}

		corner_start += face_vert_count
	}

	return tris, tri_faces
}

// Reads each materialBind face subset on `prim`, appends its material, and
// stamps the owning face indices into `face_mats`. A no-op for the common
// case of a mesh with a single whole-mesh binding.
usd_apply_subset_materials :: proc(prim: Usd_Shim_Prim, face_mats: []i32, state: ^usd_load_state) {
	count := int(usd_shim_get_subset_count(prim))
	if count <= 0 {
		return
	}

	subsets := make([]Usd_Shim_Subset_Data, count)
	defer delete(subsets)

	written := int(usd_shim_get_subsets(prim, raw_data(subsets), c.int(count)))
	defer usd_shim_free_subsets(raw_data(subsets), c.int(written))

	for i in 0 ..< written {
		subset := subsets[i]
		if subset.has_material == 0 {
			continue
		}
		mat_idx := usd_append_material(subset.material, state)
		for j in 0 ..< int(subset.face_index_count) {
			face := int(subset.face_indices[j])
			if face >= 0 && face < len(face_mats) {
				face_mats[face] = mat_idx
			}
		}
	}
}

usd_build_material :: proc(prim: Usd_Shim_Prim, state: ^usd_load_state) -> i32 {
	mat_data: Usd_Shim_Material_Data
	if usd_shim_get_bound_material(prim, &mat_data) == 0 {
		return -1
	}
	return usd_append_material(mat_data, state)
}

usd_append_material :: proc(mat_data: Usd_Shim_Material_Data, state: ^usd_load_state) -> i32 {
	mat_data := mat_data

	mat := Material{
		kind      = .Principled,
		albedo    = Color{f64(mat_data.base_color[0]), f64(mat_data.base_color[1]), f64(mat_data.base_color[2])},
		roughness = f64(mat_data.roughness),
		metallic  = f64(mat_data.metallic),
		emission  = Color{f64(mat_data.emissive_color[0]), f64(mat_data.emissive_color[1]), f64(mat_data.emissive_color[2])},
		ir        = 1.0,
	}

	if mat_data.has_base_color_tex != 0 {
		mat.albedo_tex = usd_load_texture(state, cstring(&mat_data.base_color_tex[0]), srgb = true)
	}
	if mat_data.has_normal_tex != 0 {
		mat.normal_tex = usd_load_texture(state, cstring(&mat_data.normal_tex[0]), srgb = false)
		mat.normal_scale = 1.0
	}
	if mat_data.has_emissive_tex != 0 {
		mat.emissive_tex = usd_load_texture(state, cstring(&mat_data.emissive_tex[0]), srgb = true)
	}
	// Roughness/metallic textures are packed together in glTF but are
	// independent grayscale textures in UsdPreviewSurface; Lumbre's
	// Material only has one combined metallic_roughness_tex slot (glTF
	// convention: G=roughness, B=metallic). Not wired up for v1 -- solid
	// roughness/metallic values are used even when USD supplies textures
	// for them.

	append(&state.materials, mat)
	return i32(len(state.materials) - 1)
}

usd_load_texture :: proc(state: ^usd_load_state, asset_path: cstring, srgb: bool) -> TextureMap {
	if asset_path == nil || len(asset_path) == 0 {
		return TextureMap{}
	}
	key := string(asset_path)

	if cached, ok := state.image_cache[key]; ok {
		return clone_texture(cached)
	}

	tex: TextureMap
	ok: bool

	// Read through USD's asset resolver first. A texture packaged inside a
	// .usdz resolves to "/path/model.usdz[0/tex.jpg]", which names an entry
	// in the archive and cannot be opened as a file.
	size: c.size_t
	bytes := usd_shim_read_asset(asset_path, &size)
	if bytes != nil {
		defer usd_shim_free_asset(bytes)
		tex, ok = load_texture_from_memory(bytes[:int(size)], srgb)
	}

	// Fall back to a plain file read for assets the resolver couldn't open
	// but that exist on disk next to the layer.
	if !ok {
		resolved := usd_shim_resolve_asset_path(state.stage, asset_path)
		tex, ok = load_texture(string(resolved), "", srgb)
	}

	if !ok {
		fmt.eprintln("usd: failed to load texture", key)
		return TextureMap{}
	}
	state.image_cache[strings.clone(key)] = tex
	return clone_texture(tex)
}
