package main

import "core:c"
import "core:fmt"
import "core:strings"
import m "core:math/linalg/glsl"
import cgltf "vendor:cgltf"

gltf_load_state :: struct {
	data:        ^cgltf.data,
	materials:   [dynamic]Material,
	image_cache: map[int]TextureMap,
}

load_gltf :: proc(path: string, allocator := context.allocator) -> (ObjData, bool) {
	cpath := strings.clone_to_cstring(path, allocator)
	defer delete(cpath, allocator)

	data, parse_res := cgltf.parse_file({}, cpath)
	if parse_res != .success {
		fmt.eprintln("gltf: failed to parse", path, "err=", parse_res)
		return {}, false
	}
	defer cgltf.free(data)

	_ = cgltf.load_buffers({}, data, cpath)

	state := gltf_load_state{
		data        = data,
		image_cache = make(map[int]TextureMap),
	}
	defer {
		for _, &v in state.image_cache {
			destroy_texture(&v)
		}
		delete(state.image_cache)
	}

	for mat, i in data.materials {
		lumbre_mat := gltf_build_material(mat, i32(i), &state)
		append(&state.materials, lumbre_mat)
	}

	scn: ^cgltf.scene = nil
	if len(data.scenes) > 0 {
		scn = &data.scenes[0]
	}
	if scn == nil {
		fmt.eprintln("gltf: no scenes in file")
		return {}, false
	}

	all_meshes: [dynamic]Mesh

	for root in scn.nodes {
		gltf_collect_meshes(root, &all_meshes, &state, data.materials)
	}

	// Count non-empty meshes, then copy them over.
	live_count := 0
	for m in all_meshes {
		if len(m.triangles) > 0 {
			live_count += 1
		} else {
			delete(m.triangles)
			if m.name != "" { delete(m.name) }
		}
	}

	result_meshes := make([]Mesh, live_count, allocator)
	write_idx := 0
	for m in all_meshes {
		if len(m.triangles) > 0 {
			result_meshes[write_idx] = m
			write_idx += 1
		}
	}
	delete(all_meshes)

	result_mats := make([]Material, len(state.materials), allocator)
	for mat, i in state.materials {
		result_mats[i] = mat
	}
	delete(state.materials)

	fmt.println("gltf: loaded", len(result_meshes), "meshes,", len(result_mats), "materials")
	return ObjData{meshes = result_meshes, materials = result_mats}, true
}

gltf_collect_meshes :: proc(
	node: ^cgltf.node,
	meshes: ^[dynamic]Mesh,
	state: ^gltf_load_state,
	all_materials: []cgltf.material,
) {
	if node.mesh != nil {
		emit_mesh(node.mesh, gltf_node_world_transform(node), meshes, state, all_materials)
	}
	for child in node.children {
		gltf_collect_meshes(child, meshes, state, all_materials)
	}
}

gltf_node_world_transform :: proc(node: ^cgltf.node) -> m.mat4 {
	raw: [16]f32
	cgltf.node_transform_world(node, raw_data(raw[:]))
	out := m.mat4(1)
	dst := ([^]f32)(&out)[:16]
	for i in 0 ..< len(raw) {
		dst[i] = raw[i]
	}
	return out
}

emit_mesh :: proc(
	cgltf_mesh: ^cgltf.mesh,
	transform: m.mat4,
	meshes: ^[dynamic]Mesh,
	state: ^gltf_load_state,
	all_materials: []cgltf.material,
) {
	mesh_name := ""
	if cgltf_mesh.name != nil {
		mesh_name = string(cgltf_mesh.name)
	}

	for _, pi in cgltf_mesh.primitives {
		prim := &cgltf_mesh.primitives[pi]
		if prim.type != .triangles {
			continue
		}

		mat_idx: i32 = 0
		if prim.material != nil {
			for &m, mi in all_materials {
				if &m == prim.material {
					mat_idx = i32(mi)
					break
				}
			}
		}

		positions: ^cgltf.accessor = nil
		normals:   ^cgltf.accessor = nil
		uvs:       ^cgltf.accessor = nil
		for attr in prim.attributes {
			#partial switch attr.type {
			case .position: positions = attr.data
			case .normal:   normals   = attr.data
			case .texcoord: uvs       = attr.data
			}
		}
		if positions == nil {
			fmt.eprintln("gltf: primitive", pi, "has no POSITION accessor; skipping")
			continue
		}
		if positions.type != .vec3 || positions.component_type != .r_32f {
			fmt.eprintln("gltf: only float32 vec3 POSITION supported; skipping primitive", pi)
			continue
		}

		vert_count := int(positions.count)
		pos_data := make([]f32, vert_count * 3)
		defer delete(pos_data)
		_ = cgltf.accessor_unpack_floats(positions, raw_data(pos_data), uint(len(pos_data)))

		normal_data: []f32
		if normals != nil && normals.type == .vec3 {
			normal_data = make([]f32, vert_count * 3)
			defer delete(normal_data)
			_ = cgltf.accessor_unpack_floats(normals, raw_data(normal_data), uint(len(normal_data)))
		}

		uv_data: []f32
		if uvs != nil && uvs.type == .vec2 {
			uv_data = make([]f32, vert_count * 2)
			defer delete(uv_data)
			_ = cgltf.accessor_unpack_floats(uvs, raw_data(uv_data), uint(len(uv_data)))
		}

		triangles := gltf_emit_tris(
			prim,
			pos_data, normal_data, uv_data,
			vert_count, mat_idx,
		)
		defer delete(triangles)

		if len(triangles) == 0 {
			continue
		}

		emit_name_buf: [256]u8
		prim_name := ""
		if mesh_name != "" {
			if pi > 0 {
				prim_name = fmt.bprintf(emit_name_buf[:], "%s.prim%d", mesh_name, pi)
			} else {
				prim_name = mesh_name
			}
		} else {
			if pi > 0 {
				prim_name = fmt.bprintf(emit_name_buf[:], "prim%d", pi)
			}
		}

		mm := Mesh{
			name      = strings.clone(prim_name, context.allocator),
			transform = transform,
		}
		mm.triangles = make([]Triangle, len(triangles), context.allocator)
		copy(mm.triangles, triangles[:])
		mm.material = state.materials[mat_idx] if mat_idx < i32(len(state.materials)) && len(state.materials) > 0 else Material{}
		append(meshes, mm)
	}
}

gltf_emit_tris :: proc(
	prim: ^cgltf.primitive,
	pos_data: []f32,
	normal_data: []f32,
	uv_data: []f32,
	vert_count: int,
	mat_idx: i32,
) -> [dynamic]Triangle {
	tris: [dynamic]Triangle

	emit_idx_tri :: proc(
		idx0, idx1, idx2: int,
		pos: []f32, nrm: []f32, uv: []f32,
		tris: ^[dynamic]Triangle,
		mat: i32,
	) {
		po := idx0 * 3
		p0 := Vec3{f64(pos[po]), f64(pos[po + 1]), f64(pos[po + 2])}
		po = idx1 * 3
		p1 := Vec3{f64(pos[po]), f64(pos[po + 1]), f64(pos[po + 2])}
		po = idx2 * 3
		p2 := Vec3{f64(pos[po]), f64(pos[po + 1]), f64(pos[po + 2])}

		face_n := m.normalize(m.cross(p1 - p0, p2 - p0))

		n0, n1, n2: Vec3
		if len(nrm) > 0 {
			no := idx0 * 3
			n0 = Vec3{f64(nrm[no]), f64(nrm[no + 1]), f64(nrm[no + 2])}
			no = idx1 * 3
			n1 = Vec3{f64(nrm[no]), f64(nrm[no + 1]), f64(nrm[no + 2])}
			no = idx2 * 3
			n2 = Vec3{f64(nrm[no]), f64(nrm[no + 1]), f64(nrm[no + 2])}
		} else {
			n0, n1, n2 = face_n, face_n, face_n
		}

		uv0, uv1, uv2: Vec3
		if len(uv) > 0 {
			uo := idx0 * 2
			uv0 = Vec3{f64(uv[uo]), f64(uv[uo + 1]), 0}
			uo = idx1 * 2
			uv1 = Vec3{f64(uv[uo]), f64(uv[uo + 1]), 0}
			uo = idx2 * 2
			uv2 = Vec3{f64(uv[uo]), f64(uv[uo + 1]), 0}
		}

		tri := Triangle{
			v0 = p0, v1 = p1, v2 = p2,
			n0 = n0, n1 = n1, n2 = n2,
			uv0 = uv0, uv1 = uv1, uv2 = uv2,
			mat_idx = mat,
		}
		sanitize_triangle_normals(&tri)
		append(tris, tri)
	}

	if prim.indices != nil {
		idx_count := int(prim.indices.count)
		idx_data := make([]u32, idx_count)
		defer delete(idx_data)
		_ = cgltf.accessor_unpack_indices(prim.indices, raw_data(idx_data), 4, uint(idx_count))
		i := 0
		for i + 2 < idx_count {
			emit_idx_tri(
				int(idx_data[i]), int(idx_data[i + 1]), int(idx_data[i + 2]),
				pos_data, normal_data, uv_data,
				&tris, mat_idx,
			)
			i += 3
		}
	} else {
		i := 0
		for i + 2 < vert_count {
			emit_idx_tri(i, i + 1, i + 2, pos_data, normal_data, uv_data, &tris, mat_idx)
			i += 3
		}
	}
	return tris
}

gltf_build_material :: proc(
	mat: cgltf.material,
	mat_idx: i32,
	state: ^gltf_load_state,
) -> Material {
	lumbre_mat := Material{
		kind          = .Lambertian,
		albedo        = Color{0.8, 0.8, 0.8},
		specular      = 0.5,
		specular_tint = Color{1.0, 1.0, 1.0},
	}

	if mat.has_pbr_metallic_roughness {
		pbr := mat.pbr_metallic_roughness
		lumbre_mat.albedo = Color{
			f64(pbr.base_color_factor[0]),
			f64(pbr.base_color_factor[1]),
			f64(pbr.base_color_factor[2]),
		}
		lumbre_mat.metallic  = f64(pbr.metallic_factor)
		lumbre_mat.roughness = f64(pbr.roughness_factor)

		if lumbre_mat.metallic >= 1.0 || lumbre_mat.roughness < 1.0 {
			lumbre_mat.kind = .Principled
		}

		if pbr.base_color_texture.texture != nil {
			tex, ok := gltf_load_texture(pbr.base_color_texture.texture, state, mat.name)
			if ok {
				lumbre_mat.albedo_tex = tex
			}
		}
	}

	if mat.emissive_factor[0] > 0.0 || mat.emissive_factor[1] > 0.0 || mat.emissive_factor[2] > 0.0 {
		lumbre_mat.emission = Color{
			f64(mat.emissive_factor[0]),
			f64(mat.emissive_factor[1]),
			f64(mat.emissive_factor[2]),
		}
		lumbre_mat.kind = .Emissive
		any_gt_one := mat.emissive_factor[0] > 1.0 ||
			mat.emissive_factor[1] > 1.0 ||
			mat.emissive_factor[2] > 1.0
		lumbre_mat.emission_strength = 1.0 if any_gt_one else 20.0
	}

	return lumbre_mat
}

gltf_load_texture :: proc(
	tex: ^cgltf.texture,
	state: ^gltf_load_state,
	mat_name: cstring,
) -> (TextureMap, bool) {
	img := tex.image_
	if img == nil {
		return TextureMap{}, false
	}

	image_idx := -1
	for &im, ii in state.data.images {
		if &im == img {
			image_idx = ii
			break
		}
	}
	if image_idx < 0 {
		return TextureMap{}, false
	}
	if cached, ok := state.image_cache[image_idx]; ok {
		return clone_texture(cached), true
	}

	tex_map: TextureMap
	loaded := false

	if img.buffer_view != nil {
		bv := img.buffer_view
		view_data := cgltf.buffer_view_data(bv)
		if view_data != nil {
			raw := ([^]u8)(view_data)
			length := int(bv.size)
			slice := raw[:length]
			tex_map, loaded = load_texture_from_memory(slice)
		}
	} else if img.uri != nil {
		uri := string(img.uri)
		if strings.has_prefix(uri, "data:") {
			fmt.eprintln("gltf: data: URI textures not yet supported (", mat_name, ")")
		} else {
			tex_map, loaded = load_texture(uri, "")
		}
	}

	if loaded {
		state.image_cache[image_idx] = clone_texture(tex_map)
		return tex_map, true
	}
	return TextureMap{}, false
}
