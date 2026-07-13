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
	// primvars:displayColor collapsed to one constant color; the fallback
	// albedo for prims that bind no material. See usd_shim.h.
	has_display_color:    c.int,
	display_color:        [3]f32,
}

Usd_Gprim_Type :: enum i32 {
	None     = 0,
	Cube     = 1,
	Sphere   = 2,
	Cylinder = 3,
	Cone     = 4,
	Capsule  = 5,
}

Usd_Axis :: enum i32 {
	X = 0,
	Y = 1,
	Z = 2,
}

// A parametric gprim stores no points, only these numbers. usd_gprim.odin
// turns them into triangles. See usd_shim.h for the per-field semantics.
Usd_Shim_Gprim_Data :: struct {
	type:          Usd_Gprim_Type,
	axis:          Usd_Axis,
	size:          f64,
	radius:        f64,
	radius_bottom: f64,
	radius_top:    f64,
	height:        f64,
}

// Which channel of a texture drives a scalar input. A scalar fed by a packed
// ARM/ORM map reads one channel of it; a dedicated greyscale map reads R.
Usd_Channel :: enum i32 {
	R = 0,
	G = 1,
	B = 2,
	A = 3,
}

Usd_Shim_Material_Data :: struct {
	material_path:         [512]u8,
	base_color:            [3]f32,
	has_base_color_tex:    c.int,
	base_color_tex:        [1024]u8,
	roughness:             f32,
	has_roughness_tex:     c.int,
	roughness_tex:         [1024]u8,
	roughness_tex_channel: Usd_Channel,
	// roughness = sample[channel] * scale + bias. The shim folds the
	// multiply/invert nodes it crossed reaching the image into these.
	roughness_tex_scale:   f32,
	roughness_tex_bias:    f32,
	metallic:              f32,
	has_metallic_tex:      c.int,
	metallic_tex:          [1024]u8,
	metallic_tex_channel:  Usd_Channel,
	metallic_tex_scale:    f32,
	metallic_tex_bias:     f32,
	opacity:               f32,
	ior:                   f32,
	transmission:          f32,
	transmission_color:    [3]f32,
	coat:                  f32,
	coat_roughness:        f32,
	emissive_color:        [3]f32,
	has_emissive_tex:      c.int,
	emissive_tex:          [1024]u8,
	has_normal_tex:        c.int,
	normal_tex:            [1024]u8,
	// MaterialX standard_surface SSS values. Texture-driven SSS is not yet
	// supported by the common material representation.
	subsurface:             f32,
	subsurface_color:       [3]f32,
	subsurface_radius:      [3]f32,
	subsurface_scale:       f32,
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

Usd_Shim_Camera_Data :: struct {
	focal_length_mm:        f32,
	horizontal_aperture_mm: f32,
	vertical_aperture_mm:   f32,
	clipping_range:         [2]f32,
	focus_distance:         f32, // 0 = unauthored
	f_stop:                 f32, // 0 = unauthored (no depth of field)
}

Usd_Shim_Light_Kind :: enum i32 {
	None     = 0,
	Sphere   = 1,
	Rect     = 2,
	Disk     = 3,
	Cylinder = 4,
	Distant  = 5,
	Dome     = 6,
}

Usd_Shim_Light_Data :: struct {
	kind:                   Usd_Shim_Light_Kind,
	intensity:              f32,
	exposure:               f32,
	color:                  [3]f32,
	normalize:              c.int,
	width, height:          f32,
	radius:                 f32,
	length:                 f32,
	angle:                  f32,
	treat_as_point:         c.int,
	has_shaping:            c.int,
	shaping_cone_angle:     f32,
	shaping_cone_softness:  f32,
	texture_file:           [1024]u8,
}

Usd_Shim_Stage_Info :: struct {
	up_axis:         c.int, // 0 = Y, 1 = Z
	meters_per_unit: f64,
}

Usd_Shim_Stage :: distinct rawptr
Usd_Shim_Prim :: distinct rawptr

@(default_calling_convention = "c")
foreign usd_shim {
	usd_shim_open_flattened :: proc(path: cstring, err_buf: [^]u8, err_buf_len: c.int) -> Usd_Shim_Stage ---
	usd_shim_close :: proc(stage: Usd_Shim_Stage) ---
	usd_shim_get_stage_info :: proc(stage: Usd_Shim_Stage, out: ^Usd_Shim_Stage_Info) -> c.int ---
	usd_shim_get_pseudo_root :: proc(stage: Usd_Shim_Stage) -> Usd_Shim_Prim ---
	usd_shim_get_children :: proc(prim: Usd_Shim_Prim, out: [^]Usd_Shim_Prim, max: c.int) -> c.int ---
	usd_shim_prim_type_name :: proc(prim: Usd_Shim_Prim) -> cstring ---
	usd_shim_prim_name :: proc(prim: Usd_Shim_Prim) -> cstring ---
	usd_shim_get_local_transform :: proc(prim: Usd_Shim_Prim, out_mat4x4: [^]f64) -> c.int ---
	usd_shim_get_mesh_data :: proc(prim: Usd_Shim_Prim, out: ^Usd_Shim_Mesh_Data, subdiv_level: c.int) -> c.int ---
	usd_shim_free_mesh_data :: proc(data: ^Usd_Shim_Mesh_Data) ---
	usd_shim_get_gprim_data :: proc(prim: Usd_Shim_Prim, out: ^Usd_Shim_Gprim_Data) -> c.int ---
	usd_shim_get_bound_material :: proc(prim: Usd_Shim_Prim, out: ^Usd_Shim_Material_Data) -> c.int ---
	usd_shim_resolve_asset_path :: proc(stage: Usd_Shim_Stage, asset_path: cstring) -> cstring ---
	usd_shim_read_asset :: proc(resolved_path: cstring, out_size: ^c.size_t) -> [^]u8 ---
	usd_shim_free_asset :: proc(data: [^]u8) ---
	usd_shim_load_image :: proc(asset_path: cstring, flip_v: c.int, out_w, out_h: ^c.int, out_pixels: ^[^]u8) -> c.int ---
	usd_shim_free_image :: proc(pixels: [^]u8) ---
	usd_shim_get_subset_count :: proc(mesh: Usd_Shim_Prim) -> c.int ---
	usd_shim_get_subsets :: proc(mesh: Usd_Shim_Prim, out: [^]Usd_Shim_Subset_Data, max: c.int) -> c.int ---
	usd_shim_free_subsets :: proc(subsets: [^]Usd_Shim_Subset_Data, count: c.int) ---
	usd_shim_get_camera_data :: proc(prim: Usd_Shim_Prim, out: ^Usd_Shim_Camera_Data) -> c.int ---
	usd_shim_get_light_data :: proc(prim: Usd_Shim_Prim, out: ^Usd_Shim_Light_Data) -> c.int ---
}

// One entry per UsdGeomCamera prim found while walking the stage. Lens
// parameters only -- position/orientation live in `world`, accumulated the
// same way as every other Xformable prim. See usd_camera.odin for the
// conversion into Lumbre's Camera.
Usd_Camera_Info :: struct {
	world:                  m.mat4,
	name:                   string,
	focal_length_mm:        f64,
	horizontal_aperture_mm: f64,
	vertical_aperture_mm:   f64,
	clip_near, clip_far:    f64,
	focus_distance:         f64, // 0 = unauthored
	f_stop:                 f64, // 0 = unauthored (no depth of field)
}

// One entry per UsdLux light prim found while walking the stage. See
// usd_light_import.odin for the mapping onto Light_Kind / Environment.
Usd_Light_Info :: struct {
	world:                 m.mat4,
	kind:                  Usd_Shim_Light_Kind,
	intensity:             f64,
	exposure:              f64,
	color:                 Color,
	normalize:             bool,
	width, height:         f64,
	radius:                f64,
	length:                f64,
	angle:                 f64,
	treat_as_point:        bool,
	has_shaping:           bool,
	shaping_cone_angle:    f64,
	shaping_cone_softness: f64,
	texture_file:          string,
}

usd_load_state :: struct {
	stage:       Usd_Shim_Stage,
	base_dir:    string,
	materials:   [dynamic]Material,
	// Material binding is resolved per-prim, so the same UsdShadeMaterial
	// comes back once per mesh and once per face subset that binds it.
	// Keyed on its stage path, `material_ids` collapses those repeats onto
	// one index in `materials` -- without it a 67-mesh asset like crag
	// builds 67 materials, each holding its own copy of the same textures.
	material_ids: map[string]i32,
	// Avoids re-decoding an image file shared by several materials. Also
	// holds negative entries for images that failed to load.
	image_cache: map[string]TextureMap,
	// Uniform subdivision level for catmullClark/loop/bilinear meshes. 0
	// renders the authored control cage; the shim refines to this level
	// otherwise. See usd_shim_get_mesh_data.
	subdiv_level: i32,
}

// Loads a USD stage (.usd/.usda/.usdc/.usdz) into Lumbre's common ObjData
// intermediate, mirroring load_gltf/load_obj so it plugs into the existing
// build_default_scene_graph / flatten_scene_graph pipeline unchanged.
//
// Composition (references/layers/variants) is flattened up front via
// UsdStage::Flatten() in the shim -- v1 does not support live variant
// switching or unflattened composition arcs.
load_usd :: proc(path: string, subdiv_level: i32 = 2, allocator := context.allocator) -> (data: ObjData, cameras: []Usd_Camera_Info, lights: []Usd_Light_Info, ok: bool) {
	cpath := strings.clone_to_cstring(path, allocator)
	defer delete(cpath, allocator)

	err_buf: [512]u8
	stage := usd_shim_open_flattened(cpath, &err_buf[0], c.int(len(err_buf)))
	if stage == nil {
		fmt.eprintln("usd: failed to open", path, "err=", cstring(&err_buf[0]))
		return {}, {}, {}, false
	}
	defer usd_shim_close(stage)

	state := usd_load_state{
		stage        = stage,
		material_ids = make(map[string]i32),
		image_cache  = make(map[string]TextureMap),
		subdiv_level = subdiv_level,
	}
	if idx := strings.last_index(path, "/"); idx >= 0 {
		state.base_dir = strings.clone(path[:idx + 1], allocator)
	}
	defer {
		for k, &v in state.image_cache {
			destroy_texture(&v)
			delete(k)
		}
		delete(state.image_cache)
		for k in state.material_ids {
			delete(k)
		}
		delete(state.material_ids)
		if state.base_dir != "" {
			delete(state.base_dir, allocator)
		}
	}

	// Reconcile the stage's up-axis with Lumbre's (Y-up). A Z-up stage is
	// rotated into Y-up at the root of the traversal, so meshes, cameras and
	// lights all inherit the same correction and stay mutually consistent --
	// the alternative, fixing only the camera, would leave the geometry
	// lying on its side. metersPerUnit is read for diagnostics; Lumbre's GI
	// radii are already scene-bounds-relative (PLAN.md Stage 2) and its
	// light/DoF distances are in scene units throughout, so no current
	// consumer needs the absolute scale.
	root_transform := m.mat4(1)
	stage_info: Usd_Shim_Stage_Info
	if usd_shim_get_stage_info(stage, &stage_info) != 0 {
		if stage_info.up_axis == 1 { // Z-up: map (x,y,z) -> (x, z, -y)
			root_transform = m.mat4{
				1, 0, 0, 0,
				0, 0, 1, 0,
				0, -1, 0, 0,
				0, 0, 0, 1,
			}
		}
		fmt.println(
			"usd: stage up-axis =", "Z" if stage_info.up_axis == 1 else "Y",
			" metersPerUnit =", stage_info.meters_per_unit,
		)
	}

	all_meshes: [dynamic]Mesh
	all_cameras: [dynamic]Usd_Camera_Info
	all_lights: [dynamic]Usd_Light_Info
	root := usd_shim_get_pseudo_root(stage)
	usd_collect_meshes(root, root_transform, &all_meshes, &all_cameras, &all_lights, &state)

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

	result_cameras := make([]Usd_Camera_Info, len(all_cameras), allocator)
	copy(result_cameras, all_cameras[:])
	delete(all_cameras)

	result_lights := make([]Usd_Light_Info, len(all_lights), allocator)
	copy(result_lights, all_lights[:])
	delete(all_lights)

	fmt.println(
		"usd: loaded", len(result_meshes), "meshes,", len(result_mats), "materials,",
		len(result_cameras), "cameras,", len(result_lights), "lights from", path,
	)
	return ObjData{meshes = result_meshes, materials = result_mats}, result_cameras, result_lights, true
}

usd_collect_meshes :: proc(
	prim: Usd_Shim_Prim,
	parent_transform: m.mat4,
	meshes: ^[dynamic]Mesh,
	cameras: ^[dynamic]Usd_Camera_Info,
	lights: ^[dynamic]Usd_Light_Info,
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
	switch {
	case type_name == "Mesh":
		usd_emit_mesh(prim, world, meshes, state)
	case type_name == "Camera":
		usd_emit_camera(prim, world, cameras)
	case:
		light_data: Usd_Shim_Light_Data
		if usd_shim_get_light_data(prim, &light_data) != 0 {
			usd_emit_light(prim, light_data, world, lights)
		} else {
			// Cube/Sphere/Cylinder/Cone/Capsule carry parameters, not points.
			// Asking the shim is cheaper than keeping a list of type names in
			// sync with the schemas it understands.
			gprim: Usd_Shim_Gprim_Data
			if usd_shim_get_gprim_data(prim, &gprim) != 0 {
				usd_emit_gprim(prim, gprim, world, meshes, state)
			}
		}
	}

	children: [256]Usd_Shim_Prim
	n := int(usd_shim_get_children(prim, &children[0], 256))
	for i in 0 ..< min(n, 256) {
		usd_collect_meshes(children[i], world, meshes, cameras, lights, state)
	}
}

usd_emit_mesh :: proc(
	prim: Usd_Shim_Prim,
	transform: m.mat4,
	meshes: ^[dynamic]Mesh,
	state: ^usd_load_state,
) {
	mesh_data: Usd_Shim_Mesh_Data
	if usd_shim_get_mesh_data(prim, &mesh_data, c.int(state.subdiv_level)) == 0 {
		return
	}
	defer usd_shim_free_mesh_data(&mesh_data)
	usd_build_mesh(prim, mesh_data, transform, meshes, state, use_subsets = true)
}

// Turns already-marshalled mesh data into a Mesh and appends it. Shared by
// the UsdGeomMesh path and the tessellated-gprim path, which differ only in
// where the points came from -- and in that a gprim has no face subsets, so
// its whole surface takes the one bound material.
usd_build_mesh :: proc(
	prim: Usd_Shim_Prim,
	mesh_data: Usd_Shim_Mesh_Data,
	transform: m.mat4,
	meshes: ^[dynamic]Mesh,
	state: ^usd_load_state,
	use_subsets: bool,
) {
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
	if use_subsets {
		usd_apply_subset_materials(prim, face_mats, state)
	}

	// Faces still unbound after the mesh- and subset-level bindings fall back
	// to primvars:displayColor, the only color many assets author. Resolved
	// once and shared across the mesh's faces (and other meshes of the same
	// color); -1 when no displayColor exists, leaving the default material.
	display_idx := i32(-1)
	display_resolved := false
	for i in 0 ..< len(face_mats) {
		if face_mats[i] < 0 {
			if !display_resolved {
				display_idx = usd_display_color_material(mesh_data, state)
				display_resolved = true
			}
			face_mats[i] = display_idx
		}
	}

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

// Builds a flat Principled material from the mesh's primvars:displayColor, the
// fallback for prims that bind no UsdShadeMaterial. Returns -1 when none was
// authored. Keyed on the color itself so meshes sharing a tint (a whole set of
// gray-metal appliances, say) collapse onto one material instead of thousands.
usd_display_color_material :: proc(md: Usd_Shim_Mesh_Data, state: ^usd_load_state) -> i32 {
	if md.has_display_color == 0 {
		return -1
	}
	r, g, b := md.display_color[0], md.display_color[1], md.display_color[2]
	key := fmt.tprintf("displayColor:%v,%v,%v", r, g, b)
	if existing, ok := state.material_ids[key]; ok {
		return existing
	}
	mat := Material{
		kind      = .Principled,
		albedo    = Color{f64(r), f64(g), f64(b)},
		roughness = 1.0,
		ir        = 1.0,
	}
	append(&state.materials, mat)
	idx := i32(len(state.materials) - 1)
	state.material_ids[strings.clone(key)] = idx
	return idx
}

// Returns the index in `state.materials` for the material the shim just
// read, building it on first sight and reusing it on every later mesh or
// subset that binds the same UsdShadeMaterial.
usd_append_material :: proc(mat_data: Usd_Shim_Material_Data, state: ^usd_load_state) -> i32 {
	mat_data := mat_data

	key := string(cstring(&mat_data.material_path[0]))
	if existing, ok := state.material_ids[key]; ok {
		return existing
	}

	mat := Material{
		kind                = .Principled,
		albedo              = Color{f64(mat_data.base_color[0]), f64(mat_data.base_color[1]), f64(mat_data.base_color[2])},
		roughness           = f64(mat_data.roughness),
		metallic            = f64(mat_data.metallic),
		emission            = Color{f64(mat_data.emissive_color[0]), f64(mat_data.emissive_color[1]), f64(mat_data.emissive_color[2])},
		ir                  = f64(mat_data.ior),
		spec_trans          = f64(mat_data.transmission),
		clearcoat           = f64(mat_data.coat),
		clearcoat_roughness = f64(mat_data.coat_roughness),
	}

	// The glass lobe tints the transmitted ray by `albedo`. For a transmissive
	// material the glass colour lives in transmission_color, not base_color
	// (which standard_surface leaves for the diffuse lobe and, for pure glass,
	// weights to zero via `base`). Fold it in so green/tinted glass reads
	// correctly instead of picking up the neutral base_color default.
	if mat.spec_trans > 0.0 {
		mat.albedo = Color{
			f64(mat_data.transmission_color[0]),
			f64(mat_data.transmission_color[1]),
			f64(mat_data.transmission_color[2]),
		}
	}
	// The GPU integrator currently has no BSSRDF/random-walk lobe. Preserve
	// an authored MaterialX SSS surface as an energy-carrying diffuse lobe
	// instead of importing its intentionally black base lobe. This is a
	// stable surface-scattering fallback, not a physical SSS simulation.
	if mat_data.subsurface > 0.0 {
		weight := f64(mat_data.subsurface)
		mat.albedo = Color{
			f64(mat_data.subsurface_color[0]) * weight,
			f64(mat_data.subsurface_color[1]) * weight,
			f64(mat_data.subsurface_color[2]) * weight,
		}
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

	// Lumbre carries roughness and metallic in one glTF-style packed map
	// (G = roughness, B = metallic). USD supplies them as two independent
	// inputs, each free to read any channel of any image -- a MaterialX ARM
	// map feeds roughness from G and metallic from B of the same file,
	// while a UsdPreviewSurface may point each at a separate greyscale.
	// Repack into the one slot the shaders sample.
	if mat_data.has_roughness_tex != 0 || mat_data.has_metallic_tex != 0 {
		if packed, ok := usd_pack_metallic_roughness(state, mat_data); ok {
			mat.metallic_roughness_tex = packed
			// The packed map holds final values, so the constants that
			// multiply it in the shader must be neutral.
			mat.roughness = 1.0
			mat.metallic = 1.0
		}
	}

	append(&state.materials, mat)
	idx := i32(len(state.materials) - 1)
	state.material_ids[strings.clone(key)] = idx
	return idx
}

// Builds the G=roughness, B=metallic map Lumbre's shaders expect from
// whatever pair of source images and channels the USD material named. An
// input without a texture is baked in as its constant value, so the shader
// can multiply by 1.0 for both and get the right answer either way.
//
// When the two inputs already share one image and read G and B of it (the
// ARM/ORM layout, and the common case), the source is reused untouched.
usd_pack_metallic_roughness :: proc(
	state: ^usd_load_state,
	md: Usd_Shim_Material_Data,
) -> (TextureMap, bool) {
	md := md

	rough_src, rough_ok := TextureMap{}, false
	metal_src, metal_ok := TextureMap{}, false
	if md.has_roughness_tex != 0 {
		rough_src, rough_ok = usd_cached_texture(state, cstring(&md.roughness_tex[0]), srgb = false)
	}
	if md.has_metallic_tex != 0 {
		metal_src, metal_ok = usd_cached_texture(state, cstring(&md.metallic_tex[0]), srgb = false)
	}
	if !rough_ok && !metal_ok {
		return TextureMap{}, false
	}

	identity :: proc(scale, bias: f32) -> bool {
		return scale == 1.0 && bias == 0.0
	}
	// Fast path: one ARM image already laid out the way we sample it.
	if rough_ok && metal_ok &&
	   string(cstring(&md.roughness_tex[0])) == string(cstring(&md.metallic_tex[0])) &&
	   md.roughness_tex_channel == .G && md.metallic_tex_channel == .B &&
	   identity(md.roughness_tex_scale, md.roughness_tex_bias) &&
	   identity(md.metallic_tex_scale, md.metallic_tex_bias) {
		return clone_texture(rough_src), true
	}

	// Otherwise resample both onto one grid. Nearest-neighbour: these are
	// data maps read per-hit through the sampler's own bilinear filter, and
	// the sources are near-always the same resolution anyway.
	base := rough_src if rough_ok else metal_src
	out := make_texture(base.width, base.height)
	out.srgb = false

	fetch :: proc(tex: TextureMap, ok: bool, ch: Usd_Channel, scale, bias, fallback: f32,
	              x, y, w, h: i32) -> u8 {
		if !ok {
			return u8(clamp(fallback, 0, 1) * 255 + 0.5)
		}
		sx := x * tex.width / w
		sy := y * tex.height / h
		texel := tex.pixels[(int(sy) * int(tex.width) + int(sx)) * 4 + int(ch)]
		v := f32(texel) / 255.0 * scale + bias
		return u8(clamp(v, 0, 1) * 255 + 0.5)
	}

	for y in 0 ..< out.height {
		for x in 0 ..< out.width {
			o := (int(y) * int(out.width) + int(x)) * 4
			out.pixels[o + 0] = 255 // ambient occlusion; Lumbre ignores it
			out.pixels[o + 1] = fetch(
				rough_src, rough_ok, md.roughness_tex_channel,
				md.roughness_tex_scale, md.roughness_tex_bias, md.roughness,
				x, y, out.width, out.height,
			)
			out.pixels[o + 2] = fetch(
				metal_src, metal_ok, md.metallic_tex_channel,
				md.metallic_tex_scale, md.metallic_tex_bias, md.metallic,
				x, y, out.width, out.height,
			)
			out.pixels[o + 3] = 255
		}
	}
	return out, true
}

usd_load_texture :: proc(state: ^usd_load_state, asset_path: cstring, srgb: bool) -> TextureMap {
	tex, ok := usd_cached_texture(state, asset_path, srgb)
	if !ok {
		return TextureMap{}
	}
	return clone_texture(tex)
}

// Returns the cache's own TextureMap, not a copy -- callers must not free it
// or hand it to a Material. Use usd_load_texture for that.
usd_cached_texture :: proc(
	state: ^usd_load_state,
	asset_path: cstring,
	srgb: bool,
) -> (TextureMap, bool) {
	if asset_path == nil || len(asset_path) == 0 {
		return TextureMap{}, false
	}
	// The same image can be wanted both sRGB-decoded (base color) and raw
	// (a channel of a data map), so the colour space is part of the key.
	key := strings.concatenate({string(asset_path), "|s" if srgb else "|l"})

	if cached, ok := state.image_cache[key]; ok {
		delete(key)
		// A cached entry without pixels is a texture that already failed to
		// load. Every mesh sharing that material would otherwise retry the
		// read and reprint the warning -- crag alone binds 67 of them.
		return cached, cached.has_data
	}

	tex: TextureMap
	ok: bool

	// Read through USD's asset resolver first. A texture packaged inside a
	// .usdz resolves to "/path/model.usdz[0/tex.jpg]", which names an entry
	// in the archive and cannot be opened as a file.
	size: c.size_t
	bytes := usd_shim_read_asset(asset_path, &size)
	had_bytes := bytes != nil
	if had_bytes {
		defer usd_shim_free_asset(bytes)
		tex, ok = load_texture_from_memory(bytes[:int(size)], srgb)
	}

	// Fall back to a plain file read for assets the resolver couldn't open
	// but that exist on disk next to the layer. Skip it when the resolver
	// already handed us the bytes and stb_image simply couldn't decode them
	// (an EXR, say) -- retrying the same path through stb_image would only
	// fail again and print a misleading warning; the Hio pass below is what
	// handles that case.
	if !ok && !had_bytes {
		resolved := usd_shim_resolve_asset_path(state.stage, asset_path)
		tex, ok = load_texture(string(resolved), "", srgb)
	}

	// Last resort: decode through USD's Hio, which handles formats stb_image
	// can't (notably OpenEXR, common in .usdz lookdev assets) and resolves
	// package paths through Ar. Hio returns raw linear pixels, so the map is
	// marked linear regardless of the caller's srgb request.
	if !ok {
		tex, ok = usd_load_image_via_hio(asset_path)
	}

	if !ok {
		fmt.eprintln("usd: failed to load texture", asset_path)
		state.image_cache[key] = TextureMap{} // negative entry; see above
		return TextureMap{}, false
	}
	state.image_cache[key] = tex
	return tex, true
}

// Decodes an image via the Hio shim (usd_shim_load_image) into a linear
// TextureMap. Used as the last texture-loading fallback for formats
// stb_image doesn't handle, above all OpenEXR. flip_v = 1 matches the
// stb_image loaders, which flip on load.
usd_load_image_via_hio :: proc(asset_path: cstring, allocator := context.allocator) -> (TextureMap, bool) {
	w, h: c.int
	pixels: [^]u8
	if usd_shim_load_image(asset_path, 1, &w, &h, &pixels) == 0 {
		return TextureMap{}, false
	}
	defer usd_shim_free_image(pixels)

	count := int(w) * int(h) * 4
	tex := TextureMap{
		width    = i32(w),
		height   = i32(h),
		pixels   = make([]u8, count, allocator),
		has_data = true,
		srgb     = false, // Hio returns raw linear pixels (EXR is linear)
	}
	copy(tex.pixels, pixels[:count])
	return tex, true
}
