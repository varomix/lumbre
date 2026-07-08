package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:strconv"
import lm "core:math/linalg/glsl"

OBJ_Material :: struct {
	name:   string,
	kd:     Color,
	ks:     Color,
	ns:     f64,
	d:      f64,
	map_kd: string,
}

ObjData_raw :: struct {
	vertices:    [dynamic]Vec3,
	normals:     [dynamic]Vec3,
	texcoords:   [dynamic]Vec3,
	materials:   [dynamic]OBJ_Material,
	current_mtl: string,
}

ObjData :: struct {
	meshes:    []Mesh,
	materials: []Material,
}

parse_f64 :: proc(tok: string) -> f64 {
	val, ok := strconv.parse_f64(tok)
	return val if ok else 0.0
}

parse_i32 :: proc(tok: string) -> i32 {
	val, ok := strconv.parse_i64(tok)
	return i32(val) if ok else 0
}

sanitize_triangle_normals :: proc(tri: ^Triangle) {
	edge1 := tri.v1 - tri.v0
	edge2 := tri.v2 - tri.v0
	face_cross := lm.cross(edge1, edge2)
	if lm.length(face_cross) <= 1.0e-12 {
		return
	}
	face_n := lm.normalize(face_cross)

	has_all_normals :=
		lm.length(tri.n0) > 1.0e-12 &&
		lm.length(tri.n1) > 1.0e-12 &&
		lm.length(tri.n2) > 1.0e-12
	if !has_all_normals {
		tri.n0 = face_n
		tri.n1 = face_n
		tri.n2 = face_n
		return
	}

	n0 := lm.normalize(tri.n0)
	n1 := lm.normalize(tri.n1)
	n2 := lm.normalize(tri.n2)

	avg := n0 + n1 + n2
	if lm.length(avg) <= 1.0e-12 {
		tri.n0 = face_n
		tri.n1 = face_n
		tri.n2 = face_n
		return
	}

	if lm.dot(lm.normalize(avg), face_n) < 0.0 {
		n0 = -n0
		n1 = -n1
		n2 = -n2
	}

	min_pair_dot := min(min(lm.dot(n0, n1), lm.dot(n1, n2)), lm.dot(n2, n0))
	min_face_dot := min(min(lm.dot(n0, face_n), lm.dot(n1, face_n)), lm.dot(n2, face_n))

	if min_pair_dot < 0.5 || min_face_dot < 0.25 {
		tri.n0 = face_n
		tri.n1 = face_n
		tri.n2 = face_n
		return
	}

	tri.n0 = n0
	tri.n1 = n1
	tri.n2 = n2
}

load_obj :: proc(filepath: string, allocator := context.allocator) -> (ObjData, bool) {
	data, err := os.read_entire_file_from_path(filepath, allocator)
	if err != nil {
		fmt.eprintln("Failed to read OBJ file:", filepath, err)
		return {}, false
	}
	defer delete(data, allocator)

	content := string(data)
	lines := strings.split_lines(content)
	defer delete(lines)

	raw: ObjData_raw
	raw.vertices = make([dynamic]Vec3)
	raw.normals = make([dynamic]Vec3)
	raw.texcoords = make([dynamic]Vec3)
	raw.materials = make([dynamic]OBJ_Material)
	defer {
		delete(raw.vertices)
		delete(raw.normals)
		delete(raw.texcoords)
		delete(raw.materials)
	}

	// First pass: collect vertex data
	for line in lines {
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 || strings.has_prefix(trimmed, "#") {
			continue
		}

		tokens := strings.fields(trimmed)
		if len(tokens) == 0 {
			continue
		}

		switch tokens[0] {
		case "v":
			if len(tokens) >= 4 {
				append(&raw.vertices, Vec3{parse_f64(tokens[1]), parse_f64(tokens[2]), parse_f64(tokens[3])})
			}
		case "vn":
			if len(tokens) >= 4 {
				append(&raw.normals, Vec3{parse_f64(tokens[1]), parse_f64(tokens[2]), parse_f64(tokens[3])})
			}
		case "vt":
			if len(tokens) >= 3 {
				append(&raw.texcoords, Vec3{parse_f64(tokens[1]), parse_f64(tokens[2]), 0.0})
			}
		case "usemtl":
			if len(tokens) >= 2 {
				raw.current_mtl = tokens[1]
			}
		case "mtllib":
			if len(tokens) >= 2 {
				dir := ""
				if idx := strings.last_index(filepath, "/"); idx >= 0 {
					dir = filepath[:idx + 1]
				}
				mtl_path := strings.concatenate({dir, tokens[1]})
				load_mtl(mtl_path, &raw)
				delete(mtl_path)
			}
		}
	}

	// Second pass: build triangles
	triangles := make([dynamic]Triangle)
	current_mat_idx: i32 = 0

	for line in lines {
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 || strings.has_prefix(trimmed, "#") {
			continue
		}

		tokens := strings.fields(trimmed)
		if len(tokens) == 0 {
			continue
		}

		switch tokens[0] {
		case "usemtl":
			mtl_name := tokens[1]
			current_mat_idx = 0
			for mtl, i in raw.materials {
				if mtl.name == mtl_name {
					current_mat_idx = i32(i)
					break
				}
			}
		case "f":
			if len(tokens) < 4 {
				continue
			}
			n_triangles := len(tokens) - 3
			for t in 0 ..< n_triangles {
				indices0 := parse_face_index(tokens[1])
				indices1 := parse_face_index(tokens[2 + t])
				indices2 := parse_face_index(tokens[3 + t])

				tri := Triangle{
					mat_idx = current_mat_idx,
				}

				if indices0.v_idx > 0 && indices0.v_idx <= i32(len(raw.vertices)) {
					tri.v0 = raw.vertices[indices0.v_idx - 1]
				}
				if indices1.v_idx > 0 && indices1.v_idx <= i32(len(raw.vertices)) {
					tri.v1 = raw.vertices[indices1.v_idx - 1]
				}
				if indices2.v_idx > 0 && indices2.v_idx <= i32(len(raw.vertices)) {
					tri.v2 = raw.vertices[indices2.v_idx - 1]
				}

				if indices0.n_idx > 0 && indices0.n_idx <= i32(len(raw.normals)) {
					tri.n0 = raw.normals[indices0.n_idx - 1]
				}
				if indices1.n_idx > 0 && indices1.n_idx <= i32(len(raw.normals)) {
					tri.n1 = raw.normals[indices1.n_idx - 1]
				}
				if indices2.n_idx > 0 && indices2.n_idx <= i32(len(raw.normals)) {
					tri.n2 = raw.normals[indices2.n_idx - 1]
				} else {
					// Auto-compute face normal if not present
					edge1 := tri.v1 - tri.v0
					edge2 := tri.v2 - tri.v0
					n := lm.normalize(lm.cross(edge1, edge2))
					tri.n0 = n; tri.n1 = n; tri.n2 = n
				}

				if indices0.t_idx > 0 && indices0.t_idx <= i32(len(raw.texcoords)) {
					tri.uv0 = raw.texcoords[indices0.t_idx - 1]
				}
				if indices1.t_idx > 0 && indices1.t_idx <= i32(len(raw.texcoords)) {
					tri.uv1 = raw.texcoords[indices1.t_idx - 1]
				}
				if indices2.t_idx > 0 && indices2.t_idx <= i32(len(raw.texcoords)) {
					tri.uv2 = raw.texcoords[indices2.t_idx - 1]
				}

				sanitize_triangle_normals(&tri)
				append(&triangles, tri)
			}
		}
	}

	// Build materials from OBJ material lib
	materials := make([]Material, len(raw.materials))
	for mtl, i in raw.materials {
		mat := Material{
			kind   = .Lambertian,
			albedo = mtl.kd,
			fuzz   = 0.0,
			ir     = 1.0,
		}
		// Roughness from specular exponent (inverse mapping)
		if mtl.ns > 0 {
			mat.roughness = lm.sqrt(2.0 / (mtl.ns + 2.0))
		}
		if mtl.ks.x > 0 || mtl.ks.y > 0 || mtl.ks.z > 0 {
			mat.kind = .Principled
			mat.metallic = 1.0
		}
		// Detect light-emitting materials:
		// 1. By name "light"
		// 2. By Kd values > 1.0 (HDR emissive convention)
		is_light := mtl.name == "light"
		if !is_light {
			if mtl.kd.x > 1.0 || mtl.kd.y > 1.0 || mtl.kd.z > 1.0 {
				is_light = true
			}
		}
		if is_light {
			mat.kind = .Emissive
			// Use Kd as emission color; clamp albedo to [0,1] for indirect
			mat.emission = mtl.kd
			// Emission strength: use the magnitude of Kd for HDR lights,
			// or a fixed value for named lights
			if mtl.kd.x > 1.0 || mtl.kd.y > 1.0 || mtl.kd.z > 1.0 {
				mat.emission_strength = 1.0
				// Normalize albedo to [0,1] for the surface color
				max_c := lm.max(lm.max(mtl.kd.x, mtl.kd.y), mtl.kd.z)
				if max_c > 0 {
					mat.albedo = mtl.kd / max_c
				}
			} else {
				mat.emission_strength = 20.0
			}
		}
		materials[i] = mat
	}

	// Convert dynamic array to a permanent slice (ownership transferred to mesh)
	tri_slice := make([]Triangle, len(triangles), context.allocator)
	copy(tri_slice, triangles[:])
	delete(triangles)

	mesh := Mesh{
		name      = filepath,
		triangles = tri_slice,
	}

	meshes := make([]Mesh, 1)
	meshes[0] = mesh

	return ObjData{meshes = meshes, materials = materials}, true
}

FaceIndices :: struct {
	v_idx, t_idx, n_idx: i32,
}

parse_face_index :: proc(token: string) -> FaceIndices {
	result := FaceIndices{v_idx = 0, t_idx = 0, n_idx = 0}
	parts := strings.split(token, "/")
	defer delete(parts)

	if len(parts) >= 1 && len(parts[0]) > 0 {
		result.v_idx = parse_i32(parts[0])
	}
	if len(parts) >= 2 && len(parts[1]) > 0 {
		result.t_idx = parse_i32(parts[1])
	}
	if len(parts) >= 3 && len(parts[2]) > 0 {
		result.n_idx = parse_i32(parts[2])
	}
	return result
}

load_mtl :: proc(filepath: string, raw: ^ObjData_raw) {
	data, err := os.read_entire_file_from_path(filepath, context.allocator)
	if err != nil {
		fmt.eprintln("Failed to read MTL file:", filepath, err)
		return
	}
	defer delete(data)

	content := string(data)
	lines := strings.split_lines(content)
	defer delete(lines)

	current_idx: int = -1
	for line in lines {
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 || strings.has_prefix(trimmed, "#") {
			continue
		}
		tokens := strings.fields(trimmed)
		if len(tokens) == 0 {
			continue
		}

		switch tokens[0] {
		case "newmtl":
			if len(tokens) >= 2 {
				append(&raw.materials, OBJ_Material{
					name = strings.clone(tokens[1], context.allocator),
					kd   = Color{0.8, 0.8, 0.8},
				})
				current_idx = len(raw.materials) - 1
			}
		case "Kd":
			if current_idx >= 0 && len(tokens) >= 4 {
				raw.materials[current_idx].kd = Color{parse_f64(tokens[1]), parse_f64(tokens[2]), parse_f64(tokens[3])}
			}
		case "Ks":
			if current_idx >= 0 && len(tokens) >= 4 {
				raw.materials[current_idx].ks = Color{parse_f64(tokens[1]), parse_f64(tokens[2]), parse_f64(tokens[3])}
			}
		case "Ns":
			if current_idx >= 0 && len(tokens) >= 2 {
				raw.materials[current_idx].ns = parse_f64(tokens[1])
			}
		case "d":
			if current_idx >= 0 && len(tokens) >= 2 {
				raw.materials[current_idx].d = parse_f64(tokens[1])
			}
		case "Tr":
			if current_idx >= 0 && len(tokens) >= 2 {
				raw.materials[current_idx].d = 1.0 - parse_f64(tokens[1])
			}
		case "map_Kd":
			if current_idx >= 0 && len(tokens) >= 2 {
				raw.materials[current_idx].map_kd = strings.clone(tokens[1], context.allocator)
			}
		}
	}
}
