package main

import m "core:math/linalg/glsl"
import "core:fmt"

ROOT_NODE_INDEX :: 0

make_node :: proc(local_transform: m.mat4, mesh_idx: i32, material_override_idx: i32, parent: i32) -> SceneNode {
	return SceneNode{
		local_transform = local_transform,
		world_transform = local_transform,
		mesh_idx = mesh_idx,
		material_override_idx = material_override_idx,
		parent = parent,
	}
}

compute_world_transforms :: proc(nodes: []SceneNode) {
	if len(nodes) == 0 {
		return
	}
	nodes[ROOT_NODE_INDEX].world_transform = nodes[ROOT_NODE_INDEX].local_transform
	for i in 1 ..< len(nodes) {
		node := &nodes[i]
		if node.parent >= 0 {
			node.world_transform = nodes[node.parent].world_transform * node.local_transform
		} else {
			node.world_transform = node.local_transform
		}
	}
}

transform_point :: proc(v: Vec3, mat: m.mat4) -> Vec3 {
	v4 := mat * [4]f32{f32(v.x), f32(v.y), f32(v.z), 1.0}
	return Vec3{f64(v4.x), f64(v4.y), f64(v4.z)}
}

transform_normal_dir :: proc(n: Vec3, mat: m.mat4) -> Vec3 {
	inv_transpose := m.transpose(m.inverse(mat))
	n4 := inv_transpose * [4]f32{f32(n.x), f32(n.y), f32(n.z), 0.0}
	length := m.sqrt(f64(n4.x * n4.x + n4.y * n4.y + n4.z * n4.z))
	if length < 1.0e-8 {
		return n
	}
	return Vec3{f64(n4.x) / length, f64(n4.y) / length, f64(n4.z) / length}
}

transform_triangle :: proc(tri: Triangle, mat: m.mat4) -> Triangle {
	return Triangle{
		v0 = transform_point(tri.v0, mat),
		v1 = transform_point(tri.v1, mat),
		v2 = transform_point(tri.v2, mat),
		n0 = transform_normal_dir(tri.n0, mat),
		n1 = transform_normal_dir(tri.n1, mat),
		n2 = transform_normal_dir(tri.n2, mat),
		uv0 = tri.uv0,
		uv1 = tri.uv1,
		uv2 = tri.uv2,
		has_uv = tri.has_uv,
		mat_idx = tri.mat_idx,
	}
}

FlattenedScene :: struct {
	triangles: []Triangle,
	materials: []Material,
	lights:    []Light,
}

flatten_scene_graph :: proc(scene: ^Scene, allocator := context.allocator) -> FlattenedScene {
	triangles := make([dynamic]Triangle, allocator)
	materials := make([dynamic]Material, allocator)
	lights := make([dynamic]Light, allocator)

	compute_world_transforms(scene.nodes)

	for node in scene.nodes {
		if node.mesh_idx < 0 || i32(node.mesh_idx) >= i32(len(scene.meshes)) {
			continue
		}
		mesh := scene.meshes[node.mesh_idx]
		xform := node.world_transform

		has_override := node.material_override_idx >= 0 &&
			i32(node.material_override_idx) < i32(len(scene.materials))

		for tri in mesh.triangles {
			wt := transform_triangle(tri, xform)
			if has_override {
				wt.mat_idx = node.material_override_idx
			} else if tri.mat_idx >= 0 && i32(tri.mat_idx) < i32(len(scene.materials)) {
				wt.mat_idx = tri.mat_idx
			} else {
				wt.mat_idx = 0
			}
			append(&triangles, wt)
		}
	}

	for mat in scene.materials {
		append(&materials, mat)
	}

	for light in scene.lights {
		append(&lights, light)
	}

	fmt.println("Flattened scene:",
		len(triangles), "triangles,",
		len(materials), "materials,",
		len(lights), "lights",
	)

	return FlattenedScene{
		triangles = triangles[:],
		materials = materials[:],
		lights = lights[:],
	}
}

destroy_flattened_scene :: proc(fs: FlattenedScene) {
	delete(fs.triangles)
	delete(fs.materials)
	delete(fs.lights)
}

build_default_scene_graph :: proc(scene: ^Scene) {
	total_nodes := 1 + len(scene.meshes)
	scene.nodes = make([]SceneNode, total_nodes)

	scene.nodes[0] = make_node(m.mat4(1), -1, -1, -1)

	for mesh_idx in 0 ..< len(scene.meshes) {
		idx := 1 + mesh_idx
		scene.nodes[idx] = make_node(scene.meshes[mesh_idx].transform, i32(mesh_idx), -1, -1)
		scene.nodes[idx].parent = ROOT_NODE_INDEX
	}
}

build_sphere_scene_graph :: proc(scene: ^Scene) {
	scene.nodes = make([]SceneNode, 1)
	scene.nodes[0] = make_node(m.mat4(1), -1, -1, -1)
}
