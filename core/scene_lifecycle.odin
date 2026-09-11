package lumbre_core

// Core-owned scene teardown. Frontends may construct scenes differently, but
// all renderer resources have one lifetime contract.
destroy_scene :: proc(scene: ^Scene) {
	for mesh in scene.meshes {
		delete(mesh.triangles)
		if mesh.name != "" {
			delete(mesh.name)
		}
		if mesh.path != "" {
			delete(mesh.path)
		}
		if mesh.semantic_class != "" {
			delete(mesh.semantic_class)
		}
	}
	for name in scene.semantic_classes {
		if name != "" {
			delete(name)
		}
	}
	delete(scene.semantic_classes)
	for &mat in scene.materials {
		destroy_material_textures(&mat)
	}
	destroy_environment(&scene.environment)
	delete(scene.nodes)
	delete(scene.meshes)
	delete(scene.spheres)
	delete(scene.lights)
	delete(scene.materials)
	for name in scene.camera_names {
		if name != "" {
			delete(name)
		}
	}
	delete(scene.camera_names)
	for path in scene.camera_paths {
		delete(path)
	}
	delete(scene.camera_paths)
	delete(scene.cameras)
}
