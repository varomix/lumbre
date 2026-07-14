package lumbre_core

// Core-owned scene teardown. Frontends may construct scenes differently, but
// all renderer resources have one lifetime contract.
destroy_scene :: proc(scene: ^Scene) {
	for mesh in scene.meshes {
		delete(mesh.triangles)
		if mesh.name != "" {
			delete(mesh.name)
		}
	}
	for &mat in scene.materials {
		destroy_material_textures(&mat)
	}
	destroy_environment(&scene.environment)
	delete(scene.nodes)
	delete(scene.meshes)
	delete(scene.spheres)
	delete(scene.lights)
	delete(scene.materials)
}
