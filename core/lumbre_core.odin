package lumbre_core

import m "core:math/linalg/glsl"

// Lumbre_Core is the renderer-facing state shared by every frontend. File
// import, command-line parsing, Houdini Hydra sync, and image presentation are
// adapters around this object; they do not implement a second renderer.
Lumbre_Core :: struct {
	scene:    Scene,
	settings: Render_Config,
}

lumbre_core_init :: proc(scene: Scene, settings: Render_Config) -> Lumbre_Core {
	return Lumbre_Core{scene = scene, settings = settings}
}

// Replaces the core's first mesh with already-triangulated frontend data.
// Hydra, OBJ/glTF/USD importers, and procedural generators can all use this
// without knowing about the renderer's acceleration structures.
lumbre_core_replace_triangles :: proc(core: ^Lumbre_Core, triangles: []Triangle) {
	for mesh in core.scene.meshes {
		delete(mesh.triangles)
	}
	delete(core.scene.meshes)
	delete(core.scene.nodes)

	// Heap-allocate the mesh, node, and material slices. A slice literal
	// (`[]Mesh{...}`) is backed by a temporary that dangles once this proc
	// returns, so the renderer would later read freed memory and flatten to
	// zero triangles.
	mesh_triangles := make([]Triangle, len(triangles))
	copy(mesh_triangles, triangles)

	meshes := make([]Mesh, 1)
	meshes[0] = Mesh{triangles = mesh_triangles, transform = m.mat4(1)}
	core.scene.meshes = meshes

	nodes := make([]SceneNode, 2)
	nodes[0] = make_node(m.mat4(1), -1, -1, -1)
	nodes[1] = make_node(m.mat4(1), 0, -1, 0)
	core.scene.nodes = nodes

	if len(core.scene.materials) == 0 {
		materials := make([]Material, 1)
		materials[0] = Material{kind = .Lambertian, albedo = Color{0.7, 0.7, 0.7}}
		core.scene.materials = materials
	}
}

// Replaces the analytic-light list supplied by an interactive frontend.
lumbre_core_replace_lights :: proc(core: ^Lumbre_Core, lights: []Light) {
	delete(core.scene.lights)
	core.scene.lights = make([]Light, len(lights))
	copy(core.scene.lights, lights)
}

// Frontends consume this in-memory buffer. Disk output and Houdini display
// are deliberately outside the renderer core.
lumbre_core_render_cpu :: proc(core: ^Lumbre_Core) -> Render_Buffer {
	return render_cpu_to_buffer(&core.scene, core.settings.image_width, core.settings.image_height,
		core.settings.samples_per_pixel, core.settings.max_depth,
		core.settings.max_radiance, core.settings.roughness_cutoff,
		core.settings.glossy_bias)
}
