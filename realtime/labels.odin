package lumbre_realtime

// Ground-truth label rendering: instance and semantic ids, metric depth, world
// normals.
//
// A **separate pass**, not extra G-buffer targets, for two reasons. The
// mechanical one is that SDL_GPU allows four colour targets and the G-buffer
// already spends all four. The real one is that labels and beauty want opposite
// things: beauty will eventually want temporal antialiasing and a tonemap,
// while a segmentation mask whose edges have been blended describes pixels
// belonging to neither object, and a depth that has been through a resolve is
// no longer a measurement. Keeping them apart means the beauty path can grow
// AA later without anyone having to remember to exclude the labels.
//
// The pass reuses the G-buffer's vertex buffer and batch list unchanged: a
// label is a property of geometry, and the batching is already grouped the way
// the draw calls want.

import "core:fmt"

import lc "../core"
import sdl "vendor:sdl3"

// Integer ids, so nothing downstream can blend or round them. Depth is a
// single float channel of metres; the normal keeps four channels because a
// three-channel float target is not universally supported.
INSTANCE_FORMAT :: sdl.GPUTextureFormat.R32_UINT
SEMANTIC_FORMAT :: sdl.GPUTextureFormat.R32_UINT
LABEL_DEPTH_FORMAT :: sdl.GPUTextureFormat.R32_FLOAT
LABEL_NORMAL_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT

// Mirrors `LabelUniforms` in shaders/label_vs.slang.
Label_Uniforms :: struct {
	view_proj: matrix[4, 4]f32,
	eye:       [4]f32,
	forward:   [4]f32,
}

Labels :: struct {
	pipeline:  ^sdl.GPUGraphicsPipeline,
	fallback: ^sdl.GPUTexture, // valid integer storage binding before labels are requested

	instance:  ^sdl.GPUTexture,
	semantic:  ^sdl.GPUTexture,
	depth_m:   ^sdl.GPUTexture, // metric depth, not the depth-stencil buffer
	normal:    ^sdl.GPUTexture,
	// Depth-stencil for the pass's own visibility test. Separate from
	// `depth_m`: one is the hardware's nonlinear buffer, the other is data.
	depth_test: ^sdl.GPUTexture,
	width:     i32,
	height:    i32,

	// Semantic class per instance id, indexed directly by the id the vertex
	// stream carries. A buffer rather than a second vertex attribute: the
	// class belongs to the object, not to its vertices.
	semantic_buffer:   ^sdl.GPUBuffer,
	semantic_capacity: u32,
	semantic_scratch:  [dynamic]u32,
}

labels_create :: proc(gpu: ^sdl.GPUDevice) -> (l: Labels, ok: bool) {
	vs := shader_create(gpu, SHADER_LABEL_VS, "vertexMain", .VERTEX, {uniform_buffers = 1})
	if vs == nil {
		return {}, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(gpu, SHADER_LABEL_FS, "fragmentMain", .FRAGMENT, {storage_buffers = 1})
	if fs == nil {
		return {}, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [4]sdl.GPUColorTargetDescription {
		{format = INSTANCE_FORMAT},
		{format = SEMANTIC_FORMAT},
		{format = LABEL_DEPTH_FORMAT},
		{format = LABEL_NORMAL_FORMAT},
	}
	vertex_input := vertex_input_state()

	l.pipeline = sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			vertex_input_state = vertex_input,
			primitive_type = .TRIANGLELIST,
			// Two-sided, matching the G-buffer: Lumbre's content has no
			// dependable winding, and a label pass that dropped backfaces
			// would disagree with the beauty image about what exists.
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			depth_stencil_state = {
				compare_op = .LESS,
				enable_depth_test = true,
				enable_depth_write = true,
			},
			target_info = {
				num_color_targets = len(targets),
				color_target_descriptions = raw_data(&targets),
				depth_stencil_format = DEPTH_FORMAT,
				has_depth_stencil_target = true,
			},
		},
	)
	if l.pipeline == nil {
		fmt.eprintln("realtime: label pipeline failed:", sdl.GetError())
		return {}, false
	}
	l.fallback = sdl.CreateGPUTexture(gpu, {
		type = .D2, format = INSTANCE_FORMAT, usage = {.COLOR_TARGET, .GRAPHICS_STORAGE_READ},
		width = 1, height = 1, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1,
	})
	if l.fallback == nil { labels_destroy(gpu, &l); return {}, false }
	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	if cmd == nil { labels_destroy(gpu, &l); return {}, false }
	target := sdl.GPUColorTargetInfo{texture = l.fallback, load_op = .CLEAR, store_op = .STORE}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.EndGPURenderPass(pass)
	if !sdl.SubmitGPUCommandBuffer(cmd) { labels_destroy(gpu, &l); return {}, false }
	return l, true
}

labels_destroy :: proc(gpu: ^sdl.GPUDevice, l: ^Labels) {
	if gpu == nil {
		return
	}
	labels_release_targets(gpu, l)
	if l.fallback != nil { sdl.ReleaseGPUTexture(gpu, l.fallback) }
	if l.pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(gpu, l.pipeline)
	}
	if l.semantic_buffer != nil {
		sdl.ReleaseGPUBuffer(gpu, l.semantic_buffer)
	}
	delete(l.semantic_scratch)
	l^ = {}
}

// Builds the instance→semantic table for a scene.
//
// Instance ids are scene node indices, and a node's class comes from its mesh;
// spheres continue past the node range and carry no class. The table is sized
// to cover every id the vertex stream can contain, so the shader's lookup is
// unconditional.
labels_build_semantic_table :: proc(l: ^Labels, scene: ^lc.Scene) {
	clear(&l.semantic_scratch)
	resize(&l.semantic_scratch, len(scene.nodes) + len(scene.spheres))

	for node, i in scene.nodes {
		class: i32 = 0
		if node.mesh_idx >= 0 && int(node.mesh_idx) < len(scene.meshes) {
			class = scene.meshes[node.mesh_idx].semantic_class_id
		}
		l.semantic_scratch[i] = u32(class)
	}
	// Spheres are not nodes and carry no prim, so they stay unlabelled.
	for i in len(scene.nodes) ..< len(l.semantic_scratch) {
		l.semantic_scratch[i] = 0
	}
}

labels_upload_semantic_table :: proc(gpu: ^sdl.GPUDevice, l: ^Labels) -> bool {
	needed := u32(max(len(l.semantic_scratch), 1) * size_of(u32))

	if l.semantic_buffer == nil || l.semantic_capacity < needed {
		if l.semantic_buffer != nil {
			sdl.ReleaseGPUBuffer(gpu, l.semantic_buffer)
		}
		l.semantic_buffer = sdl.CreateGPUBuffer(
			gpu,
			sdl.GPUBufferCreateInfo{usage = {.GRAPHICS_STORAGE_READ}, size = needed},
		)
		if l.semantic_buffer == nil {
			l.semantic_capacity = 0
			return false
		}
		l.semantic_capacity = needed
	}

	if len(l.semantic_scratch) == 0 {
		return true
	}
	bytes := (([^]u8)(raw_data(l.semantic_scratch)))[:len(l.semantic_scratch) * size_of(u32)]
	return upload_bytes(gpu, l.semantic_buffer, bytes)
}

labels_ensure_targets :: proc(gpu: ^sdl.GPUDevice, l: ^Labels, width, height: i32) -> bool {
	if l.instance != nil && l.width == width && l.height == height {
		return true
	}
	labels_release_targets(gpu, l)

	make_tex :: proc(
		gpu: ^sdl.GPUDevice, format: sdl.GPUTextureFormat, usage: sdl.GPUTextureUsageFlags,
		width, height: i32,
	) -> ^sdl.GPUTexture {
		tex := sdl.CreateGPUTexture(
			gpu,
			sdl.GPUTextureCreateInfo {
				type = .D2,
				format = format,
				usage = usage,
				width = u32(width),
				height = u32(height),
				layer_count_or_depth = 1,
				num_levels = 1,
				sample_count = ._1,
			},
		)
		if tex == nil {
			fmt.eprintln("realtime: label target creation failed:", sdl.GetError())
		}
		return tex
	}

	// IDs are integer storage reads in debug shaders. Floating-point channels
	// retain sampler usage. SDL forbids combining the two read usages.
	usage := sdl.GPUTextureUsageFlags{.COLOR_TARGET, .SAMPLER}

	l.instance = make_tex(gpu, INSTANCE_FORMAT, {.COLOR_TARGET, .GRAPHICS_STORAGE_READ}, width, height)
	l.semantic = make_tex(gpu, SEMANTIC_FORMAT, {.COLOR_TARGET, .GRAPHICS_STORAGE_READ}, width, height)
	l.depth_m = make_tex(gpu, LABEL_DEPTH_FORMAT, usage, width, height)
	l.normal = make_tex(gpu, LABEL_NORMAL_FORMAT, usage, width, height)
	l.depth_test = make_tex(gpu, DEPTH_FORMAT, {.DEPTH_STENCIL_TARGET}, width, height)

	if l.instance == nil || l.semantic == nil || l.depth_m == nil ||
	   l.normal == nil || l.depth_test == nil {
		labels_release_targets(gpu, l)
		return false
	}

	l.width = width
	l.height = height
	return true
}

labels_release_targets :: proc(gpu: ^sdl.GPUDevice, l: ^Labels) {
	for tex in ([]^sdl.GPUTexture{l.instance, l.semantic, l.depth_m, l.normal, l.depth_test}) {
		if tex != nil {
			sdl.ReleaseGPUTexture(gpu, tex)
		}
	}
	l.instance = nil
	l.semantic = nil
	l.depth_m = nil
	l.normal = nil
	l.depth_test = nil
	l.width = 0
	l.height = 0
}

// Renders every label channel in one pass.
labels_draw :: proc(
	l: ^Labels,
	cmd: ^sdl.GPUCommandBuffer,
	scene: ^Scene_GPU,
	cam: lc.Camera,
) {
	targets := [4]sdl.GPUColorTargetInfo {
		// Instance and semantic clear to 0, the unlabelled id, so background
		// is indistinguishable from "no object" by construction.
		{texture = l.instance, load_op = .CLEAR, store_op = .STORE},
		{texture = l.semantic, load_op = .CLEAR, store_op = .STORE},
		// Depth clears to 0, which no surface can produce: a real sample is
		// always in front of the camera.
		{texture = l.depth_m, load_op = .CLEAR, store_op = .STORE},
		{texture = l.normal, load_op = .CLEAR, store_op = .STORE},
	}
	depth := sdl.GPUDepthStencilTargetInfo {
		texture     = l.depth_test,
		clear_depth = 1.0,
		load_op     = .CLEAR,
		store_op    = .DONT_CARE,
		cycle       = true,
		stencil_load_op = .DONT_CARE,
		stencil_store_op = .DONT_CARE,
	}

	pass := sdl.BeginGPURenderPass(cmd, raw_data(&targets), len(targets), &depth)
	sdl.BindGPUGraphicsPipeline(pass, l.pipeline)

	f := camera_frame(cam)
	uniforms := Label_Uniforms {
		view_proj = camera_projection(f) * camera_view(f),
		eye = {f.eye.x, f.eye.y, f.eye.z, 0},
		forward = {f.forward.x, f.forward.y, f.forward.z, 0},
	}
	sdl.PushGPUVertexUniformData(cmd, 0, &uniforms, size_of(uniforms))

	buffers := [1]^sdl.GPUBuffer{l.semantic_buffer}
	sdl.BindGPUFragmentStorageBuffers(pass, 0, raw_data(&buffers), 1)

	scene_bind_vertex_buffers(pass, scene)
	// Labels do not vary by material.
	scene_mark_visible(scene, uniforms.view_proj)
	scene_draw_ignoring_material(pass, scene)
	sdl.EndGPURenderPass(pass)
}
