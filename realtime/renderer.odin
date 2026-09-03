package lumbre_realtime

// The realtime rasterization renderer.
//
// Phase 01 of plans/REALTIME_PHASE01.md. Unlike the path tracer — Metal compute
// that returns CPU pixels — this draws through SDL_GPU straight into a texture
// ImGui already knows how to display, so a realtime frame never round-trips
// through system memory.
//
// SDL_GPU is the render hardware interface here, with no Lumbre-owned
// abstraction over it: the raster pipeline is new code rather than a port, so
// there was nothing to abstract, and SDL already covers Metal, Vulkan and
// D3D12 behind one API.
//
// Two passes so far. The G-buffer writes surface properties for the whole
// scene; the debug pass reads one of its channels back to the screen. Deferred
// lighting is the next step and consumes exactly the same targets.

import "core:fmt"

import lc "../core"
import sdl "vendor:sdl3"

// What the viewport shows. Everything but `Shaded` reads one G-buffer channel
// raw, which is how a wrong normal or a mis-batched texture becomes visible
// rather than merely suspected.
Debug_View :: enum i32 {
	Shaded    = 0, // deferred lighting; falls back to albedo until it exists
	Albedo    = 1,
	Normal    = 2,
	Roughness = 3,
	Metallic  = 4,
	Emission  = 5,
	Depth     = 6,
}

// The presented image. UNORM rather than an sRGB format because the debug pass
// encodes gamma itself.
COLOR_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM

// G-buffer layout. See shaders/gbuffer_fs.slang for what each channel holds.
ALBEDO_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM
NORMAL_FORMAT :: sdl.GPUTextureFormat.R16G16_FLOAT
SURFACE_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM
EMISSION_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT
DEPTH_FORMAT :: sdl.GPUTextureFormat.D32_FLOAT

Renderer :: struct {
	gpu:              ^sdl.GPUDevice, // borrowed from the app; not owned
	gbuffer_pipeline: ^sdl.GPUGraphicsPipeline,
	debug_pipeline:   ^sdl.GPUGraphicsPipeline,
	lighting_pipeline: ^sdl.GPUGraphicsPipeline,
	shadow_pipeline:  ^sdl.GPUGraphicsPipeline,

	// Depth array, one layer per cascade, plus the comparison sampler that
	// does the depth test in hardware.
	shadow_map:       ^sdl.GPUTexture,
	shadow_sampler:   ^sdl.GPUSampler,

	// Analytic lights, re-uploaded whenever the scene's light list changes.
	light_buffer:     ^sdl.GPUBuffer,
	light_capacity:   u32,
	light_count:      u32,
	light_scratch:    [dynamic]Light_GPU,

	// Reads G-buffer targets in the fullscreen passes. Point-filtered and
	// clamped: these are screen-aligned, so any filtering would only smear
	// neighbouring texels of data that is not a colour.
	target_sampler:   ^sdl.GPUSampler,

	// Render targets, resized to follow the viewport panel.
	color:            ^sdl.GPUTexture,
	albedo:           ^sdl.GPUTexture,
	normal:           ^sdl.GPUTexture,
	surface:          ^sdl.GPUTexture,
	emission:         ^sdl.GPUTexture,
	depth:            ^sdl.GPUTexture,
	width:            i32,
	height:           i32,

	// The uploaded scene. Rebuilt only when `scene.key` changes, so navigation
	// never touches geometry.
	scene:            Scene_GPU,
	has_scene:        bool,
}

renderer_create :: proc(gpu: ^sdl.GPUDevice) -> (r: Renderer, ok: bool) {
	r.gpu = gpu

	r.gbuffer_pipeline = make_gbuffer_pipeline(gpu) or_return
	r.debug_pipeline = make_debug_pipeline(gpu) or_return
	r.lighting_pipeline = make_lighting_pipeline(gpu) or_return
	r.shadow_pipeline = make_shadow_pipeline(gpu) or_return

	r.shadow_map = sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = .D2_ARRAY,
			format = DEPTH_FORMAT,
			usage = {.DEPTH_STENCIL_TARGET, .SAMPLER},
			width = SHADOW_RESOLUTION,
			height = SHADOW_RESOLUTION,
			layer_count_or_depth = CASCADE_COUNT,
			num_levels = 1,
			sample_count = ._1,
		},
	)
	if r.shadow_map == nil {
		fmt.eprintln("realtime: shadow map creation failed:", sdl.GetError())
		return {}, false
	}

	// A comparison sampler filters the RESULT of the depth test, not the
	// depths. Filtering depths and comparing afterwards gives wrong occlusion
	// along every shadow edge.
	r.shadow_sampler = sdl.CreateGPUSampler(
		gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .LINEAR,
			mag_filter = .LINEAR,
			mipmap_mode = .NEAREST,
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
			compare_op = .LESS_OR_EQUAL,
			enable_compare = true,
		},
	)
	if r.shadow_sampler == nil {
		fmt.eprintln("realtime: shadow sampler creation failed:", sdl.GetError())
		return {}, false
	}

	r.target_sampler = sdl.CreateGPUSampler(
		gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .NEAREST,
			mag_filter = .NEAREST,
			mipmap_mode = .NEAREST,
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
		},
	)
	if r.target_sampler == nil {
		fmt.eprintln("realtime: CreateGPUSampler failed:", sdl.GetError())
		return {}, false
	}

	return r, true
}

renderer_destroy :: proc(r: ^Renderer) {
	if r.gpu == nil {
		return
	}
	scene_destroy(r.gpu, &r.scene)
	release_targets(r)
	if r.target_sampler != nil {
		sdl.ReleaseGPUSampler(r.gpu, r.target_sampler)
	}
	if r.gbuffer_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.gbuffer_pipeline)
	}
	if r.debug_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.debug_pipeline)
	}
	if r.lighting_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.lighting_pipeline)
	}
	if r.shadow_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.shadow_pipeline)
	}
	if r.shadow_map != nil {
		sdl.ReleaseGPUTexture(r.gpu, r.shadow_map)
	}
	if r.shadow_sampler != nil {
		sdl.ReleaseGPUSampler(r.gpu, r.shadow_sampler)
	}
	if r.light_buffer != nil {
		sdl.ReleaseGPUBuffer(r.gpu, r.light_buffer)
	}
	delete(r.light_scratch)
	r^ = {}
}

// Uploads `scene` unless the key says the already-uploaded one is still
// current. The key is the IPR's `scene_key`, which is bumped when the scene
// changes and never on a camera move.
renderer_set_scene :: proc(r: ^Renderer, scene: ^lc.Scene, key: u64) -> bool {
	if r.has_scene && r.scene.key == key {
		return true
	}
	scene_destroy(r.gpu, &r.scene)
	r.has_scene = false

	uploaded, ok := scene_upload(r.gpu, scene, key)
	if !ok {
		return false
	}
	r.scene = uploaded
	r.has_scene = true

	lights_convert(scene.lights, &r.light_scratch)
	if !lights_upload(r.gpu, &r.light_buffer, &r.light_capacity, r.light_scratch[:]) {
		fmt.eprintln("realtime: light buffer upload failed:", sdl.GetError())
		return false
	}
	r.light_count = u32(len(r.light_scratch))
	return true
}

// Draws one frame and returns the texture holding it, or nil on failure.
//
// Called only when something changed — the app's redraw model is idle-driven
// and a realtime mode that repainted unconditionally would undo that.
renderer_render :: proc(
	r: ^Renderer,
	cam: lc.Camera,
	width, height: i32,
	view: Debug_View,
) -> ^sdl.GPUTexture {
	if width <= 0 || height <= 0 || !r.has_scene {
		return nil
	}
	if !ensure_targets(r, width, height) {
		return nil
	}

	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil {
		fmt.eprintln("realtime: AcquireGPUCommandBuffer failed:", sdl.GetError())
		return nil
	}

	cascades: Cascades
	if dir, has_sun := sun_light_dir(r); has_sun && view == .Shaded {
		cascades = shadow_build_cascades(
			camera_frame(cam), dir, r.scene.bounds_min, r.scene.bounds_max,
		)
		if cascades.enabled {
			draw_shadows(r, cmd, cascades)
		}
	}

	draw_gbuffer(r, cmd, cam)
	if view == .Shaded {
		draw_lighting(r, cmd, cam, cascades)
	} else {
		draw_debug(r, cmd, cam, view)
	}

	if !sdl.SubmitGPUCommandBuffer(cmd) {
		fmt.eprintln("realtime: SubmitGPUCommandBuffer failed:", sdl.GetError())
		return nil
	}
	return r.color
}

// ── passes ───────────────────────────────────────────────────────────────────

@(private = "file")
draw_gbuffer :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera) {
	targets := [4]sdl.GPUColorTargetInfo {
		{texture = r.albedo, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
		{texture = r.normal, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
		{texture = r.surface, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
		{texture = r.emission, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
	}
	depth := sdl.GPUDepthStencilTargetInfo {
		texture     = r.depth,
		clear_depth = 1.0,
		load_op     = .CLEAR,
		store_op    = .STORE,
		cycle       = true,
	}

	pass := sdl.BeginGPURenderPass(cmd, raw_data(&targets), len(targets), &depth)
	sdl.BindGPUGraphicsPipeline(pass, r.gbuffer_pipeline)

	uniforms := camera_uniforms(cam)
	sdl.PushGPUVertexUniformData(cmd, 0, &uniforms, size_of(uniforms))

	binding := sdl.GPUBufferBinding{buffer = r.scene.vertices, offset = 0}
	sdl.BindGPUVertexBuffers(pass, 0, &binding, 1)

	for b in r.scene.batches {
		samplers := [4]sdl.GPUTextureSamplerBinding {
			{texture = b.albedo, sampler = r.scene.sampler},
			{texture = b.mr, sampler = r.scene.sampler},
			{texture = b.normal, sampler = r.scene.sampler},
			{texture = b.emissive, sampler = r.scene.sampler},
		}
		sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))

		mat := b.material
		sdl.PushGPUFragmentUniformData(cmd, 0, &mat, size_of(mat))
		sdl.DrawGPUPrimitives(pass, b.vertex_count, 1, b.first_vertex, 0)
	}

	sdl.EndGPURenderPass(pass)
}

@(private = "file")
draw_debug :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera, view: Debug_View) {
	target := sdl.GPUColorTargetInfo {
		texture     = r.color,
		clear_color = {0.05, 0.05, 0.06, 1.0},
		load_op     = .CLEAR,
		store_op    = .STORE,
	}

	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.debug_pipeline)

	samplers := [5]sdl.GPUTextureSamplerBinding {
		{texture = r.albedo, sampler = r.target_sampler},
		{texture = r.normal, sampler = r.target_sampler},
		{texture = r.surface, sampler = r.target_sampler},
		{texture = r.emission, sampler = r.target_sampler},
		{texture = r.depth, sampler = r.target_sampler},
	}
	sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))

	// The depth view needs the same near/far the projection used, or it
	// linearizes against the wrong range and reads as flat white.
	f := camera_frame(cam)
	// Scale the depth view against how far the SCENE reaches, not the focus
	// distance: anything extending past a couple of focus distances -- a ground
	// plane, most of an interior -- saturates to flat white otherwise.
	display_far := scene_view_far(f, r.scene.bounds_min, r.scene.bounds_max)
	params := [4]f32{f32(i32(view)), f.focus * NEAR_SCALE, f.focus * FAR_SCALE, display_far}
	sdl.PushGPUFragmentUniformData(cmd, 0, &params, size_of(params))

	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
}

@(private = "file")
draw_lighting :: proc(
	r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera, cascades: Cascades,
) {
	target := sdl.GPUColorTargetInfo {
		texture     = r.color,
		clear_color = {0, 0, 0, 1},
		load_op     = .CLEAR,
		store_op    = .STORE,
	}

	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.lighting_pipeline)

	samplers := [6]sdl.GPUTextureSamplerBinding {
		{texture = r.albedo, sampler = r.target_sampler},
		{texture = r.normal, sampler = r.target_sampler},
		{texture = r.surface, sampler = r.target_sampler},
		{texture = r.emission, sampler = r.target_sampler},
		{texture = r.depth, sampler = r.target_sampler},
		{texture = r.shadow_map, sampler = r.shadow_sampler},
	}
	sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))

	buffers := [1]^sdl.GPUBuffer{r.light_buffer}
	sdl.BindGPUFragmentStorageBuffers(pass, 0, raw_data(&buffers), 1)

	uniforms := lighting_uniforms(cam, r.light_count, cascades)
	sdl.PushGPUFragmentUniformData(cmd, 0, &uniforms, size_of(uniforms))

	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
}

// Renders the scene's depth from the light, once per cascade. Reuses the
// G-buffer's vertex buffer and batch list unchanged: a shadow caster is any
// geometry, and material has no bearing on depth.
@(private = "file")
draw_shadows :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cascades: Cascades) {
	binding := sdl.GPUBufferBinding{buffer = r.scene.vertices, offset = 0}

	for cascade, i in cascades.slices {
		depth := sdl.GPUDepthStencilTargetInfo {
			texture     = r.shadow_map,
			clear_depth = 1.0,
			load_op     = .CLEAR,
			store_op    = .STORE,
			layer       = u8(i),
		}

		pass := sdl.BeginGPURenderPass(cmd, nil, 0, &depth)
		sdl.BindGPUGraphicsPipeline(pass, r.shadow_pipeline)

		view_proj := cascade.view_proj
		sdl.PushGPUVertexUniformData(cmd, 0, &view_proj, size_of(view_proj))
		sdl.BindGPUVertexBuffers(pass, 0, &binding, 1)

		// One draw for the whole scene: the batches are contiguous and share a
		// vertex buffer, and depth does not care which material a triangle has.
		sdl.DrawGPUPrimitives(pass, r.scene.vertex_count, 1, 0, 0)
		sdl.EndGPURenderPass(pass)
	}
}

// The sun, if the uploaded scene has one.
@(private = "file")
sun_light_dir :: proc(r: ^Renderer) -> ([3]f32, bool) {
	for l in r.light_scratch {
		if i32(l.params.x) == i32(Light_Kind_GPU.Distant) {
			return {l.direction.x, l.direction.y, l.direction.z}, true
		}
	}
	return {}, false
}

// ── pipelines ────────────────────────────────────────────────────────────────

@(private = "file")
make_gbuffer_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_GBUFFER_VS, "vertexMain", .VERTEX, {uniform_buffers = 1})
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(
		gpu, SHADER_GBUFFER_FS, "fragmentMain", .FRAGMENT,
		{samplers = 4, uniform_buffers = 1},
	)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [4]sdl.GPUColorTargetDescription {
		{format = ALBEDO_FORMAT},
		{format = NORMAL_FORMAT},
		{format = SURFACE_FORMAT},
		{format = EMISSION_FORMAT},
	}

	vertex_input := vertex_input_state()

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			vertex_input_state = vertex_input,
			primitive_type = .TRIANGLELIST,
			// No culling. Lumbre's scenes come from USD and OBJ content with
			// no reliable winding convention, and the path tracer shades both
			// faces — culling here would make the two modes disagree about
			// which surfaces exist, which is worse than drawing backfaces.
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE, front_face = .COUNTER_CLOCKWISE},
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
	if pipeline == nil {
		fmt.eprintln("realtime: G-buffer pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

@(private = "file")
make_shadow_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_SHADOW_VS, "vertexMain", .VERTEX, {uniform_buffers = 1})
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(gpu, SHADER_SHADOW_FS, "fragmentMain", .FRAGMENT)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	vertex_input := vertex_input_state()

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			vertex_input_state = vertex_input,
			primitive_type = .TRIANGLELIST,
			// Front-face culling is the usual acne remedy, but Lumbre's content
			// has no reliable winding -- the G-buffer pass draws two-sided for
			// the same reason -- so culling here would drop real casters. The
			// slope-scaled bias in the shader does the work instead.
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			depth_stencil_state = {
				compare_op = .LESS,
				enable_depth_test = true,
				enable_depth_write = true,
			},
			target_info = {
				num_color_targets = 0,
				depth_stencil_format = DEPTH_FORMAT,
				has_depth_stencil_target = true,
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: shadow pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

@(private = "file")
make_lighting_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_FULLSCREEN_VS, "vertexMain", .VERTEX)
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(
		gpu, SHADER_LIGHTING_FS, "fragmentMain", .FRAGMENT,
		{samplers = 6, storage_buffers = 1, uniform_buffers = 1},
	)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription{{format = COLOR_FORMAT}}

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			target_info = {
				num_color_targets = 1,
				color_target_descriptions = raw_data(&targets),
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: lighting pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

@(private = "file")
make_debug_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_FULLSCREEN_VS, "vertexMain", .VERTEX)
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(
		gpu, SHADER_DEBUG_FS, "fragmentMain", .FRAGMENT,
		{samplers = 5, uniform_buffers = 1},
	)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription{{format = COLOR_FORMAT}}

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			target_info = {
				num_color_targets = 1,
				color_target_descriptions = raw_data(&targets),
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: debug pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

// ── targets ──────────────────────────────────────────────────────────────────

@(private = "file")
ensure_targets :: proc(r: ^Renderer, width, height: i32) -> bool {
	if r.color != nil && r.width == width && r.height == height {
		return true
	}
	release_targets(r)

	make_target :: proc(
		gpu: ^sdl.GPUDevice,
		format: sdl.GPUTextureFormat,
		usage: sdl.GPUTextureUsageFlags,
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
			fmt.eprintln("realtime: render target creation failed:", sdl.GetError())
		}
		return tex
	}

	color_usage := sdl.GPUTextureUsageFlags{.COLOR_TARGET, .SAMPLER}

	r.color = make_target(r.gpu, COLOR_FORMAT, color_usage, width, height)
	r.albedo = make_target(r.gpu, ALBEDO_FORMAT, color_usage, width, height)
	r.normal = make_target(r.gpu, NORMAL_FORMAT, color_usage, width, height)
	r.surface = make_target(r.gpu, SURFACE_FORMAT, color_usage, width, height)
	r.emission = make_target(r.gpu, EMISSION_FORMAT, color_usage, width, height)
	r.depth = make_target(r.gpu, DEPTH_FORMAT, {.DEPTH_STENCIL_TARGET, .SAMPLER}, width, height)

	if r.color == nil || r.albedo == nil || r.normal == nil ||
	   r.surface == nil || r.emission == nil || r.depth == nil {
		release_targets(r)
		return false
	}

	r.width = width
	r.height = height
	return true
}

@(private = "file")
release_targets :: proc(r: ^Renderer) {
	for tex in ([]^sdl.GPUTexture{r.color, r.albedo, r.normal, r.surface, r.emission, r.depth}) {
		if tex != nil {
			sdl.ReleaseGPUTexture(r.gpu, tex)
		}
	}
	r.color = nil
	r.albedo = nil
	r.normal = nil
	r.surface = nil
	r.emission = nil
	r.depth = nil
	r.width = 0
	r.height = 0
}
