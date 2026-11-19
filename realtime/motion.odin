package lumbre_realtime

// Motion vectors: per-pixel screen-space velocity, from where each surface
// point is now and where it was one frame earlier.
//
// A pass of its own, like the labels and for the same reasons: SDL_GPU allows
// four colour targets and the G-buffer and the label pass each spend all
// four, and a velocity is data, so it must not see TAA's jitter or any
// filtering. It draws the same vertex buffer and batches as every other pass;
// the previous placement of each instance comes from `Scene_GPU.prev_worlds`.
//
// The velocity is in pixels, current minus previous, x right and y down: the
// content of pixel p was at p - velocity one frame earlier. The previous
// frame's camera is a parameter, so a still from a USD stage passes the
// camera at the previous time code and the viewport its last frame's.

import "core:fmt"

import lc "../core"
import sdl "vendor:sdl3"

// Two floats per pixel at full precision: a dataset wants the measurement,
// and half floats lose a tenth of a pixel past a few hundred pixels.
MOTION_FORMAT :: sdl.GPUTextureFormat.R32G32_FLOAT

// Mirrors `MotionUniforms` in shaders/motion_vs.slang.
Motion_Vertex_Uniforms :: struct {
	view_proj:      matrix[4, 4]f32,
	prev_view_proj: matrix[4, 4]f32,
	params:         [4]f32, // x = 1 when the camera has not moved
}

Motion :: struct {
	pipeline:   ^sdl.GPUGraphicsPipeline,
	// Bound by the debug view before a motion target exists.
	fallback:   ^sdl.GPUTexture,
	velocity:   ^sdl.GPUTexture,
	depth_test: ^sdl.GPUTexture,
	width:      i32,
	height:     i32,
}

motion_create :: proc(gpu: ^sdl.GPUDevice) -> (mo: Motion, ok: bool) {
	vs := shader_create(gpu, SHADER_MOTION_VS, "vertexMain", .VERTEX, {storage_buffers = 1, uniform_buffers = 1})
	if vs == nil {
		return {}, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)
	fs := shader_create(gpu, SHADER_MOTION_FS, "fragmentMain", .FRAGMENT, {uniform_buffers = 1})
	if fs == nil {
		return {}, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription{{format = MOTION_FORMAT}}
	mo.pipeline = sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			vertex_input_state = vertex_input_state(),
			primitive_type = .TRIANGLELIST,
			// Two-sided, matching the G-buffer and labels.
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			depth_stencil_state = {
				compare_op = DEPTH_COMPARE,
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
	if mo.pipeline == nil {
		fmt.eprintln("realtime: motion pipeline failed:", sdl.GetError())
		return {}, false
	}
	mo.fallback = motion_make_texture(gpu, MOTION_FORMAT, {.COLOR_TARGET, .GRAPHICS_STORAGE_READ}, 1, 1)
	if mo.fallback == nil {
		motion_destroy(gpu, &mo)
		return {}, false
	}
	return mo, true
}

motion_destroy :: proc(gpu: ^sdl.GPUDevice, mo: ^Motion) {
	if gpu == nil {
		return
	}
	motion_release_targets(gpu, mo)
	if mo.fallback != nil { sdl.ReleaseGPUTexture(gpu, mo.fallback) }
	if mo.pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(gpu, mo.pipeline) }
	mo^ = {}
}

@(private = "file")
motion_make_texture :: proc(
	gpu: ^sdl.GPUDevice, format: sdl.GPUTextureFormat, usage: sdl.GPUTextureUsageFlags, width, height: i32,
) -> ^sdl.GPUTexture {
	tex := sdl.CreateGPUTexture(gpu, {
		type = .D2, format = format, usage = usage,
		width = u32(width), height = u32(height),
		layer_count_or_depth = 1, num_levels = 1, sample_count = ._1,
	})
	if tex == nil {
		fmt.eprintln("realtime: motion target creation failed:", sdl.GetError())
	}
	return tex
}

motion_ensure_targets :: proc(gpu: ^sdl.GPUDevice, mo: ^Motion, width, height: i32) -> bool {
	if mo.velocity != nil && mo.width == width && mo.height == height {
		return true
	}
	motion_release_targets(gpu, mo)
	// Read with Load by the debug view, as the integer id targets are.
	mo.velocity = motion_make_texture(gpu, MOTION_FORMAT, {.COLOR_TARGET, .GRAPHICS_STORAGE_READ}, width, height)
	mo.depth_test = motion_make_texture(gpu, DEPTH_FORMAT, {.DEPTH_STENCIL_TARGET}, width, height)
	if mo.velocity == nil || mo.depth_test == nil {
		motion_release_targets(gpu, mo)
		return false
	}
	mo.width = width
	mo.height = height
	return true
}

motion_release_targets :: proc(gpu: ^sdl.GPUDevice, mo: ^Motion) {
	if mo.velocity != nil { sdl.ReleaseGPUTexture(gpu, mo.velocity) }
	if mo.depth_test != nil { sdl.ReleaseGPUTexture(gpu, mo.depth_test) }
	mo.velocity = nil
	mo.depth_test = nil
	mo.width = 0
	mo.height = 0
}

// Renders the velocity of every visible surface for `cam`, given where the
// camera was one frame earlier. Background clears to zero velocity.
motion_draw :: proc(
	mo: ^Motion,
	cmd: ^sdl.GPUCommandBuffer,
	scene: ^Scene_GPU,
	cam, prev_cam: lc.Camera,
) {
	sdl.PushGPUDebugGroup(cmd, "Motion vectors")
	defer sdl.PopGPUDebugGroup(cmd)

	target := sdl.GPUColorTargetInfo{texture = mo.velocity, load_op = .CLEAR, store_op = .STORE}
	depth := sdl.GPUDepthStencilTargetInfo {
		texture          = mo.depth_test,
		clear_depth      = DEPTH_CLEAR,
		load_op          = .CLEAR,
		store_op         = .DONT_CARE,
		cycle            = true,
		stencil_load_op  = .DONT_CARE,
		stencil_store_op = .DONT_CARE,
	}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, &depth)
	sdl.BindGPUGraphicsPipeline(pass, mo.pipeline)

	f := camera_frame(cam)
	uniforms := Motion_Vertex_Uniforms {
		view_proj      = camera_view_proj(f),
		prev_view_proj = camera_view_proj(camera_frame(prev_cam)),
	}
	if uniforms.view_proj == uniforms.prev_view_proj {
		uniforms.params.x = 1
	}
	sdl.PushGPUVertexUniformData(cmd, 0, &uniforms, size_of(uniforms))
	size := [4]f32{f32(mo.width), f32(mo.height), 0, 0}
	sdl.PushGPUFragmentUniformData(cmd, 0, &size, size_of(size))
	buffers := [1]^sdl.GPUBuffer{scene.prev_worlds}
	sdl.BindGPUVertexStorageBuffers(pass, 0, raw_data(&buffers), 1)

	scene_bind_vertex_buffers(pass, scene)
	// Culled against the current view: what is not on screen now has no
	// pixel to carry a velocity.
	scene_mark_visible(scene, uniforms.view_proj)
	scene_draw_ignoring_material(pass, scene)
	sdl.EndGPURenderPass(pass)
}

// Renders the motion pass at `width` × `height` and reads the velocities
// back, two floats per pixel in row order. For tests and the dataset path.
motion_read :: proc(
	r: ^Renderer,
	cam, prev_cam: lc.Camera,
	width, height: i32,
	allocator := context.allocator,
) -> (velocity: [][2]f32, ok: bool) {
	if width <= 0 || height <= 0 || !r.has_scene {
		return nil, false
	}
	if !motion_ensure_targets(r.gpu, &r.motion, width, height) {
		return nil, false
	}
	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil {
		fmt.eprintln("realtime: AcquireGPUCommandBuffer failed:", sdl.GetError())
		return nil, false
	}
	motion_draw(&r.motion, cmd, &r.scene, cam, prev_cam)
	transfer := download_begin(r.gpu, cmd, r.motion.velocity, width, height, 8) or_return
	defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)

	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	if fence == nil {
		fmt.eprintln("realtime: motion submit failed:", sdl.GetError())
		return nil, false
	}
	defer sdl.ReleaseGPUFence(r.gpu, fence)
	if !sdl.WaitForGPUFences(r.gpu, true, &fence, 1) {
		fmt.eprintln("realtime: WaitForGPUFences failed:", sdl.GetError())
		return nil, false
	}
	velocity = make([][2]f32, int(width) * int(height), allocator)
	if !map_copy(r.gpu, transfer, velocity) {
		delete(velocity, allocator)
		return nil, false
	}
	return velocity, true
}
