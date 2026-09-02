package lumbre_realtime

// The realtime rasterization renderer.
//
// Phase 01 of plans/REALTIME_PHASE01.md. Unlike the path tracer — Metal compute
// that returns CPU pixels — this draws through SDL_GPU straight into a texture
// ImGui already knows how to display, so a realtime frame never round-trips
// through system memory.
//
// SDL_GPU *is* the render hardware interface here. There is deliberately no
// Lumbre-owned abstraction over it: the raster pipeline is new code rather than
// a port, so there was nothing to abstract, and SDL already covers Metal,
// Vulkan and D3D12 behind one API.
//
// Right now this draws a test triangle. It exists to prove the toolchain end to
// end — Slang source through committed blobs to a pipeline to a displayed
// texture — before the G-buffer is built on top of it.

import "core:fmt"

import sdl "vendor:sdl3"

// The color target's format. Chosen to match what ImGui samples, not the
// swapchain: the panel composites this like any other image.
COLOR_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM

Renderer :: struct {
	gpu:      ^sdl.GPUDevice, // borrowed from the app; not owned
	pipeline: ^sdl.GPUGraphicsPipeline,

	// Offscreen color target, resized to follow the viewport panel.
	color:    ^sdl.GPUTexture,
	width:    i32,
	height:   i32,
}

renderer_create :: proc(gpu: ^sdl.GPUDevice) -> (r: Renderer, ok: bool) {
	r.gpu = gpu

	vs := shader_create(gpu, SHADER_TRIANGLE_VS, "vertexMain", .VERTEX)
	if vs == nil {
		return {}, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(gpu, SHADER_TRIANGLE_FS, "fragmentMain", .FRAGMENT)
	if fs == nil {
		return {}, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := []sdl.GPUColorTargetDescription{{format = COLOR_FORMAT}}

	r.pipeline = sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE, front_face = .COUNTER_CLOCKWISE},
			target_info = {
				num_color_targets = 1,
				color_target_descriptions = raw_data(targets),
			},
		},
	)
	if r.pipeline == nil {
		fmt.eprintln("realtime: CreateGPUGraphicsPipeline failed:", sdl.GetError())
		return {}, false
	}

	return r, true
}

renderer_destroy :: proc(r: ^Renderer) {
	if r.gpu == nil {
		return
	}
	if r.color != nil {
		sdl.ReleaseGPUTexture(r.gpu, r.color)
	}
	if r.pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.pipeline)
	}
	r^ = {}
}

// Reallocates the color target when the panel size changes. Returns false if
// the texture could not be created, in which case the caller must not render.
@(private)
renderer_ensure_targets :: proc(r: ^Renderer, width, height: i32) -> bool {
	if r.color != nil && r.width == width && r.height == height {
		return true
	}
	if r.color != nil {
		sdl.ReleaseGPUTexture(r.gpu, r.color)
		r.color = nil
	}

	r.color = sdl.CreateGPUTexture(
		r.gpu,
		sdl.GPUTextureCreateInfo {
			type = .D2,
			format = COLOR_FORMAT,
			usage = {.COLOR_TARGET, .SAMPLER},
			width = u32(width),
			height = u32(height),
			layer_count_or_depth = 1,
			num_levels = 1,
			sample_count = ._1,
		},
	)
	if r.color == nil {
		fmt.eprintln("realtime: CreateGPUTexture failed:", sdl.GetError())
		return false
	}
	r.width = width
	r.height = height
	return true
}

// Draws one frame and returns the texture holding it, or nil on failure.
//
// Called only when something changed — the app's redraw model is idle-driven
// and a realtime mode that repainted unconditionally would undo that.
renderer_render :: proc(r: ^Renderer, width, height: i32) -> ^sdl.GPUTexture {
	if width <= 0 || height <= 0 {
		return nil
	}
	if !renderer_ensure_targets(r, width, height) {
		return nil
	}

	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil {
		fmt.eprintln("realtime: AcquireGPUCommandBuffer failed:", sdl.GetError())
		return nil
	}

	color_info := sdl.GPUColorTargetInfo {
		texture     = r.color,
		clear_color = {0.05, 0.05, 0.06, 1.0},
		load_op     = .CLEAR,
		store_op    = .STORE,
	}

	pass := sdl.BeginGPURenderPass(cmd, &color_info, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.pipeline)
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)

	if !sdl.SubmitGPUCommandBuffer(cmd) {
		fmt.eprintln("realtime: SubmitGPUCommandBuffer failed:", sdl.GetError())
		return nil
	}
	return r.color
}
