package lumbre_realtime

// Pulling the label targets back into system memory.
//
// The viewport never needs this — a realtime frame stays on the GPU and ImGui
// samples it there — but a dataset does: the whole point of the label pass is
// a file on disk. So this is the one place in `realtime` that stalls on the
// GPU, and it does so deliberately, with a fence rather than a device-wide
// wait, so the caller pays only for its own copy.
//
// The result is a plain CPU struct with no SDL in it, which is what lets
// `output` write it without depending on the renderer.

import "core:fmt"

import lc "../core"
import sdl "vendor:sdl3"

// The frame type itself lives in `core` — see core/labels.odin for why.
Label_Frame :: lc.Label_Frame

label_frame_destroy :: proc(f: ^Label_Frame) {
	delete(f.instance)
	delete(f.semantic)
	delete(f.depth)
	delete(f.normal)
	f^ = {}
}

// Renders the label pass at `width` × `height` and reads every channel back.
//
// Independent of `renderer_render`: a headless run wants labels without a
// beauty frame, and a viewport showing labels wants them without a readback.
labels_read :: proc(
	r: ^Renderer,
	cam: lc.Camera,
	width, height: i32,
	allocator := context.allocator,
) -> (
	frame: Label_Frame,
	ok: bool,
) {
	if width <= 0 || height <= 0 || !r.has_scene {
		return {}, false
	}
	if !labels_ensure_targets(r.gpu, &r.labels, width, height) {
		return {}, false
	}

	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil {
		fmt.eprintln("realtime: AcquireGPUCommandBuffer failed:", sdl.GetError())
		return {}, false
	}
	labels_draw(&r.labels, cmd, &r.scene, cam)

	// All four downloads share one copy pass and one fence: they are the same
	// frame, and waiting four times would serialise what the driver can batch.
	texels := int(width) * int(height)
	instance_t := download_begin(r.gpu, cmd, r.labels.instance, width, height, 4) or_return
	semantic_t := download_begin(r.gpu, cmd, r.labels.semantic, width, height, 4) or_return
	depth_t := download_begin(r.gpu, cmd, r.labels.depth_m, width, height, 4) or_return
	normal_t := download_begin(r.gpu, cmd, r.labels.normal, width, height, 8) or_return
	defer {
		sdl.ReleaseGPUTransferBuffer(r.gpu, instance_t)
		sdl.ReleaseGPUTransferBuffer(r.gpu, semantic_t)
		sdl.ReleaseGPUTransferBuffer(r.gpu, depth_t)
		sdl.ReleaseGPUTransferBuffer(r.gpu, normal_t)
	}

	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	if fence == nil {
		fmt.eprintln("realtime: label submit failed:", sdl.GetError())
		return {}, false
	}
	if !sdl.WaitForGPUFences(r.gpu, true, &fence, 1) {
		fmt.eprintln("realtime: WaitForGPUFences failed:", sdl.GetError())
		sdl.ReleaseGPUFence(r.gpu, fence)
		return {}, false
	}
	sdl.ReleaseGPUFence(r.gpu, fence)

	frame.width = width
	frame.height = height
	frame.instance = make([]u32, texels, allocator)
	frame.semantic = make([]u32, texels, allocator)
	frame.depth = make([]f32, texels, allocator)
	frame.normal = make([][4]f32, texels, allocator)

	if !map_copy(r.gpu, instance_t, frame.instance) ||
	   !map_copy(r.gpu, semantic_t, frame.semantic) ||
	   !map_copy(r.gpu, depth_t, frame.depth) {
		label_frame_destroy(&frame)
		return {}, false
	}

	// The normal target is half-float, the only channel that is not already
	// in its final CPU type.
	halves := make([][4]u16, texels, context.temp_allocator)
	if !map_copy(r.gpu, normal_t, halves) {
		label_frame_destroy(&frame)
		return {}, false
	}
	for h, i in halves {
		frame.normal[i] = {f16_to_f32(h[0]), f16_to_f32(h[1]), f16_to_f32(h[2]), f16_to_f32(h[3])}
	}

	return frame, true
}

// Queues one texture download into a fresh transfer buffer. The caller submits.
@(private)
download_begin :: proc(
	gpu: ^sdl.GPUDevice,
	cmd: ^sdl.GPUCommandBuffer,
	tex: ^sdl.GPUTexture,
	width, height: i32,
	bytes_per_texel: u32,
) -> (
	^sdl.GPUTransferBuffer,
	bool,
) {
	size := u32(width) * u32(height) * bytes_per_texel
	transfer := sdl.CreateGPUTransferBuffer(
		gpu,
		sdl.GPUTransferBufferCreateInfo{usage = .DOWNLOAD, size = size},
	)
	if transfer == nil {
		fmt.eprintln("realtime: download transfer buffer failed:", sdl.GetError())
		return nil, false
	}

	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.DownloadFromGPUTexture(
		pass,
		sdl.GPUTextureRegion{texture = tex, w = u32(width), h = u32(height), d = 1},
		// Tightly packed: no padding between rows, so the CPU side is a plain
		// row-major array.
		sdl.GPUTextureTransferInfo {
			transfer_buffer = transfer,
			pixels_per_row = u32(width),
			rows_per_layer = u32(height),
		},
	)
	sdl.EndGPUCopyPass(pass)
	return transfer, true
}

@(private)
map_copy :: proc(gpu: ^sdl.GPUDevice, transfer: ^sdl.GPUTransferBuffer, dst: []$T) -> bool {
	src := sdl.MapGPUTransferBuffer(gpu, transfer, false)
	if src == nil {
		fmt.eprintln("realtime: MapGPUTransferBuffer failed:", sdl.GetError())
		return false
	}
	defer sdl.UnmapGPUTransferBuffer(gpu, transfer)
	copy(dst, ([^]T)(src)[:len(dst)])
	return true
}

// IEEE 754 binary16 → binary32. The inverse of `f32_to_f16` in output/exr.odin,
// kept here rather than shared because `realtime` must not depend on `output`.
f16_to_f32 :: proc(h: u16) -> f32 {
	sign := u32(h >> 15) << 31
	exp := u32(h >> 10) & 0x1F
	frac := u32(h & 0x3FF)

	switch exp {
	case 0:
		if frac == 0 {
			return transmute(f32)sign // ±0
		}
		// Subnormal: shift until the implicit bit is set, then bias the
		// binary32 exponent by however far it travelled.
		shifts := u32(0)
		for frac & 0x400 == 0 {
			frac <<= 1
			shifts += 1
		}
		frac &= 0x3FF
		return transmute(f32)(sign | ((127 - 14 - shifts) << 23) | (frac << 13))
	case 0x1F:
		return transmute(f32)(sign | 0x7F800000 | (frac << 13)) // inf / NaN
	}
	return transmute(f32)(sign | ((exp + 127 - 15) << 23) | (frac << 13))
}

// Renders a beauty frame and reads it back as RGBA8, top-row-first.
//
// The viewport never needs this — it hands the texture straight to ImGui — but
// a headless run has no swapchain to present to, so the only way a rasterized
// beauty frame reaches a file is through system memory.
renderer_read_color :: proc(
	r: ^Renderer,
	cam: lc.Camera,
	width, height: i32,
	view: Debug_View,
	allocator := context.allocator,
) -> (
	pixels: []u8,
	ok: bool,
) {
	tex := renderer_render(r, cam, width, height, view)
	if tex == nil {
		return nil, false
	}

	// A second command buffer rather than a copy queued inside
	// `renderer_render`: the render path is shared with the viewport, which
	// must not pay for a readback it never reads.
	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil {
		fmt.eprintln("realtime: AcquireGPUCommandBuffer failed:", sdl.GetError())
		return nil, false
	}
	transfer := download_begin(r.gpu, cmd, tex, width, height, 4) or_return
	defer sdl.ReleaseGPUTransferBuffer(r.gpu, transfer)

	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	if fence == nil {
		fmt.eprintln("realtime: colour submit failed:", sdl.GetError())
		return nil, false
	}
	defer sdl.ReleaseGPUFence(r.gpu, fence)
	if !sdl.WaitForGPUFences(r.gpu, true, &fence, 1) {
		fmt.eprintln("realtime: WaitForGPUFences failed:", sdl.GetError())
		return nil, false
	}

	pixels = make([]u8, int(width) * int(height) * 4, allocator)
	if !map_copy(r.gpu, transfer, pixels) {
		delete(pixels, allocator)
		return nil, false
	}
	return pixels, true
}

// A bounded capture queue is owned by the batch session. A slot may be reused
// only after capture_finish has consumed its fence. Transfer storage survives
// frames, while CPU results belong to the caller. GPU commands never borrow
// CPU scene data, so the caller can encode a completed frame while the next
// queued camera renders.
Capture :: struct {
	transfers: [5]^sdl.GPUTransferBuffer,
	capacities: [5]u32,
	fence: ^sdl.GPUFence,
	width, height: i32,
	with_labels: bool,
}
capture_destroy :: proc(gpu: ^sdl.GPUDevice, c: ^Capture) {
	if c.fence != nil {
		_ = sdl.WaitForGPUFences(gpu, true, &c.fence, 1)
		sdl.ReleaseGPUFence(gpu, c.fence)
	}
	for transfer in c.transfers { if transfer != nil { sdl.ReleaseGPUTransferBuffer(gpu, transfer) } }
	c^ = {}
}
capture_begin :: proc(r: ^Renderer, c: ^Capture, cam: lc.Camera, width, height: i32, view: Debug_View, with_labels: bool) -> bool {
	if c.fence != nil || width <= 0 || height <= 0 { return false }
	// The transfer API uses u32 byte sizes; reject overflow before allocating.
	if u64(width) * u64(height) * 8 > u64(max(u32)) { return false }
	color := renderer_render(r, cam, width, height, view)
	if color == nil { return false }
	if with_labels && !labels_ensure_targets(r.gpu, &r.labels, width, height) { return false }
	count := with_labels ? 5 : 1
	textures := [5]^sdl.GPUTexture{color, r.labels.instance, r.labels.semantic, r.labels.depth_m, r.labels.normal}
	for i in 0 ..< count {
		bytes := u32(width) * u32(height) * (i == 4 ? 8 : 4)
		if c.capacities[i] < bytes {
			if c.transfers[i] != nil { sdl.ReleaseGPUTransferBuffer(r.gpu, c.transfers[i]); c.transfers[i] = nil }
			c.capacities[i] = 0
			c.transfers[i] = sdl.CreateGPUTransferBuffer(r.gpu, {usage = .DOWNLOAD, size = bytes})
			if c.transfers[i] == nil { return false }
			c.capacities[i] = bytes
		}
	}
	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil { return false }
	if with_labels && view != .Instance && view != .Semantic { labels_draw(&r.labels, cmd, &r.scene, cam) }
	pass := sdl.BeginGPUCopyPass(cmd)
	for i in 0 ..< count {
		sdl.DownloadFromGPUTexture(pass,
			{texture = textures[i], w = u32(width), h = u32(height), d = 1},
			{transfer_buffer = c.transfers[i], pixels_per_row = u32(width), rows_per_layer = u32(height)})
	}
	sdl.EndGPUCopyPass(pass)
	c.fence = sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	if c.fence == nil { return false }
	c.width, c.height, c.with_labels = width, height, with_labels
	return true
}
capture_finish :: proc(gpu: ^sdl.GPUDevice, c: ^Capture) -> (pixels: []u8, frame: Label_Frame, ok: bool) {
	if c.fence == nil { return nil, {}, false }
	defer { sdl.ReleaseGPUFence(gpu, c.fence); c.fence = nil }
	if !sdl.WaitForGPUFences(gpu, true, &c.fence, 1) { return nil, {}, false }
	texels := int(c.width) * int(c.height)
	pixels = make([]u8, texels * 4)
	defer if !ok { delete(pixels); pixels = nil; label_frame_destroy(&frame) }
	if !map_copy(gpu, c.transfers[0], pixels) { return pixels, frame, false }
	if c.with_labels {
		frame = {width = c.width, height = c.height,
			instance = make([]u32, texels), semantic = make([]u32, texels),
			depth = make([]f32, texels), normal = make([][4]f32, texels)}
		if !map_copy(gpu, c.transfers[1], frame.instance) || !map_copy(gpu, c.transfers[2], frame.semantic) ||
		   !map_copy(gpu, c.transfers[3], frame.depth) { return pixels, frame, false }
		halves := make([][4]u16, texels)
		defer delete(halves)
		if !map_copy(gpu, c.transfers[4], halves) { return pixels, frame, false }
		for h, i in halves { for k in 0 ..< 4 { frame.normal[i][k] = f16_to_f32(h[k]) } }
	}
	return pixels, frame, true
}
