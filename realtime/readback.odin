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

// One frame of ground truth, top-row-first — the order the GPU hands textures
// back, and the opposite of the path tracer's bottom-row-first beauty buffer.
// Whoever writes it to a file is responsible for the flip; see
// `output/labels.odin`.
Label_Frame :: struct {
	width:    i32,
	height:   i32,
	instance: []u32,
	semantic: []u32,
	depth:    []f32,    // metres along the view axis; 0 is background
	normal:   [][4]f32, // world space, raw [-1, 1]
}

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
@(private = "file")
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

@(private = "file")
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
