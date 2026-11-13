package lumbre_realtime

// Area lights -- quad, disc, sphere and cylinder -- shaded with linearly
// transformed cosines (see realtime/ltc). Specular comes from the fitted
// table, diffuse from the same polygon integral with no transform, which is
// the polygon's exact form factor. Each light is reduced to a polygon in the
// shader: a quad is itself, a disc a 12-gon, a sphere the 12-gon facing the
// shading point that covers the same solid angle, and a cylinder the
// rectangle along its axis facing the point.
//
// The table is uploaded once, as two 64x64 half-float textures read with
// linear filtering: the inverse transform, and the magnitudes with the
// horizon table in the third channel (see Entry in ltc.odin).

import "core:fmt"

import ltc "ltc"
import sdl "vendor:sdl3"

LTC_TABLE :: #load("ltc/" + ltc.TABLE_FILE)
LTC_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT

// The two lookup textures: the inverse transform, and the magnitudes.
Ltc_Textures :: struct {
	transform: ^sdl.GPUTexture,
	magnitude: ^sdl.GPUTexture,
}

ltc_create :: proc(gpu: ^sdl.GPUDevice) -> (out: Ltc_Textures, ok: bool) {
	assert(len(LTC_TABLE) == size_of(ltc.Table), "realtime/ltc table does not match the Table layout; rerun tools/ltc_fit")
	table := (^ltc.Table)(raw_data(LTC_TABLE))
	transform := make([][4]u16, ltc.SIZE * ltc.SIZE)
	defer delete(transform)
	magnitude := make([][4]u16, ltc.SIZE * ltc.SIZE)
	defer delete(magnitude)
	for e, i in table {
		transform[i] = {f32_to_f16(e.inv.x), f32_to_f16(e.inv.y), f32_to_f16(e.inv.z), f32_to_f16(e.inv.w)}
		magnitude[i] = {f32_to_f16(e.norm), f32_to_f16(e.fresnel), f32_to_f16(e.sphere), 0}
	}
	defer if !ok { ltc_destroy(gpu, &out) }
	out.transform = upload_rgba16f(gpu, transform, ltc.SIZE, ltc.SIZE) or_return
	out.magnitude = upload_rgba16f(gpu, magnitude, ltc.SIZE, ltc.SIZE) or_return
	return out, true
}

ltc_destroy :: proc(gpu: ^sdl.GPUDevice, t: ^Ltc_Textures) {
	if t.transform != nil { sdl.ReleaseGPUTexture(gpu, t.transform) }
	if t.magnitude != nil { sdl.ReleaseGPUTexture(gpu, t.magnitude) }
	t^ = {}
}

@(private = "file")
upload_rgba16f :: proc(gpu: ^sdl.GPUDevice, texels: [][4]u16, width, height: u32) -> (^sdl.GPUTexture, bool) {
	tex := sdl.CreateGPUTexture(gpu, sdl.GPUTextureCreateInfo {
		type = .D2, format = LTC_FORMAT, usage = {.SAMPLER},
		width = width, height = height, layer_count_or_depth = 1, num_levels = 1, sample_count = ._1,
	})
	if tex == nil {
		fmt.eprintln("realtime: LTC texture creation failed:", sdl.GetError())
		return nil, false
	}
	bytes := u32(len(texels) * size_of([4]u16))
	transfer := sdl.CreateGPUTransferBuffer(gpu, {usage = .UPLOAD, size = bytes})
	if transfer == nil {
		sdl.ReleaseGPUTexture(gpu, tex)
		return nil, false
	}
	defer sdl.ReleaseGPUTransferBuffer(gpu, transfer)
	dst := sdl.MapGPUTransferBuffer(gpu, transfer, false)
	if dst == nil {
		sdl.ReleaseGPUTexture(gpu, tex)
		return nil, false
	}
	copy(([^]u8)(dst)[:bytes], ([^]u8)(raw_data(texels))[:bytes])
	sdl.UnmapGPUTransferBuffer(gpu, transfer)
	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	if cmd == nil {
		sdl.ReleaseGPUTexture(gpu, tex)
		return nil, false
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUTexture(pass,
		{transfer_buffer = transfer, pixels_per_row = width, rows_per_layer = height},
		{texture = tex, w = width, h = height, d = 1}, false)
	sdl.EndGPUCopyPass(pass)
	if !sdl.SubmitGPUCommandBuffer(cmd) {
		sdl.ReleaseGPUTexture(gpu, tex)
		return nil, false
	}
	return tex, true
}
