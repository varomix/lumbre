package lumbre_realtime

// Image-based lighting from the scene's environment.
//
// Three derived maps, all built once when the scene is uploaded:
//
//   irradiance  — the diffuse term: the environment convolved with a cosine
//                 lobe, so a lookup along the normal is the whole hemisphere's
//                 contribution.
//   specular    — the environment pre-convolved with a GGX lobe, one mip per
//                 roughness, sampled along the reflection vector.
//   brdf_lut    — the second half of the split-sum approximation. Depends only
//                 on roughness and viewing angle, never on the environment, so
//                 it is built once at startup.
//
// Rotation and intensity are applied at LOOKUP time, not baked into the
// prefiltered maps. A rotation about Y and a scale both commute with the
// convolution, so prefiltering the raw environment and transforming the lookup
// direction is exactly equivalent — and it means turning the HDRI or changing
// its intensity costs nothing, where baking would rebuild all three maps.
//
// The path tracer's `env_lookup` (shaders/raytrace.metal) defines the mapping
// this must agree with: rotate by -rotation about Y, theta from acos(d.y), phi
// from atan2(d.z, d.x), u = (phi + PI) / 2PI, v = theta / PI. Row v=0 is the +Y
// pole. `equirect_direction` in shaders/common.slang is the inverse.

import "core:fmt"
import "core:math"

import lc "../core"
import sdl "vendor:sdl3"

// The environment the rasterizer samples. Equirectangular like the path
// tracer's, rather than a cubemap: it matches how `core.Environment` already
// stores an HDRI, so there is no face-splitting step and no second convention
// to keep in agreement.
ENV_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT
LUT_FORMAT :: sdl.GPUTextureFormat.R16G16_FLOAT

// Resolution of the derived maps. The diffuse term is almost constant across a
// hemisphere, so its map can be tiny; the specular one needs enough base
// resolution to hold a sharp reflection at roughness 0.
IRRADIANCE_WIDTH :: 64
IRRADIANCE_HEIGHT :: 32
SPECULAR_WIDTH :: 256
SPECULAR_HEIGHT :: 128
SPECULAR_MIPS :: 6
BRDF_LUT_SIZE :: 128

// Fallback when a scene has no HDRI: the same procedural sky the path tracer
// draws, baked to an equirect so the prefilter has one input format to handle.
SKY_WIDTH :: 128
SKY_HEIGHT :: 64

Environment_GPU :: struct {
	source:     ^sdl.GPUTexture, // the raw environment, equirect
	irradiance: ^sdl.GPUTexture,
	specular:   ^sdl.GPUTexture,
	sampler:    ^sdl.GPUSampler,
	// Y rotation in radians and a linear scale, both applied at lookup.
	rotation:   f32,
	intensity:  f32,
	has_env:    bool,
}

// Builds the equirect source for a scene: its HDRI, or the procedural sky, or
// nothing when the sky is explicitly hidden.
//
// Returns RGBA16F-ready f32 data; the caller converts. `core.Environment`
// stores three floats per texel, and a GPU texture wants four.
env_source_pixels :: proc(
	scene: ^lc.Scene,
	hide_default_sky: bool,
) -> (
	pixels: [][4]f32,
	width, height: i32,
	ok: bool,
) {
	env := scene.environment

	if env.has_data && env.width > 0 && env.height > 0 && len(env.pixels) >= int(env.width * env.height) * 3 {
		width, height = env.width, env.height
		pixels = make([][4]f32, int(width) * int(height))
		for i in 0 ..< len(pixels) {
			pixels[i] = {env.pixels[i * 3], env.pixels[i * 3 + 1], env.pixels[i * 3 + 2], 1}
		}
		return pixels, width, height, true
	}

	if hide_default_sky {
		return nil, 0, 0, false
	}

	// The path tracer's default sky: a vertical lerp from white to a pale blue.
	// Baked here rather than evaluated in the shader so the prefilter and the
	// background have exactly one input.
	width, height = SKY_WIDTH, SKY_HEIGHT
	pixels = make([][4]f32, int(width) * int(height))
	for y in 0 ..< int(height) {
		// v = 0 is the +Y pole, matching the path tracer's row order.
		theta := (f32(y) + 0.5) / f32(height) * math.PI
		dir_y := math.cos(theta)
		t := 0.5 * (dir_y + 1.0)
		r := (1.0 - t) * 1.0 + t * 0.5
		g := (1.0 - t) * 1.0 + t * 0.7
		b := (1.0 - t) * 1.0 + t * 1.0
		for x in 0 ..< int(width) {
			pixels[y * int(width) + x] = {r, g, b, 1}
		}
	}
	return pixels, width, height, true
}

// Uploads the equirect source. Mips are generated so the prefilter passes can
// sample a lower level for rough lobes instead of undersampling a 4K map.
env_upload_source :: proc(
	gpu: ^sdl.GPUDevice,
	pixels: [][4]f32,
	width, height: i32,
) -> ^sdl.GPUTexture {
	halves := make([][4]u16, len(pixels))
	defer delete(halves)
	for p, i in pixels {
		halves[i] = {f32_to_f16(p.x), f32_to_f16(p.y), f32_to_f16(p.z), f32_to_f16(p.w)}
	}

	levels := mip_levels(width, height)
	tex := sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = .D2,
			format = ENV_FORMAT,
			usage = {.SAMPLER, .COLOR_TARGET},
			width = u32(width),
			height = u32(height),
			layer_count_or_depth = 1,
			num_levels = levels,
			sample_count = ._1,
		},
	)
	if tex == nil {
		fmt.eprintln("realtime: environment texture failed:", sdl.GetError())
		return nil
	}

	byte_count := u32(len(halves) * size_of([4]u16))
	transfer := sdl.CreateGPUTransferBuffer(
		gpu,
		sdl.GPUTransferBufferCreateInfo{usage = .UPLOAD, size = byte_count},
	)
	if transfer == nil {
		sdl.ReleaseGPUTexture(gpu, tex)
		return nil
	}
	defer sdl.ReleaseGPUTransferBuffer(gpu, transfer)

	dst := sdl.MapGPUTransferBuffer(gpu, transfer, false)
	if dst == nil {
		sdl.ReleaseGPUTexture(gpu, tex)
		return nil
	}
	copy(([^]u8)(dst)[:byte_count], (([^]u8)(raw_data(halves)))[:byte_count])
	sdl.UnmapGPUTransferBuffer(gpu, transfer)

	cmd := sdl.AcquireGPUCommandBuffer(gpu)
	if cmd == nil {
		sdl.ReleaseGPUTexture(gpu, tex)
		return nil
	}
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUTexture(
		pass,
		sdl.GPUTextureTransferInfo {
			transfer_buffer = transfer,
			pixels_per_row = u32(width),
			rows_per_layer = u32(height),
		},
		sdl.GPUTextureRegion{texture = tex, w = u32(width), h = u32(height), d = 1},
		false,
	)
	sdl.EndGPUCopyPass(pass)
	if levels > 1 {
		sdl.GenerateMipmapsForGPUTexture(cmd, tex)
	}
	_ = sdl.SubmitGPUCommandBuffer(cmd)
	return tex
}

env_destroy :: proc(gpu: ^sdl.GPUDevice, e: ^Environment_GPU) {
	if gpu == nil {
		return
	}
	for tex in ([]^sdl.GPUTexture{e.source, e.irradiance, e.specular}) {
		if tex != nil {
			sdl.ReleaseGPUTexture(gpu, tex)
		}
	}
	if e.sampler != nil {
		sdl.ReleaseGPUSampler(gpu, e.sampler)
	}
	e^ = {}
}

// IEEE 754 binary32 to binary16. Handles the ranges an HDRI actually contains:
// normals, subnormals flushed to zero, and overflow clamped to the largest
// finite half rather than becoming infinity, which would poison the prefilter.
@(private = "file")
f32_to_f16 :: proc(f: f32) -> u16 {
	bits := transmute(u32)f
	sign := u16((bits >> 16) & 0x8000)
	exponent := i32((bits >> 23) & 0xFF) - 127 + 15
	mantissa := bits & 0x7FFFFF

	if exponent <= 0 {
		return sign // underflow: flush to signed zero
	}
	if exponent >= 0x1F {
		return sign | 0x7BFF // clamp to the largest finite half
	}
	return sign | u16(exponent << 10) | u16(mantissa >> 13)
}
