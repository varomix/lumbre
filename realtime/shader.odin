package lumbre_realtime

// Shader loading for SDL_GPU.
//
// Every shader ships pre-compiled in both formats SDL can consume on the
// platforms we target — MSL for Metal, SPIR-V for Vulkan — and the right one is
// picked at runtime from what the device reports. That is the entire reason
// this renderer is portable without a hand-written RHI: SDL owns the backend
// and Slang owns the shading language, so neither is our problem.
//
// The blobs come from `scripts/build_shaders.sh` and are committed, so a plain
// `odin build gui` never needs Slang. See that script for the source list.

import "core:fmt"
import "core:strings"

import sdl "vendor:sdl3"

// A single entry point, compiled to every format. `msl` is source text (SDL
// compiles it when creating the shader); `spv` is a binary module.
Shader_Blob :: struct {
	msl: string,
	spv: []u8,
}

// Resource counts SDL needs up front — it cannot introspect the blob, so these
// are declared per shader at the call site and must match what the Slang source
// actually binds.
Shader_Bindings :: struct {
	samplers:         u32,
	storage_textures: u32,
	storage_buffers:  u32,
	uniform_buffers:  u32,
}

shader_create :: proc(
	gpu: ^sdl.GPUDevice,
	blob: Shader_Blob,
	entry: string,
	stage: sdl.GPUShaderStage,
	bindings: Shader_Bindings = {},
) -> ^sdl.GPUShader {
	formats := sdl.GetGPUShaderFormats(gpu)

	info := sdl.GPUShaderCreateInfo {
		stage                = stage,
		num_samplers         = bindings.samplers,
		num_storage_textures = bindings.storage_textures,
		num_storage_buffers  = bindings.storage_buffers,
		num_uniform_buffers  = bindings.uniform_buffers,
	}

	// Slang names the Metal entry point exactly as written in the source, so
	// the same string serves both formats.
	c_entry := strings.clone_to_cstring(entry, context.temp_allocator)
	info.entrypoint = c_entry

	switch {
	case .MSL in formats:
		// SDL reads MSL as a NUL-terminated C string, and `code_size` must
		// include that terminator.
		src := strings.clone_to_cstring(blob.msl, context.temp_allocator)
		info.format = {.MSL}
		info.code = ([^]u8)(rawptr(src))
		info.code_size = uint(len(blob.msl) + 1)
	case .SPIRV in formats:
		info.format = {.SPIRV}
		info.code = raw_data(blob.spv)
		info.code_size = uint(len(blob.spv))
	case:
		fmt.eprintln("realtime: no supported shader format on this GPU device")
		return nil
	}

	shader := sdl.CreateGPUShader(gpu, info)
	if shader == nil {
		fmt.eprintln("realtime: CreateGPUShader failed for", entry, "-", sdl.GetError())
	}
	return shader
}

// ── The compiled blobs ───────────────────────────────────────────────────────
// Listed by hand because `#load` takes a literal path. Keep in step with the
// `shaders` array in scripts/build_shaders.sh.

SHADER_GBUFFER_VS :: Shader_Blob {
	msl = #load("shaders/build/gbuffer_vs.vertex.msl", string),
	spv = #load("shaders/build/gbuffer_vs.vertex.spv", []u8),
}

SHADER_GBUFFER_FS :: Shader_Blob {
	msl = #load("shaders/build/gbuffer_fs.fragment.msl", string),
	spv = #load("shaders/build/gbuffer_fs.fragment.spv", []u8),
}

SHADER_FULLSCREEN_VS :: Shader_Blob {
	msl = #load("shaders/build/fullscreen_vs.vertex.msl", string),
	spv = #load("shaders/build/fullscreen_vs.vertex.spv", []u8),
}

SHADER_DEBUG_FS :: Shader_Blob {
	msl = #load("shaders/build/debug_fs.fragment.msl", string),
	spv = #load("shaders/build/debug_fs.fragment.spv", []u8),
}

SHADER_LIGHTING_FS :: Shader_Blob {
	msl = #load("shaders/build/lighting_fs.fragment.msl", string),
	spv = #load("shaders/build/lighting_fs.fragment.spv", []u8),
}
