package lumbre_core

// Persistent Metal state for the GPU path tracer.
//
// `gpu_render_frame` used to create the Metal device, compile
// `shaders/raytrace.metal` from source, and build all four compute pipelines on
// every call. This holds that scene-independent setup so a frontend can pay it
// once per session instead.
//
// Measured on an M3 Max at 400x300, so the cost is not mis-attributed later:
// the first render in a process takes ~76 ms and subsequent ones ~24 ms, even
// with a throwaway renderer, because Metal caches the compiled library within
// the process. So this saves roughly 50 ms once at startup — it does NOT save
// per render.
//
// The per-render cost that remains is ~12 ms fixed plus ~1.4 ms per sample.
// That fixed 12 ms is scene flattening, buffer upload, acceleration-structure
// build, and readback — none of which belong here yet. Moving those onto this
// struct is what actually pays off for the interactive viewport, where a 4-spp
// batch is ~5.6 ms of useful work against that same 12 ms of setup.
//
// Frontends that do not care (the CLI, the Houdini bridge) can keep calling
// `gpu_render_frame` with no renderer and get the old behaviour: it creates a
// temporary one internally.

import "core:fmt"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

GPU_Renderer :: struct {
	device: ^MTL.Device,
	queue:  ^MTL.CommandQueue,
	library: ^MTL.Library,

	// The main path-tracing kernel, plus the three photon-map passes.
	pipeline:                ^MTL.ComputePipelineState,
	photon_emit_pipeline:    ^MTL.ComputePipelineState,
	photon_count_pipeline:   ^MTL.ComputePipelineState,
	photon_scatter_pipeline: ^MTL.ComputePipelineState,
}

// Creates the device, compiles the kernel, and builds every pipeline. Returns
// ok = false with a message already printed if anything fails; callers must not
// use the renderer in that case.
gpu_renderer_create :: proc() -> (r: GPU_Renderer, ok: bool) {
	r.device = MTL.CreateSystemDefaultDevice()
	if r.device == nil {
		fmt.eprintln("Metal device required")
		return {}, false
	}
	if !bool(r.device->supportsRaytracing()) {
		fmt.eprintln("Raytracing required")
		return {}, false
	}
	fmt.println("Device:", r.device->name()->odinString())

	r.queue = r.device->newCommandQueue()

	msl_source := #load("shaders/raytrace.metal", string)
	src := NS.String.alloc()->initWithOdinString(msl_source)
	opts := MTL.CompileOptions.alloc()->init()
	opts->setFastMathEnabled(true)
	opts->setLanguageVersion(.Version3_0)

	library, err := r.device->newLibraryWithSource(src, opts)
	if err != nil {
		fmt.eprintln("Shader compilation failed:", err->localizedDescription()->odinString())
		return {}, false
	}
	r.library = library

	r.pipeline = gpu_make_pipeline(r.device, library, "raytraceKernel") or_return
	r.photon_emit_pipeline = gpu_make_pipeline(r.device, library, "photonEmitKernel") or_return
	r.photon_count_pipeline = gpu_make_pipeline(r.device, library, "photonCountKernel") or_return
	r.photon_scatter_pipeline = gpu_make_pipeline(r.device, library, "photonScatterKernel") or_return

	return r, true
}

@(private = "file")
gpu_make_pipeline :: proc(
	device: ^MTL.Device,
	library: ^MTL.Library,
	name: string,
) -> (
	^MTL.ComputePipelineState,
	bool,
) {
	// NS.AT caches a compile-time literal, so it cannot take a runtime name.
	ns_name := NS.String.alloc()->initWithOdinString(name)
	defer ns_name->release()

	fn := library->newFunctionWithName(ns_name)
	if fn == nil {
		fmt.eprintln("Kernel function not found:", name)
		return nil, false
	}

	desc := MTL.ComputePipelineDescriptor.alloc()->init()
	desc->setComputeFunction(fn)

	pipeline, err := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
		device, desc, MTL.PipelineOption{}, nil,
	)
	if err != nil {
		fmt.eprintln("Pipeline creation failed for", name, "-", err->localizedDescription()->odinString())
		return nil, false
	}
	return pipeline, true
}

gpu_renderer_destroy :: proc(r: ^GPU_Renderer) {
	// Metal objects here are reference-counted Objective-C instances owned by
	// the autorelease machinery; dropping the handles is all that is required.
	r^ = {}
}
