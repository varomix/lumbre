package lumbre_realtime

// Per-pass GPU timing, approximately.
//
// SDL_GPU has no timestamp queries -- fences are the only completion signal it
// exposes -- so this measures by serialising: with `Renderer.profile` set, the
// frame is cut into one command buffer per pass, and each is submitted and
// waited for before the next is recorded. A pass's time is the wall clock from
// the previous wait returning to its own wait returning, which covers
// recording it on the CPU, submitting it, and running it on an otherwise idle
// GPU. The cost of a round trip with nothing in it is measured each frame and
// subtracted.
//
// What this does NOT measure: the overlap a real frame gets between passes, or
// between CPU recording and GPU work. Profiled frames are therefore slower in
// total than unprofiled ones, and the per-pass split is the useful part --
// which pass dominates, and whether a change made it cheaper. For exact
// numbers, the passes carry debug groups for Xcode's GPU frame capture or
// RenderDoc.

import "core:fmt"
import "core:time"

import sdl "vendor:sdl3"

Pass :: enum {
	Clusters, // froxel light lists, uploaded
	Shadows,  // sun cascades
	Labels,   // ground-truth ids, when a label view shows
	GBuffer,
	AO,
	Lighting,
	SSGI,     // screen-space indirect diffuse
	HiZ,      // depth pyramid for reflections
	SSR,
	Forward,  // transmissive surfaces
	Fog,
	TAA,
	Display,  // exposure and view transform
	FXAA,
	Debug,    // a G-buffer channel view
}

Profiler :: struct {
	last_frame:  [Pass]f32, // ms, the most recent profiled frame
	total:       [Pass]f64, // ms, summed over `frames`
	frames:      u64,
	overhead_ms: f32, // an empty submit-and-wait, this frame
	mark:        time.Tick,
	current:     [Pass]f32,
}

// Clears the accumulated totals; the last frame's numbers stay readable.
profile_reset :: proc(r: ^Renderer) {
	r.profiler.total = {}
	r.profiler.frames = 0
}

@(private)
profile_begin :: proc(r: ^Renderer) {
	if !r.profile { return }
	p := &r.profiler
	p.current = {}
	// The best of three empty round trips: the fixed cost each split pays
	// whatever the pass did.
	best := max(f32)
	for _ in 0 ..< 3 {
		start := time.tick_now()
		if !submit_and_wait(r.gpu, sdl.AcquireGPUCommandBuffer(r.gpu)) {
			best = 0
			break
		}
		best = min(best, f32(time.duration_milliseconds(time.tick_since(start))))
	}
	p.overhead_ms = best
	p.mark = time.tick_now()
}

// Submits what `cmd` holds, waits for it, charges the time to `pass`, and
// replaces `cmd` with a fresh command buffer for the next pass.
@(private)
profile_split :: proc(r: ^Renderer, cmd: ^^sdl.GPUCommandBuffer, pass: Pass) -> bool {
	p := &r.profiler
	if !submit_and_wait(r.gpu, cmd^) {
		cmd^ = nil
		fmt.eprintln("realtime: profiled submit failed:", sdl.GetError())
		return false
	}
	now := time.tick_now()
	ms := f32(time.duration_milliseconds(time.tick_diff(p.mark, now))) - p.overhead_ms
	p.current[pass] += max(ms, 0)
	p.mark = now
	cmd^ = sdl.AcquireGPUCommandBuffer(r.gpu)
	return cmd^ != nil
}

@(private)
profile_end :: proc(r: ^Renderer) {
	if !r.profile { return }
	p := &r.profiler
	p.last_frame = p.current
	for ms, pass in p.current {
		p.total[pass] += f64(ms)
	}
	p.frames += 1
}

// Prints each pass's mean time over the profiled frames, skipping passes that
// never ran.
profile_print :: proc(r: ^Renderer, title: string) {
	p := &r.profiler
	if p.frames == 0 { return }
	fmt.printfln("%s: GPU pass times over %d frame(s), ms (approximate; see realtime/profile.odin)", title, p.frames)
	sum: f64
	for total, pass in p.total {
		if total <= 0 { continue }
		mean := total / f64(p.frames)
		sum += mean
		fmt.printfln("  %-9v %8.3f", pass, mean)
	}
	fmt.printfln("  %-9s %8.3f", "total", sum)
}

@(private = "file")
submit_and_wait :: proc(gpu: ^sdl.GPUDevice, cmd: ^sdl.GPUCommandBuffer) -> bool {
	if cmd == nil { return false }
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	if fence == nil { return false }
	defer sdl.ReleaseGPUFence(gpu, fence)
	return bool(sdl.WaitForGPUFences(gpu, true, &fence, 1))
}
