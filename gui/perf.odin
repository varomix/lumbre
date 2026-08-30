package main

// UI-thread frame profiler.
//
// Batch render time is not the thing that makes navigation feel bad — what
// matters is how long the UI thread is unavailable. This records each stage of
// a frame separately so a stall can be attributed rather than guessed at.
//
// Enabled with --nav-bench, which also drives the camera automatically so the
// measurement is reproducible without a human dragging a mouse.

import "core:fmt"
import "core:sort"
import "core:time"

Perf_Stage :: enum {
	Wait,          // blocked in SDL waiting for events
	Events,        // dispatching SDL/ImGui events
	CameraApply,   // app_apply_camera — includes any wait on scene_mutex
	ViewportPull,  // taking a finished batch and uploading it as a texture
	DrawFrame,     // building and submitting the ImGui frame
	FrameTotal,    // everything above, per loop iteration
}

PERF_MAX_SAMPLES :: 20000

Perf :: struct {
	enabled: bool,
	samples: [Perf_Stage][dynamic]f64, // milliseconds
}

perf_record :: proc(p: ^Perf, stage: Perf_Stage, start: time.Tick) {
	if !p.enabled {
		return
	}
	ms := time.duration_milliseconds(time.tick_since(start))
	if len(p.samples[stage]) < PERF_MAX_SAMPLES {
		append(&p.samples[stage], ms)
	}
}

perf_destroy :: proc(p: ^Perf) {
	for stage in Perf_Stage {
		delete(p.samples[stage])
	}
}

@(private = "file")
percentile :: proc(sorted: []f64, q: f64) -> f64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(q * f64(len(sorted) - 1) + 0.5)
	return sorted[clamp(idx, 0, len(sorted) - 1)]
}

perf_report :: proc(p: ^Perf, header: string) {
	fmt.println()
	fmt.println("=====================================================================")
	fmt.println(header)
	fmt.println("=====================================================================")
	fmt.printfln(
		"%-14s %7s %9s %9s %9s %9s %9s",
		"stage", "n", "mean", "p50", "p95", "p99", "max",
	)

	for stage in Perf_Stage {
		s := p.samples[stage]
		if len(s) == 0 {
			continue
		}
		sorted := make([]f64, len(s))
		defer delete(sorted)
		copy(sorted, s[:])
		sort.quick_sort(sorted)

		total: f64 = 0
		for v in sorted {
			total += v
		}
		fmt.printfln(
			"%-14s %7d %8.2fms %8.2fms %8.2fms %8.2fms %8.2fms",
			stage,
			len(sorted),
			total / f64(len(sorted)),
			percentile(sorted, 0.50),
			percentile(sorted, 0.95),
			percentile(sorted, 0.99),
			sorted[len(sorted) - 1],
		)
	}
	fmt.println("=====================================================================")

	// The number that actually describes how navigation feels: how often the UI
	// thread completed a frame while the camera was being driven every frame.
	ft := p.samples[.FrameTotal]
	if len(ft) > 0 {
		total: f64 = 0
		for v in ft {
			total += v
		}
		mean := total / f64(len(ft))
		fmt.printfln(
			"interactive frame rate: %.1f fps (mean %.2f ms per UI frame)",
			1000.0 / max(mean, 0.001),
			mean,
		)
	}
	fmt.println()
}
