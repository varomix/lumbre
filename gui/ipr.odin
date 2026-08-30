package main

// Progressive interactive render (IPR).
//
// A worker thread steps the path tracer a few samples at a time and publishes
// each result to the UI thread. The UI thread never blocks on rendering, and
// the worker never spins: when there is nothing to converge it parks on a
// condition variable, which is what keeps an idle Lumbre at ~0% CPU.
//
// All Metal work stays on the worker thread — the GPU_Renderer is created and
// used there and nowhere else.

import "core:fmt"
import "core:sync"
import "core:thread"
import "core:time"

import lc "../core"

// Samples per dispatch. Small batches keep the image responsive to camera
// changes, since a dispatch already in flight cannot be cancelled; large ones
// amortise the ~30 ms of per-render setup better. This ramps: early batches are
// small so the first image appears fast, later ones grow as the image settles.
IPR_BATCH_MIN :: 2
IPR_BATCH_MAX :: 32

IPR :: struct {
	worker:  ^thread.Thread,
	// Guards every field below it, and is paired with `cond`.
	mutex:   sync.Mutex,
	cond:    sync.Cond,

	// Worker lifetime, and whether it is allowed to render right now.
	running: bool,
	enabled: bool,

	// Bumped whenever the accumulated image stops being valid, which restarts
	// accumulation at the next batch. A batch already in flight still publishes
	// its result — see the note at the publish site.
	generation: u64,

	// The scene being rendered. Borrowed from App; `scene_mutex` below is held
	// by both the worker (while stepping) and the UI thread (while replacing
	// the scene), which is what makes the borrow safe.
	scene:      ^lc.Scene,
	scene_mutex: sync.Mutex,
	// Identifies the scene for the renderer's GPU resource cache. Bumped only
	// when the scene itself changes — never on camera moves, which is the whole
	// point: navigation must not rebuild geometry.
	scene_key:  u64,

	width, height: i32,
	target_spp:    i32,

	// Camera posted by the UI thread, applied by the worker at the start of its
	// next batch. The UI must never take scene_mutex to set the camera: the
	// worker holds that lock for the whole of a dispatch and re-acquires it
	// immediately, so an unlucky UI thread waits for the image to fully
	// converge. Measured at 2.1 s per camera change on a 157k-triangle scene
	// before this existed.
	pending_camera:     lc.Camera,
	has_pending_camera: bool,

	// Set while a dispatch is in flight, so the UI can wait for the worker to
	// park before it does something that genuinely needs exclusive access to
	// the scene, such as replacing it.
	worker_busy: bool,
	idle_cond:   sync.Cond,

	// ── published result ─────────────────────────────────────────────────────
	// Guarded by `result_mutex`. The UI thread swaps `result` out rather than
	// copying, so a 1080p handoff costs a pointer exchange.
	result_mutex:  sync.Mutex,
	result:        []u8, // RGB8, bottom-row-first (as the renderer produces it)
	result_w:      i32,
	result_h:      i32,
	result_spp:    i32,
	result_ready:  bool,

	// ── stats, read by the UI for the viewport HUD ───────────────────────────
	stat_spp:       i32,
	stat_batch_ms:  f64,
	stat_total_ms:  f64,
	stat_converged: bool,
}

ipr_init :: proc(ipr: ^IPR) {
	ipr.running = true
	ipr.enabled = true
	ipr.target_spp = 512
	ipr.width = 640
	ipr.height = 360

	ipr.worker = thread.create(ipr_worker_proc)
	ipr.worker.data = ipr
	thread.start(ipr.worker)
}

ipr_shutdown :: proc(ipr: ^IPR) {
	sync.mutex_lock(&ipr.mutex)
	ipr.running = false
	sync.cond_broadcast(&ipr.cond)
	sync.mutex_unlock(&ipr.mutex)

	if ipr.worker != nil {
		thread.join(ipr.worker)
		thread.destroy(ipr.worker)
		ipr.worker = nil
	}

	sync.mutex_lock(&ipr.result_mutex)
	delete(ipr.result)
	ipr.result = nil
	sync.mutex_unlock(&ipr.result_mutex)
}

// Discards the accumulated image and wakes the worker to start over. Every edit
// that changes what the render should look like funnels through here: camera
// moves, material edits, resolution changes, scene loads.
ipr_invalidate :: proc(ipr: ^IPR) {
	sync.mutex_lock(&ipr.mutex)
	ipr.generation += 1
	sync.cond_broadcast(&ipr.cond)
	sync.mutex_unlock(&ipr.mutex)
}

ipr_set_scene :: proc(ipr: ^IPR, scene: ^lc.Scene) {
	sync.mutex_lock(&ipr.mutex)
	ipr.scene = scene
	ipr.scene_key += 1
	ipr.generation += 1
	sync.cond_broadcast(&ipr.cond)
	sync.mutex_unlock(&ipr.mutex)
}

ipr_set_resolution :: proc(ipr: ^IPR, width, height: i32) {
	if width <= 0 || height <= 0 {
		return
	}
	sync.mutex_lock(&ipr.mutex)
	if ipr.width != width || ipr.height != height {
		ipr.width = width
		ipr.height = height
		ipr.generation += 1
		sync.cond_broadcast(&ipr.cond)
	}
	sync.mutex_unlock(&ipr.mutex)
}

// Posts a new camera and restarts accumulation. Cheap and non-blocking: it
// takes only the short state lock, never the one held across a dispatch.
ipr_set_camera :: proc(ipr: ^IPR, cam: lc.Camera) {
	sync.mutex_lock(&ipr.mutex)
	ipr.pending_camera = cam
	ipr.has_pending_camera = true
	ipr.generation += 1
	sync.cond_broadcast(&ipr.cond)
	sync.mutex_unlock(&ipr.mutex)
}

// Stops the worker and waits until it is parked. Use before touching the scene
// itself — replacing it, or freeing the old one.
ipr_pause_and_wait :: proc(ipr: ^IPR) {
	sync.mutex_lock(&ipr.mutex)
	ipr.enabled = false
	ipr.generation += 1
	for ipr.worker_busy {
		sync.cond_wait(&ipr.idle_cond, &ipr.mutex)
	}
	sync.mutex_unlock(&ipr.mutex)
}

ipr_set_enabled :: proc(ipr: ^IPR, enabled: bool) {
	sync.mutex_lock(&ipr.mutex)
	ipr.enabled = enabled
	sync.cond_broadcast(&ipr.cond)
	sync.mutex_unlock(&ipr.mutex)
}

// ── worker ───────────────────────────────────────────────────────────────────

@(private = "file")
ipr_worker_proc :: proc(t: ^thread.Thread) {
	ipr := (^IPR)(t.data)

	renderer, ok := lc.gpu_renderer_create()
	if !ok {
		fmt.eprintln("IPR: could not create GPU renderer; interactive rendering disabled")
		return
	}
	defer lc.gpu_renderer_destroy(&renderer)

	// Generation this worker is currently accumulating for, and how far it got.
	current_gen: u64 = 0
	accumulated: i32 = 0
	batch: i32 = IPR_BATCH_MIN
	total_ms: f64 = 0

	for {
		// ── decide what to do, under the lock ────────────────────────────────
		sync.mutex_lock(&ipr.mutex)
		for {
			if !ipr.running {
				sync.mutex_unlock(&ipr.mutex)
				return
			}
			// Something to render?
			has_work := ipr.enabled && ipr.scene != nil &&
				(ipr.generation != current_gen || accumulated < ipr.target_spp)
			if has_work {
				break
			}
			// Nothing to do: park. This is what stops the worker from spinning
			// while the image is converged and the user is idle.
			sync.cond_wait(&ipr.cond, &ipr.mutex)
		}

		gen := ipr.generation
		scene_key := ipr.scene_key
		scene := ipr.scene
		width := ipr.width
		height := ipr.height
		target := ipr.target_spp

		pending_cam := ipr.pending_camera
		apply_cam := ipr.has_pending_camera
		ipr.has_pending_camera = false

		ipr.worker_busy = true
		sync.mutex_unlock(&ipr.mutex)

		// A new generation restarts accumulation from scratch.
		if gen != current_gen {
			current_gen = gen
			accumulated = 0
			batch = IPR_BATCH_MIN
			total_ms = 0
			lc.gpu_renderer_reset_accum(&renderer)
		}

		if accumulated >= target {
			continue
		}
		this_batch := min(batch, target - accumulated)

		// ── render ───────────────────────────────────────────────────────────
		start := time.tick_now()
		sync.mutex_lock(&ipr.scene_mutex)
		if apply_cam {
			scene.camera = pending_cam
		}
		frame := lc.gpu_render_frame(
			scene, width, height,
			this_batch, 12, 1000.0,
			0, 0.95, 0.0,
			// Biased GI is on for interactive rendering: the irradiance cache
			// and photon map live in the renderer's scene cache now, so they
			// are built once and reused across batches and camera moves rather
			// than rebuilt per dispatch. This is what makes the viewport match
			// what a final render produces.
			true, 0, 0.5,
			true, 200000, 0, 8,
			false, false, false, false,
			&renderer, accumulated, scene_key,
		)
		sync.mutex_unlock(&ipr.scene_mutex)
		batch_ms := time.duration_milliseconds(time.tick_since(start))

		sync.mutex_lock(&ipr.mutex)
		ipr.worker_busy = false
		sync.cond_broadcast(&ipr.idle_cond)
		sync.mutex_unlock(&ipr.mutex)

		if frame.pixels == nil {
			// Render failed (no geometry, device trouble). Do not spin on it.
			sync.mutex_lock(&ipr.mutex)
			ipr.enabled = false
			sync.mutex_unlock(&ipr.mutex)
			continue
		}

		accumulated += this_batch
		total_ms += batch_ms

		// Publish even if the generation moved on while this batch was in
		// flight. The result is a consistent image of the camera as it was when
		// the dispatch started, which is exactly what an interactive viewport
		// should show. Discarding it instead meant that during a drag — where
		// the generation bumps on every mouse-move — practically every batch
		// was thrown away and the viewport appeared frozen until the mouse
		// stopped.
		sync.mutex_lock(&ipr.result_mutex)
		delete(ipr.result)
		ipr.result = frame.pixels // ownership moves to the IPR
		ipr.result_w = frame.width
		ipr.result_h = frame.height
		ipr.result_spp = accumulated
		ipr.result_ready = true
		sync.mutex_unlock(&ipr.result_mutex)

		frame.pixels = nil // do not free what we just handed over
		lc.destroy_gpu_frame(&frame)

		sync.mutex_lock(&ipr.mutex)
		ipr.stat_spp = accumulated
		ipr.stat_batch_ms = batch_ms
		ipr.stat_total_ms = total_ms
		ipr.stat_converged = accumulated >= target
		sync.mutex_unlock(&ipr.mutex)

		// Grow the batch as the image settles: quick first light, then fewer
		// larger dispatches so the fixed per-render cost is amortised.
		if batch < IPR_BATCH_MAX {
			batch = min(batch * 2, IPR_BATCH_MAX)
		}

		// Bring the UI thread back if it is parked in WaitEvent.
		request_wake()
	}
}

// Hands the newest result to the caller, if there is one. Returns the buffer
// the IPR was holding and takes ownership of `recycle` in exchange, so a 1080p
// handoff is a pointer swap rather than a copy.
ipr_take_result :: proc(
	ipr: ^IPR,
	recycle: []u8,
) -> (
	pixels: []u8,
	width, height, spp: i32,
	got: bool,
) {
	sync.mutex_lock(&ipr.result_mutex)
	defer sync.mutex_unlock(&ipr.result_mutex)

	if !ipr.result_ready {
		return nil, 0, 0, 0, false
	}

	pixels = ipr.result
	width = ipr.result_w
	height = ipr.result_h
	spp = ipr.result_spp

	ipr.result = recycle
	ipr.result_ready = false
	return pixels, width, height, spp, true
}

IPR_Stats :: struct {
	spp:       i32,
	target:    i32,
	batch_ms:  f64,
	total_ms:  f64,
	converged: bool,
	enabled:   bool,
}

ipr_stats :: proc(ipr: ^IPR) -> IPR_Stats {
	sync.mutex_lock(&ipr.mutex)
	defer sync.mutex_unlock(&ipr.mutex)
	return IPR_Stats {
		spp = ipr.stat_spp,
		target = ipr.target_spp,
		batch_ms = ipr.stat_batch_ms,
		total_ms = ipr.stat_total_ms,
		converged = ipr.stat_converged,
		enabled = ipr.enabled,
	}
}
