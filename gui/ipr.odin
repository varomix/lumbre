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
// amortise the per-render setup better.
//
// The batch is sized by *time*, not by a fixed sample count, because how long
// 32 samples take varies by two orders of magnitude across scenes: a 15k-tri
// helmet under an HDRI renders 32 spp in 0.10 s, while a 32-triangle Cornell
// box — closed, so every ray bounces to max depth through the irradiance cache
// — takes 1.3 s for the same 32. A fixed count made the second case update the
// viewport less than once a second and stall every camera move for a full
// dispatch. Aiming at a wall-clock target instead keeps both feeling the same,
// and costs only the setup overhead of the extra dispatches.
IPR_BATCH_MIN :: 2
IPR_BATCH_MAX :: 64
IPR_BATCH_TARGET_MS :: 200.0

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
	max_depth:     i32,
	// Denoise the viewport with OIDN. When set, each batch also renders the
	// albedo/normal AOVs the denoiser needs; see the worker's dispatch below.
	denoise:       bool,
	// Snapshot of the settings the worker renders with. Copied under the state
	// lock so the UI can edit App.core.settings freely without racing a
	// dispatch.
	render_settings: lc.Render_Config,

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

	// Material edits. Guarded by its own short lock rather than scene_mutex,
	// for the same reason the camera is: scene_mutex is held across a whole
	// dispatch, so taking it from the UI stalls slider drags for seconds.
	material_mutex:  sync.Mutex,
	materials_dirty: bool,
	lights_dirty:    bool,
	// Set when an edit has *settled* (the slider was released, or the change
	// came in whole from a script or a look load). Only then does the worker
	// re-emit the photon map, which the new lighting invalidates but which
	// costs a blocking GPU pass to replace. Mid-drag edits leave it standing:
	// the image is briefly lit by the old bounce data, and it settles the
	// moment the mouse comes up.
	photons_dirty:   bool,
	// Stronger, and rarer: also drop the irradiance cache. That cache is
	// gathered over many batches rather than built in one pass, so dropping it
	// sends the viewport back to looking unconverged — worth it when the
	// gather settings themselves changed, not for an ordinary relight.
	gi_dirty:        bool,

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
	ipr.target_spp = 64
	ipr.max_depth = 4
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

// Marks the scene's materials as edited. The worker rewrites just the material
// buffer before its next batch — a full cache rebuild would take ~0.6 s on a
// large scene and make dragging a slider useless.
//
// `settled` says whether this is the final value of an edit rather than one
// frame of a drag; see `gi_dirty`. Callers that deliver a change all at once —
// a script, a look load — take the default.
ipr_materials_changed :: proc(ipr: ^IPR, settled := true) {
	sync.mutex_lock(&ipr.material_mutex)
	ipr.materials_dirty = true
	ipr.photons_dirty |= settled
	sync.mutex_unlock(&ipr.material_mutex)

	sync.mutex_lock(&ipr.mutex)
	ipr.generation += 1
	sync.cond_broadcast(&ipr.cond)
	sync.mutex_unlock(&ipr.mutex)
}

// Drops both bounce-lighting caches — the photon map and the irradiance cache —
// without touching geometry or the material buffer. For settings that change
// how those caches are gathered rather than what the scene contains, which is
// the one case where the cached irradiance is answering the wrong question.
ipr_gi_changed :: proc(ipr: ^IPR) {
	sync.mutex_lock(&ipr.material_mutex)
	ipr.gi_dirty = true
	sync.mutex_unlock(&ipr.material_mutex)

	sync.mutex_lock(&ipr.mutex)
	ipr.generation += 1
	sync.cond_broadcast(&ipr.cond)
	sync.mutex_unlock(&ipr.mutex)
}

// Raising the target simply lets the worker keep converging; it does not
// invalidate what has already accumulated.
ipr_set_target_spp :: proc(ipr: ^IPR, target: i32) {
	sync.mutex_lock(&ipr.mutex)
	ipr.target_spp = max(target, 1)
	sync.cond_broadcast(&ipr.cond)
	sync.mutex_unlock(&ipr.mutex)
}

// Marks the scene's analytic lights as edited; the worker rewrites just the
// light buffers before its next batch. `settled` works as it does for
// ipr_materials_changed.
ipr_lights_changed :: proc(ipr: ^IPR, settled := true) {
	sync.mutex_lock(&ipr.material_mutex)
	ipr.lights_dirty = true
	ipr.photons_dirty |= settled
	sync.mutex_unlock(&ipr.material_mutex)

	sync.mutex_lock(&ipr.mutex)
	ipr.generation += 1
	sync.cond_broadcast(&ipr.cond)
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
		cfg := ipr.render_settings
		max_depth := ipr.max_depth
		denoise := ipr.denoise

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

		// Denoise only the batch that reaches the target, not the noisy
		// intermediates on the way there. OIDN plus its guide passes cost more
		// than a plain batch, so denoising every batch would make a denoised
		// converge slower than an un-denoised one — the opposite of the point.
		// This way convergence costs one extra denoise at the end; lower the
		// target spp to trade raw samples for the denoiser and it becomes a net
		// speed-up.
		want_denoise := denoise && accumulated + this_batch >= target

		// ── render ───────────────────────────────────────────────────────────
		start := time.tick_now()
		sync.mutex_lock(&ipr.scene_mutex)
		if apply_cam {
			scene.camera = pending_cam
		}

		sync.mutex_lock(&ipr.material_mutex)
		mats_dirty := ipr.materials_dirty
		lights_dirty := ipr.lights_dirty
		photons_dirty := ipr.photons_dirty
		gi_dirty := ipr.gi_dirty
		ipr.materials_dirty = false
		ipr.lights_dirty = false
		ipr.photons_dirty = false
		ipr.gi_dirty = false
		sync.mutex_unlock(&ipr.material_mutex)
		if lights_dirty {
			// Falls back to a full rebuild when the per-kind light counts
			// changed, which the buffers cannot absorb in place.
			if !lc.gpu_scene_cache_update_lights(&renderer, scene) {
				sync.mutex_lock(&ipr.mutex)
				ipr.scene_key += 1
				scene_key = ipr.scene_key
				sync.mutex_unlock(&ipr.mutex)
			}
		}
		if mats_dirty {
			// Ignored when no cache exists yet; the pending build reads the
			// current materials regardless.
			_ = lc.gpu_scene_cache_update_materials(&renderer, scene)
		}
		// The irradiance cache survives an ordinary relight. It is gathered
		// over many batches, so dropping it costs the viewport its converged
		// look for seconds — far more than a slightly stale bounce term is
		// worth, and the same trade the renderer made before any of this was
		// invalidated at all.
		switch {
		case gi_dirty:      lc.gpu_scene_cache_reset_gi(&renderer)
		case photons_dirty: lc.gpu_scene_cache_reset_photons(&renderer)
		}

		frame := lc.gpu_render_frame(
			scene, width, height,
			this_batch, max_depth, cfg.max_radiance,
			cfg.debug_mode, cfg.roughness_cutoff, cfg.glossy_bias,
			// Biased GI is on for interactive rendering: the irradiance cache
			// and photon map live in the renderer's scene cache now, so they
			// are built once and reused across batches and camera moves rather
			// than rebuilt per dispatch. This is what makes the viewport match
			// what a final render produces.
			cfg.gi_cache_enabled, cfg.gi_cache_distance, cfg.gi_cache_normal_angle,
			cfg.photon_enabled, cfg.photon_count, cfg.photon_radius, cfg.photon_bounces,
			false, false, b32(want_denoise), false,
			&renderer, accumulated, scene_key,
			// The viewport shows the 8-bit sRGB image; the float copy would be
			// a per-batch allocation and full-image copy for nothing.
			want_linear = false,
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

		batch = ipr_next_batch(batch, this_batch, batch_ms)

		// Bring the UI thread back if it is parked in WaitEvent.
		request_wake()
	}
}

// Sizes the next dispatch from how long the last one actually took, aiming at
// IPR_BATCH_TARGET_MS. Starting from IPR_BATCH_MIN this still ramps up quickly
// on a cheap scene, so first light arrives as fast as it used to.
//
// The step is capped at half or double the current batch so that one unusually
// slow dispatch — a photon map rebuild landing in the same batch, say — nudges
// the size rather than collapsing it.
@(private = "file")
ipr_next_batch :: proc(batch, rendered: i32, ms: f64) -> i32 {
	if ms <= 0 || rendered <= 0 {
		return min(batch * 2, IPR_BATCH_MAX)
	}
	ideal := f64(rendered) * IPR_BATCH_TARGET_MS / ms
	next := i32(ideal + 0.5)
	next = clamp(next, max(batch / 2, 1), max(batch * 2, 1))
	return clamp(next, IPR_BATCH_MIN, IPR_BATCH_MAX)
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
