package main

// Offline "Render to File".
//
// Runs on its own thread with its own GPU_Renderer, so a final render neither
// disturbs the viewport's scene cache nor blocks the UI. It renders in batches
// and accumulates, which is what makes progress reporting and cancellation
// possible at all — a single large dispatch could report neither.
//
// The file is written through output.write_gpu_frame, the same path the CLI
// uses, so a render started here is byte-identical to the same render from the
// command line.

import "core:fmt"
import "core:strings"
import "core:sync"
import "core:thread"
import "core:time"

import lc "../core"
import "../output"

// Batch size for an offline render. Larger than the viewport's, because
// responsiveness here only means progress updates, not interactivity.
RENDER_JOB_BATCH :: 32

Render_Job :: struct {
	worker: ^thread.Thread,
	mutex:  sync.Mutex,

	running:   bool,
	cancel:    bool,
	done:      bool,
	progress:  f64, // 0..1
	status:    string,
	elapsed:   f64,

	// Request. Copied under the lock before the worker starts, so the UI can
	// keep editing its own settings while a render is in flight.
	scene:     ^lc.Scene,
	settings:  lc.Render_Config,
	width:     i32,
	height:    i32,
	spp:       i32,
	max_depth: i32,
	path:      string,
	aovs:      bool,
	denoise:   bool,
	compress:  bool,
}

render_job_busy :: proc(j: ^Render_Job) -> bool {
	sync.mutex_lock(&j.mutex)
	defer sync.mutex_unlock(&j.mutex)
	return j.running
}

render_job_state :: proc(j: ^Render_Job) -> (running: bool, progress: f64, status: string, elapsed: f64) {
	sync.mutex_lock(&j.mutex)
	defer sync.mutex_unlock(&j.mutex)
	return j.running, j.progress, j.status, j.elapsed
}

render_job_cancel :: proc(j: ^Render_Job) {
	sync.mutex_lock(&j.mutex)
	j.cancel = true
	sync.mutex_unlock(&j.mutex)
}

render_job_start :: proc(app: ^App) -> bool {
	j := &app.render_job
	if render_job_busy(j) {
		return false
	}

	// Reap a finished worker before starting another.
	if j.worker != nil {
		thread.join(j.worker)
		thread.destroy(j.worker)
		j.worker = nil
	}

	sync.mutex_lock(&j.mutex)
	j.scene = &app.core.scene
	j.settings = app.core.settings
	j.width = app.out_width
	j.height = app.out_height
	j.spp = app.out_spp
	j.max_depth = app.out_max_depth
	j.aovs = app.out_aovs
	j.denoise = app.out_denoise
	j.compress = app.out_compress
	delete(j.path)
	j.path = strings.clone(string(cstring(raw_data(app.out_path[:]))))
	j.running = true
	j.cancel = false
	j.done = false
	j.progress = 0
	delete(j.status)
	j.status = strings.clone("starting")
	j.elapsed = 0
	sync.mutex_unlock(&j.mutex)

	j.worker = thread.create(render_job_proc)
	j.worker.data = app
	thread.start(j.worker)
	return true
}

render_job_shutdown :: proc(j: ^Render_Job) {
	render_job_cancel(j)
	if j.worker != nil {
		thread.join(j.worker)
		thread.destroy(j.worker)
		j.worker = nil
	}
	delete(j.path)
	delete(j.status)
}

@(private = "file")
job_set_status :: proc(j: ^Render_Job, progress: f64, text: string) {
	sync.mutex_lock(&j.mutex)
	j.progress = progress
	delete(j.status)
	j.status = strings.clone(text)
	sync.mutex_unlock(&j.mutex)
}

@(private = "file")
render_job_proc :: proc(t: ^thread.Thread) {
	app := (^App)(t.data)
	j := &app.render_job
	start := time.tick_now()

	sync.mutex_lock(&j.mutex)
	scene := j.scene
	cfg := j.settings
	width, height := j.width, j.height
	total_spp := max(j.spp, 1)
	max_depth := j.max_depth
	path := strings.clone(j.path)
	aovs, denoise, compress := j.aovs, j.denoise, j.compress
	sync.mutex_unlock(&j.mutex)
	defer delete(path)

	finish :: proc(j: ^Render_Job, start: time.Tick, text: string) {
		sync.mutex_lock(&j.mutex)
		j.running = false
		j.done = true
		delete(j.status)
		j.status = strings.clone(text)
		j.elapsed = time.duration_seconds(time.tick_since(start))
		sync.mutex_unlock(&j.mutex)
	}

	// A renderer of its own: sharing the viewport's would evict its scene cache
	// and force a rebuild the moment the render finished.
	renderer, ok := lc.gpu_renderer_create()
	if !ok {
		finish(j, start, "could not create a GPU renderer")
		return
	}
	defer lc.gpu_renderer_destroy(&renderer)

	// Accumulate towards the target so progress can be reported and the job
	// cancelled between batches. The last batch carries the AOV and denoise
	// work, which only makes sense on the finished image.
	frame: lc.GPU_Frame
	accumulated: i32 = 0

	for accumulated < total_spp {
		sync.mutex_lock(&j.mutex)
		cancelled := j.cancel
		sync.mutex_unlock(&j.mutex)
		if cancelled {
			lc.destroy_gpu_frame(&frame)
			finish(j, start, "cancelled")
			return
		}

		batch := min(i32(RENDER_JOB_BATCH), total_spp - accumulated)
		last := (accumulated + batch) >= total_spp

		if accumulated > 0 {
			lc.destroy_gpu_frame(&frame)
		}

		// The scene is shared with the viewport worker, which may be writing a
		// camera into it between batches; hold the same lock it uses.
		sync.mutex_lock(&app.ipr.scene_mutex)
		frame = lc.gpu_render_frame(
			scene, width, height,
			batch, max_depth, cfg.max_radiance,
			0, cfg.roughness_cutoff, cfg.glossy_bias,
			cfg.gi_cache_enabled, cfg.gi_cache_distance, cfg.gi_cache_normal_angle,
			cfg.photon_enabled, cfg.photon_count, cfg.photon_radius, cfg.photon_bounces,
			b32(aovs && last), b32(compress), b32(denoise && last),
			false,
			&renderer, accumulated,
			// A key of its own: this renderer's cache is separate from the
			// viewport's and must not be confused with it.
			1,
		)
		sync.mutex_unlock(&app.ipr.scene_mutex)

		if frame.pixels == nil {
			finish(j, start, "render failed")
			return
		}

		accumulated += batch
		job_set_status(
			j,
			f64(accumulated) / f64(total_spp),
			fmt.tprintf("%d / %d spp", accumulated, total_spp),
		)
		request_wake()
	}
	defer lc.destroy_gpu_frame(&frame)

	job_set_status(j, 1.0, "writing")
	msg, wrote := output.write_gpu_frame(&frame, path, aovs, compress)
	defer delete(msg)

	finish(j, start, wrote ? msg : fmt.tprintf("failed: %s", msg))
	request_wake()
}
