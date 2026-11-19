package main

// Timeline: play, pause, step and scrub an animated USD stage.
//
// The scene is imported once, at the stage's start time. Changing frame does
// not re-import it: `usd_eval_motion` walks the stage the USD panels already
// hold open for placements alone, and `scene_apply_usd_motion` moves the
// imported meshes to them, with their previous-frame transforms for motion
// vectors. The rasterizer then rewrites only its instance buffers; the path
// tracer has to rebuild its acceleration structure, so playback there is as
// fast as that rebuild.
//
// Frames are USD time codes. Playback runs on the wall clock at the stage's
// framesPerSecond, skipping frames rather than slowing down when a frame
// takes longer than its slot, and wakes the idle-driven loop only as often as
// a frame is due.
//
// Not animated yet: lights, and the stage's cameras (the viewport looks
// through its own orbit camera).

import "core:fmt"
import "core:math"
import "core:sync"
import "core:time"

import imp "../importers"
import imgui "../third_party/odin-imgui"

WINDOW_TIMELINE :: "Timeline"

ICON_PLAY       :: "" // play
ICON_PAUSE      :: "" // pause
ICON_STEP_BACK  :: "" // backward-step
ICON_STEP_FWD   :: "" // forward-step
ICON_TO_START   :: "" // backward-fast
ICON_TO_END     :: "" // forward-fast
ICON_LOOP       :: "" // repeat

Timeline :: struct {
	// The stage's authored range; without one there is nothing to play.
	has_range:       bool,
	start, end:      f64,
	fps:             f64, // framesPerSecond
	codes_per_frame: f64, // timeCodesPerSecond / framesPerSecond
	frame:           f64, // the time code the scene is at
	playing:         bool,
	looping:         bool,
	// Playback is timed from where it last started, so a slow frame is
	// followed by a skip rather than drift.
	play_tick:       time.Tick,
	play_from:       f64,
}

// Reads the range of the stage the USD panels hold. Called whenever a scene
// is adopted; the importer read that scene at `start`, so that is where the
// timeline begins.
timeline_reset :: proc(app: ^App) {
	tl := &app.timeline
	looping := tl.looping
	tl^ = {looping = looping}
	if !app.usd.open {
		return
	}
	info: imp.Usd_Shim_Time_Info
	if imp.usd_shim_get_time_info(app.usd.stage, &info) == 0 || info.has_authored_range == 0 {
		return
	}
	tl.has_range = info.end_time_code >= info.start_time_code
	tl.start = info.start_time_code
	tl.end = info.end_time_code
	tl.fps = info.frames_per_second > 0 ? info.frames_per_second : 24
	tl.codes_per_frame = 1
	if info.frames_per_second > 0 && info.time_codes_per_second > 0 {
		tl.codes_per_frame = info.time_codes_per_second / info.frames_per_second
	}
	tl.frame = tl.start
}

// Moves the scene to time code `frame`, clamped to the range.
timeline_set_frame :: proc(app: ^App, frame: f64) {
	tl := &app.timeline
	if !tl.has_range || !app.scene_loaded || !app.usd.open {
		return
	}
	frame := clamp(frame, tl.start, tl.end)
	if frame == tl.frame {
		return
	}
	tl.frame = frame

	motion := imp.usd_eval_motion(app.usd.stage, frame, context.temp_allocator)
	// The IPR worker reads the scene under this lock for a whole batch; in
	// realtime mode it is parked, so this does not wait.
	sync.mutex_lock(&app.ipr.scene_mutex)
	changed := imp.scene_apply_usd_motion(&app.core.scene, motion)
	sync.mutex_unlock(&app.ipr.scene_mutex)
	if changed > 0 {
		// A new scene key: the rasterizer sees placements changed and rewrites
		// its instances; the path tracer rebuilds.
		ipr_scene_changed(app)
	}
	app_wake(app)
}

timeline_step :: proc(app: ^App, frames: int) {
	tl := &app.timeline
	next := tl.frame + f64(frames) * tl.codes_per_frame
	if tl.looping {
		span := tl.end - tl.start + tl.codes_per_frame
		if next > tl.end { next -= span }
		if next < tl.start { next += span }
	}
	timeline_set_frame(app, next)
}

timeline_play :: proc(app: ^App, play: bool) {
	tl := &app.timeline
	if !tl.has_range {
		return
	}
	if play && tl.frame >= tl.end && !tl.looping {
		timeline_set_frame(app, tl.start)
	}
	tl.playing = play
	tl.play_tick = time.tick_now()
	tl.play_from = tl.frame
	app_wake(app)
}

// Advances playback to the frame the wall clock says is due. Called once per
// loop iteration, before the frame is drawn.
timeline_tick :: proc(app: ^App) {
	tl := &app.timeline
	if !tl.playing {
		return
	}
	if !tl.has_range || !app.scene_loaded {
		tl.playing = false
		return
	}
	elapsed := time.duration_seconds(time.tick_since(tl.play_tick))
	frames := math.floor(elapsed * tl.fps)
	next := tl.play_from + frames * tl.codes_per_frame
	if next > tl.end {
		if tl.looping {
			span := tl.end - tl.start + tl.codes_per_frame
			next = tl.start + math.mod(next - tl.start, span)
		} else {
			next = tl.end
			tl.playing = false
		}
	}
	if next != tl.frame {
		timeline_set_frame(app, next)
	}
}

// How long the loop may sleep before the next frame is due, in ms, or -1
// when nothing is playing.
timeline_wait_ms :: proc(app: ^App) -> i32 {
	tl := &app.timeline
	if !tl.playing {
		return -1
	}
	elapsed := time.duration_seconds(time.tick_since(tl.play_tick))
	due := (math.floor(elapsed * tl.fps) + 1) / tl.fps
	return max(i32(math.ceil((due - elapsed) * 1000)), 1)
}

draw_timeline_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_TIMELINE, &app.show_timeline) {
		imgui.End()
		return
	}
	defer imgui.End()

	tl := &app.timeline
	if !app.scene_loaded || !tl.has_range {
		imgui.TextDisabled(app.scene_loaded ? "This stage has no time range (startTimeCode/endTimeCode)." : "No scene loaded.")
		return
	}

	if imgui.Button(ICON_TO_START) { tl.playing = false; timeline_set_frame(app, tl.start) }
	imgui.SetItemTooltip("First frame (Home)")
	imgui.SameLine()
	if imgui.Button(ICON_STEP_BACK) { tl.playing = false; timeline_step(app, -1) }
	imgui.SetItemTooltip("Previous frame (Left)")
	imgui.SameLine()
	if imgui.Button(tl.playing ? ICON_PAUSE : ICON_PLAY) { timeline_play(app, !tl.playing) }
	imgui.SetItemTooltip(tl.playing ? "Pause (Space)" : "Play (Space)")
	imgui.SameLine()
	if imgui.Button(ICON_STEP_FWD) { tl.playing = false; timeline_step(app, 1) }
	imgui.SetItemTooltip("Next frame (Right)")
	imgui.SameLine()
	if imgui.Button(ICON_TO_END) { tl.playing = false; timeline_set_frame(app, tl.end) }
	imgui.SetItemTooltip("Last frame (End)")
	imgui.SameLine()
	if tl.looping { imgui.PushStyleColorImVec4(.Button, {0.26, 0.42, 0.62, 1.0}) }
	if imgui.Button(ICON_LOOP) { tl.looping = !tl.looping }
	if tl.looping { imgui.PopStyleColor() }
	imgui.SetItemTooltip("Loop")

	imgui.SameLine()
	imgui.TextDisabled(fmt.ctprintf("%v fps", tl.fps))

	// Whole frames when a frame is one time code, the usual case; otherwise
	// the slider moves in time codes.
	imgui.SetNextItemWidth(-1)
	if tl.codes_per_frame == 1 && tl.start == math.floor(tl.start) {
		value := i32(math.round(tl.frame))
		if imgui.SliderInt("##frame", &value, i32(tl.start), i32(tl.end), "Frame %d") {
			tl.playing = false
			timeline_set_frame(app, f64(value))
		}
	} else {
		value := f32(tl.frame)
		if imgui.SliderFloat("##frame", &value, f32(tl.start), f32(tl.end), "Time %.2f") {
			tl.playing = false
			timeline_set_frame(app, math.round((f64(value) - tl.start) / tl.codes_per_frame) * tl.codes_per_frame + tl.start)
		}
	}

	// Transport keys while the panel or the viewport has focus, not while
	// typing into a field.
	if imgui.IsWindowFocused({.RootWindow, .ChildWindows}) && !imgui.GetIO().WantTextInput {
		if imgui.IsKeyPressed(.Space, false) { timeline_play(app, !tl.playing) }
		if imgui.IsKeyPressed(.LeftArrow) { tl.playing = false; timeline_step(app, -1) }
		if imgui.IsKeyPressed(.RightArrow) { tl.playing = false; timeline_step(app, 1) }
		if imgui.IsKeyPressed(.Home, false) { tl.playing = false; timeline_set_frame(app, tl.start) }
		if imgui.IsKeyPressed(.End, false) { tl.playing = false; timeline_set_frame(app, tl.end) }
	}
}
