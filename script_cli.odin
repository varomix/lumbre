package main

// `lumbre --script file.py`: the embedded interpreter with no window.
//
// Phase 03 of plans/REALTIME_PHASE03.md. A dataset is a script that authors a
// stage with pxr, varies it, and renders it over and over; this is the host
// that script runs in when nobody is watching. It answers a different command
// set from lumbre-gui's editor — there is no viewport to frame or material
// panel to select into — and says so by name when asked for one of those,
// rather than failing as an unexplained "command failed".

import "base:runtime"
import "core:c"
import "core:encoding/json"
import "core:fmt"
import "core:strings"

import imp "./importers"
import rt "./realtime"
import sc "./script"

@(private = "file")
Script_Host :: struct {
	// Command-line settings a script inherits: resolution, subdivision, the
	// raster view and label switches. `lumbre.render` overrides per call.
	cfg:     Render_Config,
	raster:  Raster_Options,
	// Opened by the first render and kept for the rest of the script.
	session: Raster_Session,
}

// Starts the interpreter, runs the script, and reports whether it completed
// without raising. `args` become sys.argv[1:].
run_script :: proc(path: string, args: []string, cfg: Render_Config, raster: Raster_Options) -> bool {
	host := Script_Host {
		cfg    = cfg,
		raster = raster,
	}
	defer raster_session_close(&host.session)

	status, ok := sc.start(cli_script_command, &host)
	defer delete(status)
	if !ok {
		fmt.eprintln("python:", status)
		return false
	}
	return sc.run_file(path, args)
}

@(private = "file")
cli_script_command :: proc "c" (user: rawptr, cmd: cstring, payload: cstring) -> [^]u8 {
	context = runtime.default_context()
	host := (^Script_Host)(user)

	reply: string
	switch string(cmd) {
	case "host":
		reply = sc.json_object({{"host", sc.json_quote("cli")}})
	case "render":
		reply = script_render(host, string(payload))
	case:
		reply = sc.error_reply(fmt.tprintf("'%s' is not available under `lumbre --script`", cmd))
	}
	defer delete(reply)
	free_all(context.temp_allocator)
	return sc.c_reply(reply)
}

// `lumbre.render(stage, output, ...)`: import the stage the script cached and
// rasterize it. The reply lists every file written.
@(private = "file")
script_render :: proc(host: ^Script_Host, payload: string) -> string {
	value, perr := json.parse_string(payload, allocator = context.temp_allocator)
	if perr != nil {
		return sc.error_reply("malformed request")
	}
	obj, is_obj := value.(json.Object)
	if !is_obj {
		return sc.error_reply("malformed request")
	}

	// Through json_number, not a type assertion on json.Integer: the parser
	// reads every number as a float unless asked not to.
	stage_id := i64(sc.json_number(obj["stage_id"], -1))
	if stage_id < 0 {
		return sc.error_reply("no stage_id")
	}
	out_path, has_out := obj["output"].(json.String)
	if !has_out || out_path == "" {
		return sc.error_reply("no output path")
	}

	cfg := host.cfg
	opts := host.raster
	target := Raster_Target {
		output = string(out_path),
		frame  = -1,
	}
	if v, has := obj["width"]; has  { cfg.image_width = i32(sc.json_number(v, f64(cfg.image_width))) }
	if v, has := obj["height"]; has { cfg.image_height = i32(sc.json_number(v, f64(cfg.image_height))) }
	if v, has := obj["frame"]; has  { target.frame = int(sc.json_number(v, -1)) }
	if v, has := obj["labels"]; has { opts.labels = sc.json_bool(v, opts.labels) }
	if v, has := obj["view"]; has   { opts.view = rt.Debug_View(sc.json_number(v, f64(opts.view))) }
	if v, has := obj["camera"].(json.String); has { target.camera = string(v) }
	target.width, target.height = cfg.image_width, cfg.image_height
	if target.width <= 0 || target.height <= 0 {
		return sc.error_reply(fmt.tprintf("bad resolution %dx%d", target.width, target.height))
	}
	base_dir := ""
	if v, has := obj["base_dir"].(json.String); has { base_dir = string(v) }

	err_buf: [512]u8
	stage := imp.usd_shim_open_cached(c.long(stage_id), raw_data(err_buf[:]), len(err_buf))
	if stage == nil {
		return sc.error_reply(string(cstring(raw_data(err_buf[:]))))
	}
	defer imp.usd_shim_close(stage)

	label := fmt.tprintf("stage cache id %d", stage_id)
	scene, scene_ok := imp.make_scene_from_usd_stage(stage, base_dir, label, cfg)
	if !scene_ok {
		return sc.error_reply("the stage did not import")
	}
	defer destroy_scene(&scene)

	if !host.session.open && !raster_session_open(&host.session) {
		return sc.error_reply("no GPU: the realtime renderer could not start")
	}

	files: [dynamic]string
	defer {
		for f in files {
			delete(f)
		}
		delete(files)
	}
	rendered, ok := raster_session_render(&host.session, &scene, target, opts, &files)
	if !ok {
		return sc.error_reply("nothing rendered; see the log above")
	}

	b := strings.builder_make(context.temp_allocator)
	strings.write_string(&b, "{\"rendered\":")
	fmt.sbprint(&b, rendered)
	strings.write_string(&b, ",\"files\":[")
	for f, i in files {
		if i > 0 {
			strings.write_byte(&b, ',')
		}
		strings.write_string(&b, sc.json_quote(f))
	}
	strings.write_string(&b, "]}")
	return strings.clone(strings.to_string(b))
}
