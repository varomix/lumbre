package main

// Script editor and the host side of the `lumbre` Python module.
//
// The interpreter starts on first use, never at launch: Python is the one part
// of the app that can fail for environmental reasons, and a failure there must
// not stop Lumbre from opening a scene and rendering it.
//
// Python calls `lumbre_native.call(cmd, json)`, which lands in
// `script_command` below. One JSON entry point rather than per-call C bindings
// means extending the API is Odin plus Python, with no native rebuild.

import "base:runtime"
import "core:c"
import "core:mem"
import "core:c/libc"
import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"
import "core:sync"

import lc "../core"
import imgui "../third_party/odin-imgui"
import imp "../importers"

SCRIPT_BUF_SIZE :: 64 * 1024

Script_State :: struct {
	buf:         []u8,
	output:      strings.Builder,
	ready:       bool, // interpreter up
	tried:       bool, // and we have attempted to start it
	status:      string,
	path:        string, // file currently open, if any
	last_ok:     bool,
}

script_init :: proc(s: ^Script_State) {
	s.buf = make([]u8, SCRIPT_BUF_SIZE)
	s.output = strings.builder_make()

	sample := `# Lumbre scripting. Ctrl+Enter or F5 to run.
import lumbre

print(lumbre.stats())

for i, m in enumerate(lumbre.materials()):
    print(i, m["kind"], "roughness", round(m["roughness"], 3))
`
	copy(s.buf, transmute([]u8)sample)
}

script_destroy :: proc(s: ^Script_State) {
	delete(s.buf)
	strings.builder_destroy(&s.output)
	delete(s.status)
	delete(s.path)
	if s.ready {
		imp.lumbre_py_finalize()
	}
}

// The vendored interpreter lives beside the executable, not the working
// directory, so a Lumbre launched from anywhere still finds it.
@(private = "file")
python_home :: proc() -> string {
	exe, err := os.get_executable_path(context.allocator)
	if err != nil {
		return strings.clone("lib/darwin/python3.12")
	}
	defer delete(exe)
	joined, jerr := filepath.join({filepath.dir(exe), "lib", "darwin", "python3.12"})
	if jerr != nil {
		return strings.clone("lib/darwin/python3.12")
	}
	return joined
}

script_ensure_python :: proc(app: ^App) -> bool {
	s := &app.script
	if s.ready {
		return true
	}
	if s.tried {
		return false
	}
	s.tried = true

	home := python_home()
	defer delete(home)

	if !os.exists(home) {
		delete(s.status)
		s.status = fmt.aprintf("no vendored interpreter at %s (run scripts/vendor_python.sh)", home)
		log_printf(&app.log, "[python] %s", s.status)
		return false
	}

	err_buf: [512]u8
	chome := strings.clone_to_cstring(home, context.temp_allocator)
	if imp.lumbre_py_init(chome, raw_data(err_buf[:]), len(err_buf)) == 0 {
		delete(s.status)
		s.status = fmt.aprintf("interpreter failed to start: %s", string(cstring(raw_data(err_buf[:]))))
		log_printf(&app.log, "[python] %s", s.status)
		return false
	}

	imp.lumbre_py_set_command_handler(script_command, app)
	s.ready = true

	delete(s.status)
	s.status = strings.clone(string(imp.lumbre_py_version()))
	log_printf(&app.log, "[python] %s", s.status)
	return true
}

script_run :: proc(app: ^App, code: string) {
	s := &app.script
	if !script_ensure_python(app) {
		strings.builder_reset(&s.output)
		strings.write_string(&s.output, s.status)
		s.last_ok = false
		return
	}

	ccode := strings.clone_to_cstring(code, context.temp_allocator)
	ok: i32
	out := imp.lumbre_py_run(ccode, &ok)

	strings.builder_reset(&s.output)
	if out != nil {
		defer imp.usd_shim_free_string(out)
		strings.write_string(&s.output, string(cstring(out)))
	}
	s.last_ok = ok != 0
}

// ── host command handler ─────────────────────────────────────────────────────

@(private = "file")
script_command :: proc "c" (user: rawptr, cmd: cstring, payload: cstring) -> [^]u8 {
	context = runtime.default_context()
	app := (^App)(user)
	if app == nil {
		return nil
	}

	reply, ok := script_dispatch(app, string(cmd), string(payload))
	if !ok {
		return nil
	}
	defer delete(reply)

	// The shim releases this with free(), so it must come from libc's malloc.
	// Odin's default heap allocator is not malloc-backed, and handing one of
	// its pointers to free() aborts the process.
	n := len(reply)
	buf := ([^]u8)(libc.malloc(c.size_t(n + 1)))
	if buf == nil {
		return nil
	}
	copy(buf[:n], transmute([]u8)reply)
	buf[n] = 0
	return buf
}

@(private = "file")
script_dispatch :: proc(app: ^App, cmd: string, payload: string) -> (string, bool) {
	switch cmd {
	case "stats":
		s := ipr_stats(&app.ipr)
		return json_object({
			{"spp", fmt.tprintf("%d", s.spp)},
			{"target", fmt.tprintf("%d", s.target)},
			{"width", fmt.tprintf("%d", app.ipr.width)},
			{"height", fmt.tprintf("%d", app.ipr.height)},
			{"converged", s.converged ? "true" : "false"},
			{"batch_ms", fmt.tprintf("%.3f", s.batch_ms)},
			{"scene", json_quote(app.scene_path)},
		}), true

	case "materials":
		b := strings.builder_make(context.temp_allocator)
		strings.write_string(&b, "{\"materials\":[")
		for m, i in app.core.scene.materials {
			if i > 0 {
				strings.write_string(&b, ",")
			}
			strings.write_string(&b, material_to_json(m))
		}
		strings.write_string(&b, "]}")
		return strings.clone(strings.to_string(b)), true

	case "set_material":
		return script_set_material(app, payload)

	case "settings":
		cfg := app.core.settings
		return json_object({
			{"spp", fmt.tprintf("%d", app.ipr.target_spp)},
			{"max_depth", fmt.tprintf("%d", app.ipr_max_depth)},
			{"roughness_cutoff", fmt.tprintf("%v", cfg.roughness_cutoff)},
			{"glossy_bias", fmt.tprintf("%v", cfg.glossy_bias)},
			{"gi_cache", cfg.gi_cache_enabled ? "true" : "false"},
			{"photons", cfg.photon_enabled ? "true" : "false"},
			{"photon_count", fmt.tprintf("%d", cfg.photon_count)},
			{"debug_mode", fmt.tprintf("%d", cfg.debug_mode)},
		}), true

	case "set_settings":
		return script_set_settings(app, payload)

	case "prims":
		b := strings.builder_make(context.temp_allocator)
		strings.write_string(&b, "{\"prims\":[")
		n := 0
		for node in app.usd.nodes {
			if node.path == "/" || node.path == "" {
				continue
			}
			if n > 0 {
				strings.write_string(&b, ",")
			}
			strings.write_byte(&b, '{')
			write_field(&b, "path", json_quote(node.path), true)
			write_field(&b, "type", json_quote(node.type_name), false)
			strings.write_byte(&b, '}')
			n += 1
		}
		strings.write_string(&b, "]}")
		return strings.clone(strings.to_string(b)), true

	case "render_to_file":
		return script_render_to_file(app, payload)

	case "render_status":
		running, progress, status, elapsed := render_job_state(&app.render_job)
		return json_object({
			{"running", running ? "true" : "false"},
			{"progress", fmt.tprintf("%v", progress)},
			{"status", json_quote(status)},
			{"elapsed", fmt.tprintf("%v", elapsed)},
		}), true

	case "render_cancel":
		render_job_cancel(&app.render_job)
		return strings.clone("{\"ok\":true}"), true

	case "save_look":
		return json_object({{"ok", look_save(app) ? "true" : "false"}}), true

	case "load_look":
		applied := look_load(app)
		if applied {
			ipr_materials_changed(&app.ipr)
		}
		return json_object({{"ok", applied ? "true" : "false"}}), true

	case "pick":
		value, perr := json.parse_string(payload, allocator = context.temp_allocator)
		if perr != nil {
			return "", false
		}
		obj, is_obj := value.(json.Object)
		if !is_obj {
			return "", false
		}
		u := json_number(obj["u"], 0.5)
		v := json_number(obj["v"], 0.5)

		sync.mutex_lock(&app.ipr.scene_mutex)
		hit := pick_at(&app.core.scene, u, v)
		sync.mutex_unlock(&app.ipr.scene_mutex)

		if !hit.hit {
			return json_object({{"hit", "false"}}), true
		}
		app.selected_material = hit.material
		return json_object({
			{"hit", "true"},
			{"material", fmt.tprintf("%d", hit.material)},
			{"distance", fmt.tprintf("%v", hit.distance)},
			{"point", json_vec3(hit.point)},
			{"normal", json_vec3(hit.normal)},
		}), true

	case "frame_all":
		app_frame_all(app)
		return strings.clone("{\"ok\":true}"), true

	case "restart":
		ipr_invalidate(&app.ipr)
		return strings.clone("{\"ok\":true}"), true
	}

	return "", false
}

@(private = "file")
script_set_material :: proc(app: ^App, payload: string) -> (string, bool) {
	value, err := json.parse_string(payload, allocator = context.temp_allocator)
	if err != nil {
		return "", false
	}
	obj, is_obj := value.(json.Object)
	if !is_obj {
		return "", false
	}

	index := int(json_number(obj["index"], -1))
	if index < 0 || index >= len(app.core.scene.materials) {
		return "", false
	}
	fields, has_fields := obj["fields"].(json.Object)
	if !has_fields {
		return strings.clone("{\"ok\":true}"), true
	}

	m := &app.core.scene.materials[index]
	for key, v in fields {
		switch key {
		case "base_color":            m.albedo = json_color(v, m.albedo)
		case "roughness":             m.roughness = json_number(v, m.roughness)
		case "metallic":              m.metallic = json_number(v, m.metallic)
		case "specular":              m.specular = json_number(v, m.specular)
		case "ior":                   m.ir = json_number(v, m.ir)
		case "clearcoat":             m.clearcoat = json_number(v, m.clearcoat)
		case "clearcoat_roughness":   m.clearcoat_roughness = json_number(v, m.clearcoat_roughness)
		case "sheen":                 m.sheen = json_number(v, m.sheen)
		case "anisotropic":           m.anisotropic = json_number(v, m.anisotropic)
		case "transmission":          m.spec_trans = json_number(v, m.spec_trans)
		case "subsurface":            m.subsurface = json_number(v, m.subsurface)
		case "emission":              m.emission = json_color(v, m.emission)
		case "emission_strength":     m.emission_strength = json_number(v, m.emission_strength)
		}
	}
	ipr_materials_changed(&app.ipr)
	return strings.clone("{\"ok\":true}"), true
}

@(private = "file")
script_render_to_file :: proc(app: ^App, payload: string) -> (string, bool) {
	value, err := json.parse_string(payload, allocator = context.temp_allocator)
	if err != nil {
		return "", false
	}
	obj, is_obj := value.(json.Object)
	if !is_obj {
		return "", false
	}

	if path, has := obj["path"].(json.String); has {
		n := min(len(path), len(app.out_path) - 1)
		mem.zero(raw_data(app.out_path[:]), len(app.out_path))
		copy(app.out_path[:n], transmute([]u8)string(path)[:n])
	}
	if v, has := obj["width"]; has  { app.out_width = i32(json_number(v, f64(app.out_width))) }
	if v, has := obj["height"]; has { app.out_height = i32(json_number(v, f64(app.out_height))) }
	if v, has := obj["spp"]; has    { app.out_spp = i32(json_number(v, f64(app.out_spp))) }
	if v, has := obj["depth"]; has  { app.out_max_depth = i32(json_number(v, f64(app.out_max_depth))) }
	if v, has := obj["aovs"]; has   { app.out_aovs = json_bool(v, app.out_aovs) }
	if v, has := obj["denoise"]; has{ app.out_denoise = json_bool(v, app.out_denoise) }

	started := render_job_start(app)
	return json_object({{"started", started ? "true" : "false"}}), true
}

@(private = "file")
script_set_settings :: proc(app: ^App, payload: string) -> (string, bool) {
	value, err := json.parse_string(payload, allocator = context.temp_allocator)
	if err != nil {
		return "", false
	}
	obj, is_obj := value.(json.Object)
	if !is_obj {
		return "", false
	}
	fields, has_fields := obj["fields"].(json.Object)
	if !has_fields {
		return strings.clone("{\"ok\":true}"), true
	}

	restart := false
	for key, v in fields {
		switch key {
		case "spp":
			ipr_set_target_spp(&app.ipr, i32(json_number(v, 512)))
		case "max_depth":
			app.ipr_max_depth = i32(json_number(v, f64(app.ipr_max_depth)));  restart = true
		case "roughness_cutoff":
			app.core.settings.roughness_cutoff = json_number(v, app.core.settings.roughness_cutoff); restart = true
		case "glossy_bias":
			app.core.settings.glossy_bias = json_number(v, app.core.settings.glossy_bias); restart = true
		case "gi_cache":
			app.core.settings.gi_cache_enabled = b32(json_bool(v, bool(app.core.settings.gi_cache_enabled))); restart = true
		case "photons":
			app.core.settings.photon_enabled = b32(json_bool(v, bool(app.core.settings.photon_enabled))); restart = true
		case "photon_count":
			app.core.settings.photon_count = i32(json_number(v, f64(app.core.settings.photon_count))); restart = true
		case "debug_mode":
			app.core.settings.debug_mode = i32(json_number(v, f64(app.core.settings.debug_mode))); restart = true
		}
	}
	if restart {
		ipr_settings_changed(app)
	}
	return strings.clone("{\"ok\":true}"), true
}

// ── small JSON helpers ───────────────────────────────────────────────────────
//
// Replies are short and fixed-shape, so they are written directly rather than
// marshalled through a struct per command.

@(private = "file")
json_number :: proc(v: json.Value, fallback: f64) -> f64 {
	#partial switch n in v {
	case json.Integer: return f64(n)
	case json.Float:   return f64(n)
	}
	return fallback
}

@(private = "file")
json_bool :: proc(v: json.Value, fallback: bool) -> bool {
	if b, ok := v.(json.Boolean); ok {
		return bool(b)
	}
	return fallback
}

@(private = "file")
json_color :: proc(v: json.Value, fallback: lc.Color) -> lc.Color {
	arr, ok := v.(json.Array)
	if !ok || len(arr) < 3 {
		return fallback
	}
	return lc.Color{
		json_number(arr[0], fallback.x),
		json_number(arr[1], fallback.y),
		json_number(arr[2], fallback.z),
	}
}

@(private = "file")
json_quote :: proc(s: string) -> string {
	b := strings.builder_make(context.temp_allocator)
	strings.write_byte(&b, '"')
	for ch in transmute([]u8)s {
		switch ch {
		case '"':  strings.write_string(&b, "\\\"")
		case '\\': strings.write_string(&b, "\\\\")
		case '\n': strings.write_string(&b, "\\n")
		case '\r': strings.write_string(&b, "\\r")
		case '\t': strings.write_string(&b, "\\t")
		case:
			if ch < 0x20 {
				strings.write_string(&b, fmt.tprintf("\\u%04x", ch))
			} else {
				strings.write_byte(&b, ch)
			}
		}
	}
	strings.write_byte(&b, '"')
	return strings.to_string(b)
}

@(private = "file")
json_object :: proc(pairs: [][2]string) -> string {
	b := strings.builder_make(context.temp_allocator)
	strings.write_byte(&b, '{')
	for pair, i in pairs {
		if i > 0 {
			strings.write_byte(&b, ',')
		}
		strings.write_string(&b, json_quote(pair[0]))
		strings.write_byte(&b, ':')
		strings.write_string(&b, pair[1])
	}
	strings.write_byte(&b, '}')
	return strings.clone(strings.to_string(b))
}

@(private = "file")
material_to_json :: proc(m: lc.Material) -> string {
	kind := ""
	switch m.kind {
	case .Lambertian: kind = "Lambertian"
	case .Metal:      kind = "Metal"
	case .Dielectric: kind = "Dielectric"
	case .Principled: kind = "Principled"
	case .Emissive:   kind = "Emissive"
	}

	// Built with a builder rather than a format string: Odin's fmt treats `{`
	// as a formatting directive, so literal JSON braces in a format string come
	// out as "%!(MISSING CLOSE BRACE)".
	b := strings.builder_make(context.temp_allocator)
	strings.write_byte(&b, '{')
	write_field(&b, "kind", json_quote(kind), true)
	write_field(&b, "base_color", json_vec3(m.albedo), false)
	write_field(&b, "roughness", fmt.tprintf("%v", m.roughness), false)
	write_field(&b, "metallic", fmt.tprintf("%v", m.metallic), false)
	write_field(&b, "specular", fmt.tprintf("%v", m.specular), false)
	write_field(&b, "ior", fmt.tprintf("%v", m.ir), false)
	write_field(&b, "clearcoat", fmt.tprintf("%v", m.clearcoat), false)
	write_field(&b, "clearcoat_roughness", fmt.tprintf("%v", m.clearcoat_roughness), false)
	write_field(&b, "sheen", fmt.tprintf("%v", m.sheen), false)
	write_field(&b, "anisotropic", fmt.tprintf("%v", m.anisotropic), false)
	write_field(&b, "transmission", fmt.tprintf("%v", m.spec_trans), false)
	write_field(&b, "subsurface", fmt.tprintf("%v", m.subsurface), false)
	write_field(&b, "emission", json_vec3(m.emission), false)
	write_field(&b, "emission_strength", fmt.tprintf("%v", m.emission_strength), false)
	strings.write_byte(&b, '}')
	return strings.to_string(b)
}

@(private = "file")
write_field :: proc(b: ^strings.Builder, name: string, value: string, first: bool) {
	if !first {
		strings.write_byte(b, ',')
	}
	strings.write_string(b, json_quote(name))
	strings.write_byte(b, ':')
	strings.write_string(b, value)
}

@(private = "file")
json_vec3 :: proc(v: lc.Color) -> string {
	return fmt.tprintf("[%v,%v,%v]", v.x, v.y, v.z)
}

// ── panel ────────────────────────────────────────────────────────────────────

draw_script_panel :: proc(app: ^App) {
	if !imgui.Begin(WINDOW_SCRIPT, &app.show_script) {
		imgui.End()
		return
	}
	defer imgui.End()

	s := &app.script

	if imgui.Button("Run") {
		script_run(app, string(cstring(raw_data(s.buf))))
	}
	imgui.SameLine()
	imgui.TextDisabled("Ctrl+Enter / F5")
	imgui.SameLine()
	if imgui.SmallButton("Clear output") {
		strings.builder_reset(&s.output)
	}
	imgui.SameLine()
	if s.ready {
		imgui.TextDisabled(tmp_cstring(fmt.tprintf("python %s", s.status)))
	} else if s.tried {
		imgui.TextColored({1.0, 0.5, 0.4, 1}, tmp_cstring(s.status))
	} else {
		// Starting on first Run keeps a Python problem from being a startup
		// problem: the rest of the app works whether or not this succeeds.
		imgui.TextDisabled("interpreter starts on first run")
	}

	imgui.Separator()

	if imgui.IsWindowFocused(imgui.FocusedFlags_RootAndChildWindows) {
		io := imgui.GetIO()
		if imgui.IsKeyPressed(.F5, false) ||
		   ((io.KeyCtrl || io.KeySuper) && imgui.IsKeyPressed(.Enter, false)) {
			script_run(app, string(cstring(raw_data(s.buf))))
		}
	}

	avail := imgui.GetContentRegionAvail()
	editor_h := avail.y * 0.6

	imgui.InputTextMultiline(
		"##scriptsrc",
		cstring(raw_data(s.buf)),
		len(s.buf),
		{avail.x, editor_h},
		{.AllowTabInput},
	)

	imgui.Separator()
	if imgui.BeginChild("##scriptout", {0, 0}, {}, {.HorizontalScrollbar}) {
		text := strings.to_string(s.output)
		if text == "" {
			imgui.TextDisabled("(no output)")
		} else {
			col := s.last_ok ? imgui.Vec4{0.85, 0.85, 0.88, 1} : imgui.Vec4{1.0, 0.55, 0.45, 1}
			imgui.PushStyleColorImVec4(.Text, col)
			// Explicit end pointer: output can be long, and this avoids copying
			// it every frame just to null-terminate.
			start := raw_data(text)
			end := rawptr(uintptr(start) + uintptr(len(text)))
			imgui.TextUnformatted(cstring(start), cstring(end))
			imgui.PopStyleColor()
		}
	}
	imgui.EndChild()
}
