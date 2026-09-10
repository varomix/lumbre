package lumbre_script

// The host side of the embedded interpreter, shared by lumbre-gui's script
// editor and `lumbre --script`.
//
// Both front ends drive the same interpreter the same way — start it against
// the vendored stdlib, route `lumbre_native.call` to a command handler, answer
// in JSON — and differ only in which commands they answer. What they share
// lives here so the two cannot drift: a script that starts in one starts in
// the other.

import "core:c"
import "core:c/libc"
import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"

import lc "../core"
import imp "../importers"

// The vendored interpreter lives beside the executable, not the working
// directory, so a Lumbre launched from anywhere still finds it. Caller owns
// the result.
python_home :: proc(allocator := context.allocator) -> string {
	exe, err := os.get_executable_path(context.allocator)
	if err != nil {
		return strings.clone("lib/darwin/python3.12", allocator)
	}
	defer delete(exe)
	joined, jerr := filepath.join({filepath.dir(exe), "lib", "darwin", "python3.12"}, allocator)
	if jerr != nil {
		return strings.clone("lib/darwin/python3.12", allocator)
	}
	return joined
}

// Starts the interpreter and routes `lumbre_native.call` to `handler`.
// `status` is the Python version on success and the reason on failure; the
// caller owns it either way.
start :: proc(handler: imp.Lumbre_Command_Proc, user: rawptr) -> (status: string, ok: bool) {
	home := python_home()
	defer delete(home)

	if !os.exists(home) {
		return fmt.aprintf("no vendored interpreter at %s (run scripts/vendor_python.sh)", home), false
	}

	err_buf: [512]u8
	chome := strings.clone_to_cstring(home, context.temp_allocator)
	if imp.lumbre_py_init(chome, raw_data(err_buf[:]), len(err_buf)) == 0 {
		return fmt.aprintf("interpreter failed to start: %s", string(cstring(raw_data(err_buf[:])))), false
	}

	imp.lumbre_py_set_command_handler(handler, user)
	return strings.clone(string(imp.lumbre_py_version())), true
}

// Runs the script at `path` as `__main__`, with `args` after it in sys.argv
// and its own directory importable. Returns false if it raised.
//
// Output streams to the process's real stdout and stderr as the script runs.
// `lumbre_py_run` captures into a buffer handed back at the end, which is right
// for the GUI's output pane and wrong for a dataset run that prints progress
// for an hour — so the script swaps the capture back out before it starts.
run_file :: proc(path: string, args: []string) -> bool {
	b := strings.builder_make(context.temp_allocator)
	qpath := json_quote(path)
	strings.write_string(&b, "import sys, os\n")
	strings.write_string(&b, "sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__\n")
	// Line-buffered, so Python's prints interleave with the host's own
	// output in the order they happened.
	strings.write_string(&b, "sys.stdout.reconfigure(line_buffering=True)\n")
	strings.write_string(&b, "sys.argv = [")
	strings.write_string(&b, qpath)
	for arg in args {
		strings.write_string(&b, ", ")
		strings.write_string(&b, json_quote(arg))
	}
	strings.write_string(&b, "]\n")
	fmt.sbprintf(&b, "sys.path.insert(0, os.path.dirname(os.path.abspath(%s)))\n", qpath)
	fmt.sbprintf(&b, "with open(%s, 'rb') as _lumbre_f:\n    _lumbre_code = compile(_lumbre_f.read(), %s, 'exec')\n", qpath, qpath)
	fmt.sbprintf(&b, "exec(_lumbre_code, {{'__name__': '__main__', '__file__': %s}})\n", qpath)

	ok: c.int
	code := strings.clone_to_cstring(strings.to_string(b), context.temp_allocator)
	out := imp.lumbre_py_run(code, &ok)
	if out != nil {
		imp.usd_shim_free_string(out)
	}
	return ok != 0
}

// Hands a reply to the shim, which releases it with free(): it must come from
// libc's malloc. Odin's default heap allocator is not malloc-backed, and
// handing one of its pointers to free() aborts the process.
c_reply :: proc(reply: string) -> [^]u8 {
	n := len(reply)
	buf := ([^]u8)(libc.malloc(c.size_t(n + 1)))
	if buf == nil {
		return nil
	}
	copy(buf[:n], transmute([]u8)reply)
	buf[n] = 0
	return buf
}

// A reply that makes `lumbre.call` raise with `message`. Caller owns it.
error_reply :: proc(message: string) -> string {
	return json_object({{"error", json_quote(message)}})
}

// ── JSON ─────────────────────────────────────────────────────────────────────
//
// Replies are short and fixed-shape, so they are written directly rather than
// marshalled through a struct per command.

json_number :: proc(v: json.Value, fallback: f64) -> f64 {
	#partial switch n in v {
	case json.Integer: return f64(n)
	case json.Float:   return f64(n)
	}
	return fallback
}

json_bool :: proc(v: json.Value, fallback: bool) -> bool {
	if b, ok := v.(json.Boolean); ok {
		return bool(b)
	}
	return fallback
}

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

// Also a valid Python string literal, which is what `run_file` relies on.
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

// Caller owns the result.
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

write_field :: proc(b: ^strings.Builder, name: string, value: string, first: bool) {
	if !first {
		strings.write_byte(b, ',')
	}
	strings.write_string(b, json_quote(name))
	strings.write_byte(b, ':')
	strings.write_string(b, value)
}

json_vec3 :: proc(v: lc.Color) -> string {
	return fmt.tprintf("[%v,%v,%v]", v.x, v.y, v.z)
}
