package main

// In-app log panel.
//
// The renderer and the importers already print useful things with `fmt.println`
// — the flatten summary, photon counts, per-pass timings. Rather than making
// them log through a new API, this captures fd 1 with a pipe and drains it on a
// worker thread, so every existing print shows up in the GUI unchanged.
//
// The drain thread mirrors everything to the real stdout as well, so running
// `lumbre-gui` from a terminal still behaves normally and nothing is lost if the
// GUI dies before the panel is visible.

import "core:c"
import "core:fmt"
import "core:strings"
import "core:sync"
import "core:sys/posix"
import "core:thread"

LOG_MAX_LINES :: 4000

// Read chunk for the drain thread. The pipe buffer is finite (~64 KB on
// darwin); if nothing drains it a chatty render would block the writer, which
// is the renderer's own thread. Hence the dedicated thread.
LOG_READ_CHUNK :: 4096

Log :: struct {
	mutex:        sync.Mutex,
	lines:        [dynamic]string,
	// Set by any producer, cleared by the panel. Used to decide whether the log
	// has anything new worth a redraw.
	dirty:        bool,
	scroll_to_end: bool,

	// stdout capture state
	capturing:    bool,
	saved_stdout: posix.FD,
	read_fd:      posix.FD,
	write_fd:     posix.FD,
	drain:        ^thread.Thread,
}

// Appends one line. Safe from any thread.
log_line :: proc(l: ^Log, text: string) {
	sync.mutex_lock(&l.mutex)
	defer sync.mutex_unlock(&l.mutex)

	append(&l.lines, strings.clone(text))

	// Trim from the front once the backlog is too long. Dropping a block at a
	// time keeps this from turning into a per-line memmove.
	if len(l.lines) > LOG_MAX_LINES {
		drop := len(l.lines) - LOG_MAX_LINES
		for i in 0 ..< drop {
			delete(l.lines[i])
		}
		copy(l.lines[:], l.lines[drop:])
		resize(&l.lines, len(l.lines) - drop)
	}

	l.dirty = true
	l.scroll_to_end = true
}

log_printf :: proc(l: ^Log, format: string, args: ..any) {
	text := fmt.tprintf(format, ..args)
	log_line(l, text)
}

// Redirects fd 1 into a pipe and starts draining it. Returns false and leaves
// stdout untouched if any step fails — the app must still run without capture.
log_capture_stdout :: proc(l: ^Log) -> bool {
	fds: [2]posix.FD
	if posix.pipe(&fds) != .OK {
		return false
	}
	l.read_fd = fds[0]
	l.write_fd = fds[1]

	l.saved_stdout = posix.dup(posix.STDOUT_FILENO)
	if l.saved_stdout < 0 {
		posix.close(l.read_fd)
		posix.close(l.write_fd)
		return false
	}

	if posix.dup2(l.write_fd, posix.STDOUT_FILENO) < 0 {
		posix.close(l.read_fd)
		posix.close(l.write_fd)
		posix.close(l.saved_stdout)
		return false
	}

	l.capturing = true
	l.drain = thread.create(log_drain_proc)
	l.drain.data = l
	thread.start(l.drain)
	return true
}

// Restores fd 1 and shuts the drain thread down. Closing the write end is what
// makes the blocking `read` return 0 so the thread can exit.
log_stop_capture :: proc(l: ^Log) {
	if !l.capturing {
		return
	}
	l.capturing = false

	posix.dup2(l.saved_stdout, posix.STDOUT_FILENO)
	posix.close(l.write_fd)

	if l.drain != nil {
		thread.join(l.drain)
		thread.destroy(l.drain)
		l.drain = nil
	}

	posix.close(l.read_fd)
	posix.close(l.saved_stdout)
}

@(private = "file")
log_drain_proc :: proc(t: ^thread.Thread) {
	l := (^Log)(t.data)

	buf: [LOG_READ_CHUNK]byte
	// Carries an unterminated tail between reads: a write can land mid-line.
	pending := strings.builder_make()
	defer strings.builder_destroy(&pending)

	for {
		n := posix.read(l.read_fd, raw_data(buf[:]), LOG_READ_CHUNK)
		if n <= 0 {
			break
		}
		chunk := buf[:n]

		// Mirror to the terminal we inherited, so piping/redirecting still works.
		posix.write(l.saved_stdout, raw_data(chunk), c.size_t(n))

		for ch in chunk {
			if ch == '\n' {
				log_line(l, strings.to_string(pending))
				strings.builder_reset(&pending)
			} else if ch != '\r' {
				strings.write_byte(&pending, ch)
			}
		}

		// A producer wrote something; if the UI thread is parked in
		// WaitEventTimeout it needs to come back and repaint.
		request_wake()
	}

	// Flush a trailing line with no newline.
	if strings.builder_len(pending) > 0 {
		log_line(l, strings.to_string(pending))
		request_wake()
	}
}

log_destroy :: proc(l: ^Log) {
	log_stop_capture(l)
	for line in l.lines {
		delete(line)
	}
	delete(l.lines)
}
