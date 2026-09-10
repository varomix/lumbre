package output

// Making sure a path can be written to before trying to write it.
//
// A renderer that has just spent minutes on a frame should not lose it to a
// directory that does not exist yet, and "-o dev_renders/today/frame.png" is
// an entirely reasonable thing to type at a shell that has no such directory.
// Every writer here calls this first.

import "core:os"
import "core:path/filepath"
import "core:strings"

// Creates the directory `path`'s file sits in, including any missing parents.
// True when the directory exists afterwards — including when it already did,
// which is the common case and not an error.
ensure_parent_dir :: proc(path: string) -> bool {
	// A slice of `path`, not an allocation — deleting it would abort.
	dir := filepath.dir(path)
	if dir == "" || dir == "." || dir == "/" {
		return true
	}
	if os.exists(dir) {
		return os.is_dir(dir)
	}

	// Walk down from the top so each level exists before its child. `dir` is
	// split rather than recursed on because a relative path bottoms out at ""
	// and an absolute one at "/", and one loop handles both.
	parts := strings.split(dir, "/", context.temp_allocator)
	built := strings.builder_make(context.temp_allocator)
	for part, i in parts {
		if i > 0 || part == "" {
			strings.write_byte(&built, '/')
		}
		strings.write_string(&built, part)
		step := strings.to_string(built)
		if step == "" || step == "/" || os.exists(step) {
			continue
		}
		if err := os.make_directory(step); err != nil && !os.exists(step) {
			return false
		}
	}
	return os.is_dir(dir)
}
