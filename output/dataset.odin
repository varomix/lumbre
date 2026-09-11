package output

import "core:encoding/json"
import "core:fmt"
import "core:hash/xxhash"
import "core:os"
import "core:path/filepath"
import "core:strings"
import lc "../core"

// Stable, JSON-exact integers. File identity is independent of the output
// directory so a dataset can be moved or reproduced in a scratch directory.
// A merger must still reject duplicate file identities and hash collisions.
dataset_id :: proc(identity: string) -> int {
	return int(xxhash.XXH64(transmute([]u8)identity) & ((1 << 53) - 1))
}

// Preserve the existing dataset's IDs when prims disappear, reorder, or are
// added. Explicit classes fix the vocabulary before distributed generation.
// This manifest has one writer; parallel workers use copies of a predeclared
// manifest and merge their disjoint frame outputs afterwards.
dataset_classes :: proc(scene: ^lc.Scene, output_path: string, explicit: []string = nil) -> bool {
	base := strings.trim_suffix(output_path, filepath.ext(output_path))
	path := fmt.tprintf("%s.classes.json", base)
	previous := ""
	names := make([dynamic]string, context.temp_allocator)
	append(&names, "")
	if os.exists(path) {
		data, err := os.read_entire_file(path, context.temp_allocator)
		if err != nil { return false }
		previous = string(data)
		value, jerr := json.parse(data, allocator = context.temp_allocator)
		if jerr != nil { return false }
		array, ok := value.(json.Array)
		if !ok || len(array) == 0 { return false }
		clear(&names)
		seen := make(map[string]bool, context.temp_allocator)
		for item, i in array {
			name, valid := item.(json.String)
			if !valid || (i == 0 && name != "") || (i > 0 && name == "") || seen[string(name)] {
				return false
			}
			seen[string(name)] = true
			append(&names, string(name))
		}
	}
	if len(explicit) > 0 {
		if len(names) > 1 && len(names) != len(explicit) + 1 { return false }
		for name, i in explicit {
			if name == "" { return false }
			for prior in explicit[:i] { if prior == name { return false } }
			if i + 1 < len(names) {
				if names[i + 1] != name { return false }
			} else { append(&names, name) }
		}
	}
	for mesh in scene.meshes {
		name := mesh.semantic_class
		found := false
		for existing in names { found ||= existing == name }
		if !found {
			if len(explicit) > 0 { return false }
			append(&names, name)
		}
	}
	b := strings.builder_make(context.temp_allocator)
	strings.write_string(&b, "[")
	for name, i in names {
		if i > 0 { strings.write_string(&b, ", ") }
		json_string(&b, name)
	}
	strings.write_string(&b, "]\n")
	encoded := strings.to_string(b)
	if encoded != previous {
		if !ensure_parent_dir(path) { return false }
		// Publish only complete manifests. A crash cannot leave half a JSON table.
		tmp := fmt.tprintf("%s.tmp", path)
		if os.write_entire_file(tmp, transmute([]u8)encoded) != nil { return false }
		if os.rename(tmp, path) != nil { return false }
	}

	for name in scene.semantic_classes { if name != "" { delete(name) } }
	delete(scene.semantic_classes)
	scene.semantic_classes = make([]string, len(names))
	for name, i in names { if i > 0 { scene.semantic_classes[i] = strings.clone(name) } }
	for &mesh in scene.meshes {
		for name, i in names {
			if mesh.semantic_class == name { mesh.semantic_class_id = i32(i); break }
		}
	}
	return true
}
