package output

// COCO annotation JSON.
//
// COCO is the format nearly every detection and segmentation toolchain reads,
// so it is what a dataset produced here should speak first. KITTI is the same
// derived data in a different shape and can follow.
//
// Written by hand rather than through a JSON encoder: the schema is fixed and
// small, the numbers must be formatted exactly (COCO consumers are strict
// about integer category ids and float boxes), and this way the file has no
// dependency on how any particular encoder decides to spell a float.
//
// One file per rendered frame rather than one per dataset. A batch run writes
// its frames independently and may be interrupted or resumed, and merging N
// self-contained files afterwards is trivial where truncating one shared file
// is not.

import "core:fmt"
import "core:os"
import "core:strings"

import lc "../core"

// The scene-level facts a COCO file needs beyond the annotations themselves.
COCO_Frame :: struct {
	// Written into `images[0]`; the file name a consumer will look for next
	// to the annotations, normally the beauty image.
	image_name:  string,
	image_id:    int,
	width:       i32,
	height:      i32,
	// Class id → name, indexed by id. Index 0 is the unlabelled class and is
	// not written as a category: COCO has no category for "background".
	class_names: []string,
	intrinsics:  lc.Camera_Intrinsics,
}

// Escapes the few characters JSON forbids in a string. Prim paths are the
// realistic source of these — a stage may legally hold a backslash — and an
// unescaped one produces a file that fails to parse rather than one that is
// merely wrong.
@(private)
json_string :: proc(b: ^strings.Builder, s: string) {
	strings.write_byte(b, '"')
	for c in transmute([]u8)s {
		switch c {
		case '"':  strings.write_string(b, "\\\"")
		case '\\': strings.write_string(b, "\\\\")
		case '\n': strings.write_string(b, "\\n")
		case '\r': strings.write_string(b, "\\r")
		case '\t': strings.write_string(b, "\\t")
		case:
			if c < 0x20 {
				strings.write_string(b, fmt.tprintf("\\u%04x", c))
			} else {
				strings.write_byte(b, c)
			}
		}
	}
	strings.write_byte(b, '"')
}

// Serializes one frame's annotations as a COCO document.
//
// Returns the JSON text, owned by the caller. Split from the file write so a
// test can check the document without touching a disk.
coco_encode :: proc(
	frame: COCO_Frame,
	annotations: []lc.Annotation,
	allocator := context.allocator,
) -> string {
	b := strings.builder_make(allocator)

	strings.write_string(&b, "{\n")

	// `info` carries the camera intrinsics. COCO has no standard place for
	// them, and a synthetic dataset without them cannot be related back to a
	// real capture rig, so they go in the one object the schema leaves open.
	strings.write_string(&b, "  \"info\": {\n")
	strings.write_string(&b, "    \"description\": \"Lumbre synthetic frame\",\n")
	strings.write_string(&b, fmt.tprintf(
		"    \"camera\": {{\"fx\": %.6f, \"fy\": %.6f, \"cx\": %.6f, \"cy\": %.6f, \"width\": %d, \"height\": %d}}\n",
		frame.intrinsics.fx, frame.intrinsics.fy, frame.intrinsics.cx, frame.intrinsics.cy,
		frame.intrinsics.width, frame.intrinsics.height,
	))
	strings.write_string(&b, "  },\n")

	strings.write_string(&b, "  \"images\": [\n    {\"id\": ")
	strings.write_string(&b, fmt.tprintf("%d, \"file_name\": ", frame.image_id))
	json_string(&b, frame.image_name)
	strings.write_string(&b, fmt.tprintf(", \"width\": %d, \"height\": %d}\n  ],\n",
		frame.width, frame.height))

	// Categories come from the scene's class table, not from what happens to
	// be visible: a frame where the chair is off-camera must still agree with
	// every other frame about what category 3 means.
	strings.write_string(&b, "  \"categories\": [\n")
	first := true
	for name, id in frame.class_names {
		if id == 0 {
			continue // background is not a category
		}
		if !first {
			strings.write_string(&b, ",\n")
		}
		first = false
		strings.write_string(&b, fmt.tprintf("    {{\"id\": %d, \"name\": ", id))
		json_string(&b, name)
		strings.write_string(&b, ", \"supercategory\": \"\"}")
	}
	strings.write_string(&b, "\n  ],\n")

	strings.write_string(&b, "  \"annotations\": [\n")
	first = true
	for a in annotations {
		// Unlabelled geometry stays in the instance mask, not the detection set.
		if a.semantic_id <= 0 || int(a.semantic_id) >= len(frame.class_names) { continue }
		if !first {
			strings.write_string(&b, ",\n")
		}
		first = false
		// COCO annotation identity is dataset-wide; retain the mask ID separately.
		strings.write_string(&b, fmt.tprintf(
			"    {{\"id\": %d, \"image_id\": %d, \"category_id\": %d, \"bbox\": [%d, %d, %d, %d], \"area\": %d, \"iscrowd\": 0",
			dataset_id(fmt.tprintf("%d:%d", frame.image_id, a.instance_id)), frame.image_id, a.semantic_id,
			a.bbox[0], a.bbox[1], a.bbox[2], a.bbox[3], a.pixel_area,
		))
		strings.write_string(&b, fmt.tprintf(", \"instance_id\": %d", a.instance_id))
		// Extensions past the COCO schema, which consumers ignore and a
		// 3D-aware one needs: the prim it came from, its world bounds, and
		// its pose. A detector reading this file sees only the fields above.
		if a.path != "" {
			strings.write_string(&b, ", \"prim_path\": ")
			json_string(&b, a.path)
		}
		if a.has_pose {
			strings.write_string(&b, fmt.tprintf(
				", \"bbox3d\": [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]",
				a.bounds_min.x, a.bounds_min.y, a.bounds_min.z,
				a.bounds_max.x, a.bounds_max.y, a.bounds_max.z,
			))
			// Row-major, so a consumer reading four rows of four gets the
			// transform the way it is written in USD rather than transposed.
			strings.write_string(&b, ", \"pose\": [")
			for row in 0 ..< 4 {
				for col in 0 ..< 4 {
					if row > 0 || col > 0 {
						strings.write_string(&b, ", ")
					}
					strings.write_string(&b, fmt.tprintf("%.6f", a.pose[row, col]))
				}
			}
			strings.write_string(&b, "]")
		}
		strings.write_string(&b, "}")
	}
	strings.write_string(&b, "\n  ]\n}\n")

	return strings.to_string(b)
}

// Writes the encoded document to `path`.
coco_write :: proc(
	frame: COCO_Frame,
	annotations: []lc.Annotation,
	path: string,
) -> (
	message: string,
	ok: bool,
) {
	if !ensure_parent_dir(path) {
		return fmt.aprintf("cannot create the directory for %s", path), false
	}
	json := coco_encode(frame, annotations, context.temp_allocator)
	if err := os.write_entire_file(path, transmute([]u8)json); err != nil {
		return fmt.aprintf("failed to write %s: %v", path, err), false
	}
	return fmt.aprintf("wrote %s (%d annotations)", path, len(annotations)), true
}
