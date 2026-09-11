package output

// Checks the COCO document.
//
//   odin test output
//
// The interesting cases are the ones a consumer chokes on rather than the
// happy path: a category table that must agree across frames whether or not
// the object was visible, and a prim path with a character JSON forbids, which
// turns a merely-wrong file into an unparseable one.

import "core:strings"
import "core:testing"

import lc "../core"

// The class table is a package-level array rather than a literal inside the
// helper: a slice of a compound literal would point at a dead stack frame.
@(private = "file")
CLASS_NAMES := [3]string{"", "chair", "table"}

@(private = "file")
test_frame :: proc() -> COCO_Frame {
	return COCO_Frame {
		image_name = "frame.0001.png",
		image_id = 7,
		width = 640,
		height = 480,
		class_names = CLASS_NAMES[:],
		intrinsics = lc.Camera_Intrinsics{fx = 500, fy = 500, cx = 320, cy = 240, width = 640, height = 480},
	}
}

@(test)
test_categories_come_from_the_class_table :: proc(t: ^testing.T) {
	// One annotation, of a chair. The table must still declare `table`: a
	// consumer that built its label map from this frame and a different one
	// from the next would disagree about what category 2 means.
	anns := []lc.Annotation {
		{instance_id = 3, semantic_id = 1, path = "/World/Chair", bbox = {10, 20, 30, 40}, pixel_area = 900},
	}
	json := coco_encode(test_frame(), anns, context.temp_allocator)

	testing.expect(t, strings.contains(json, `{"id": 1, "name": "chair"`), "chair category")
	testing.expect(t, strings.contains(json, `{"id": 2, "name": "table"`), "table category, though nothing here is one")
	// Background is a class id but never a COCO category.
	testing.expect(t, !strings.contains(json, `"id": 0, "name": ""`), "background must not be a category")

	testing.expect(t, strings.contains(json, `"image_id": 7, "category_id": 1`), "annotation identity")
	testing.expect(t, strings.contains(json, `"bbox": [10, 20, 30, 40]`), "box")
	testing.expect(t, strings.contains(json, `"area": 900`), "area")
	testing.expect(t, strings.contains(json, `"file_name": "frame.0001.png"`), "image name")
	testing.expect(t, strings.contains(json, `"fx": 500.000000`), "intrinsics")
}

@(test)
test_paths_are_escaped :: proc(t: ^testing.T) {
	anns := []lc.Annotation {
		{instance_id = 1, semantic_id = 1, path = `/World/od"d\prim`, bbox = {0, 0, 1, 1}, pixel_area = 1},
	}
	json := coco_encode(test_frame(), anns, context.temp_allocator)
	testing.expect(t, strings.contains(json, `"prim_path": "/World/od\"d\\prim"`), "quote and backslash escaped")
}

// A frame where nothing is visible is still a valid frame: it says so, rather
// than being absent, so a training set knows the negative exists.
@(test)
test_empty_frame_is_still_a_document :: proc(t: ^testing.T) {
	json := coco_encode(test_frame(), nil, context.temp_allocator)
	testing.expect(t, strings.contains(json, `"annotations": [`), "annotations key")
	testing.expect(t, strings.contains(json, `"images": [`), "images key")
	// No trailing comma anywhere — the failure mode of hand-written JSON.
	for i in 0 ..< len(json) - 1 {
		if json[i] == ',' {
			j := i + 1
			for j < len(json) && (json[j] == ' ' || json[j] == '\n') {
				j += 1
			}
			if j < len(json) && (json[j] == ']' || json[j] == '}') {
				testing.fail_now(t, "trailing comma in the emitted JSON")
			}
		}
	}
}

@(test)
test_unlabelled_annotations_are_not_exported :: proc(t: ^testing.T) {
	anns := []lc.Annotation{{instance_id = 1, semantic_id = 0}, {instance_id = 2, semantic_id = 1}}
	text := coco_encode(test_frame(), anns, context.temp_allocator)
	testing.expect(t, !strings.contains(text, `"category_id": 0`))
	testing.expect(t, strings.contains(text, `"instance_id": 2`))
}

@(test)
test_annotation_ids_differ_between_images :: proc(t: ^testing.T) {
	anns := []lc.Annotation{{instance_id = 1, semantic_id = 1}}
	a := test_frame()
	b := a
	b.image_id += 1
	ta := coco_encode(a, anns, context.temp_allocator)
	tb := coco_encode(b, anns, context.temp_allocator)
	testing.expect(t, ta != tb)
	testing.expect(t, dataset_id("7:1") != dataset_id("8:1"))
}
