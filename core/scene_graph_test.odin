package lumbre_core

// Checks semantic class interning.
//
//   odin test core
//
// The ids these produce end up in a segmentation mask that a training run
// consumes, so two properties matter beyond "it works": the unlabelled class
// must be id 0, because a label buffer cleared to zero has to read as "nothing
// here" rather than as the first real class; and the table must be ordered the
// same way every run, because a dataset's frames are rendered across many
// invocations and a class id that shifts between them silently corrupts every
// label already written.

import "core:testing"

import m "core:math/linalg/glsl"

@(private = "file")
scene_with_classes :: proc(classes: []string) -> Scene {
	meshes := make([]Mesh, len(classes))
	for c, i in classes {
		meshes[i] = Mesh{semantic_class = c, transform = m.mat4(1)}
	}
	return Scene{meshes = meshes}
}

@(private = "file")
free_scene :: proc(s: ^Scene) {
	delete(s.meshes)
	delete(s.semantic_classes)
}

@(test)
test_unlabelled_is_class_zero :: proc(t: ^testing.T) {
	scene := scene_with_classes({"", "chair", ""})
	defer free_scene(&scene)

	scene_build_semantic_classes(&scene)

	testing.expectf(t, scene.semantic_classes[0] == "", "class 0 must be the unlabelled one")
	testing.expectf(t, scene.meshes[0].semantic_class_id == 0, "unlabelled mesh got id %d", scene.meshes[0].semantic_class_id)
	testing.expectf(t, scene.meshes[2].semantic_class_id == 0, "unlabelled mesh got id %d", scene.meshes[2].semantic_class_id)
	testing.expectf(t, scene.meshes[1].semantic_class_id == 1, "first real class must be 1, got %d", scene.meshes[1].semantic_class_id)
}

@(test)
test_classes_are_interned_not_repeated :: proc(t: ^testing.T) {
	scene := scene_with_classes({"chair", "table", "chair", "table", "lamp"})
	defer free_scene(&scene)

	scene_build_semantic_classes(&scene)

	// Three distinct classes plus the unlabelled entry.
	testing.expectf(t, len(scene.semantic_classes) == 4, "table has %d entries, want 4", len(scene.semantic_classes))

	// Equal names must share an id, or the same object type splits into
	// several categories across a dataset.
	testing.expectf(
		t, scene.meshes[0].semantic_class_id == scene.meshes[2].semantic_class_id,
		"two 'chair' meshes got ids %d and %d",
		scene.meshes[0].semantic_class_id, scene.meshes[2].semantic_class_id,
	)
	testing.expectf(
		t, scene.meshes[1].semantic_class_id == scene.meshes[3].semantic_class_id,
		"two 'table' meshes got different ids",
	)
	testing.expect(
		t, scene.meshes[0].semantic_class_id != scene.meshes[1].semantic_class_id,
		"distinct classes must not collide",
	)

	// Every id must index the table it was built from.
	for mesh in scene.meshes {
		id := int(mesh.semantic_class_id)
		testing.expectf(t, id >= 0 && id < len(scene.semantic_classes), "id %d outside the table", id)
		testing.expectf(
			t, scene.semantic_classes[id] == mesh.semantic_class,
			"id %d resolves to %q, want %q", id, scene.semantic_classes[id], mesh.semantic_class,
		)
	}
}

@(test)
test_class_ids_are_first_seen_order :: proc(t: ^testing.T) {
	// Deliberately not alphabetical: the order must follow the meshes, so that
	// the same stage yields the same table on every run and in every process.
	names := []string{"zebra", "apple", "mango"}
	scene := scene_with_classes(names)
	defer free_scene(&scene)

	scene_build_semantic_classes(&scene)

	for name, i in names {
		want := i32(i + 1) // 0 is unlabelled
		testing.expectf(
			t, scene.meshes[i].semantic_class_id == want,
			"%q got id %d, want %d (first-seen order)", name, scene.meshes[i].semantic_class_id, want,
		)
	}
	testing.expectf(t, scene.semantic_classes[1] == "zebra", "table[1] = %q", scene.semantic_classes[1])
}
