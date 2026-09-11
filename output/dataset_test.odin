package output

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:testing"
import lc "../core"

@(test)
test_class_manifest_survives_new_scene_and_rejects_reassignment :: proc(t: ^testing.T) {
	tmp, _ := os.temp_dir(context.temp_allocator)
	path, _ := filepath.join({tmp, "lumbre_class_registry_test.png"}, context.temp_allocator)
	manifest := fmt.tprintf("%s.classes.json", path[:len(path)-4])
	_ = os.remove(manifest)
	defer os.remove(manifest)
	first := lc.Scene{meshes = []lc.Mesh{{semantic_class = "chair"}, {semantic_class = "table"}}}
	second := lc.Scene{meshes = []lc.Mesh{{semantic_class = "table"}}}
	defer {
		for name in first.semantic_classes { if name != "" { delete(name) } }
		for name in second.semantic_classes { if name != "" { delete(name) } }
		delete(first.semantic_classes)
		delete(second.semantic_classes)
	}
	testing.expect(t, dataset_classes(&first, path))
	testing.expect_value(t, first.meshes[1].semantic_class_id, 2)
	// No shared session or class table: only the on-disk manifest connects them.
	testing.expect(t, dataset_classes(&second, path))
	testing.expect_value(t, second.meshes[0].semantic_class_id, 2)
	testing.expect(t, !dataset_classes(&second, path, {"table", "chair"}))
	testing.expect_value(t, second.meshes[0].semantic_class_id, 2)
}
