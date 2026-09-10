package lumbre_realtime

// Checks the annotations derived from a label frame.
//
//   odin test realtime
//
// The box is the part that has to be exactly right. A 2D box that is one pixel
// short on any side clips the object it labels; one pixel long includes
// background, and both errors survive every eyeball check on a 1080p frame
// while quietly capping what a detector trained on the data can learn. So the
// test states the property directly: the box contains every pixel carrying the
// id, and no pixel outside it carries that id.
//
// The 3D bounds are tested under rotation specifically, because transforming
// the corners of the local AABB — the obvious shortcut — is wrong there and
// right everywhere else.

import "core:math"
import "core:testing"

import m "core:math/linalg/glsl"

import lc "../core"

// A right triangle in the XY plane. Deliberately not a quad: its local AABB
// has a corner at (1, 1) that no vertex occupies, which is what makes the
// rotated-bounds test below able to tell an exact bound from the
// transform-the-eight-corners shortcut. For a box-shaped mesh the two agree
// and the test would pass either way.
@(private = "file")
unit_tri :: proc() -> []lc.Triangle {
	tris := make([]lc.Triangle, 1)
	tris[0] = lc.Triangle{v0 = {0, 0, 0}, v1 = {1, 0, 0}, v2 = {0, 1, 0}, mat_idx = 0}
	return tris
}

// Two meshes on two nodes, so instance ids 1 and 2 both resolve to geometry.
// Node 0 is the root and carries none, which is the real layout: the root is
// never a labelled object.
@(private = "file")
make_annotation_scene :: proc(xform: m.mat4) -> lc.Scene {
	meshes := make([]lc.Mesh, 2)
	meshes[0] = lc.Mesh {
		path = "/World/Chair",
		semantic_class_id = 1,
		triangles = unit_tri(),
		transform = m.mat4(1),
	}
	meshes[1] = lc.Mesh {
		path = "/World/Table",
		semantic_class_id = 2,
		triangles = unit_tri(),
		transform = m.mat4(1),
	}

	nodes := make([]lc.SceneNode, 3)
	nodes[0] = lc.SceneNode{local_transform = m.mat4(1), mesh_idx = -1, material_override_idx = -1, parent = -1}
	nodes[1] = lc.SceneNode{local_transform = xform, mesh_idx = 0, material_override_idx = -1, parent = 0}
	nodes[2] = lc.SceneNode{local_transform = m.mat4(1), mesh_idx = 1, material_override_idx = -1, parent = 0}

	classes := make([]string, 3)
	classes[0] = ""
	classes[1] = "chair"
	classes[2] = "table"

	return lc.Scene{meshes = meshes, nodes = nodes, semantic_classes = classes}
}

@(private = "file")
destroy_annotation_scene :: proc(scene: ^lc.Scene) {
	for mesh in scene.meshes {
		delete(mesh.triangles)
	}
	delete(scene.meshes)
	delete(scene.nodes)
	delete(scene.semantic_classes)
}

// An 8×4 frame with id 1 in a 3×2 block at (2,1) and id 2 in a single pixel at
// (7,0) — one object with area, one at the very edge where an off-by-one in
// the box would run out of the image.
@(private = "file")
painted_frame :: proc() -> Label_Frame {
	w, h :: 8, 4
	f := Label_Frame {
		width = w,
		height = h,
		instance = make([]u32, w * h),
	}
	for y in 1 ..= 2 {
		for x in 2 ..= 4 {
			f.instance[y * w + x] = 1
		}
	}
	f.instance[0 * w + 7] = 2
	return f
}

@(test)
test_boxes_bound_exactly_their_pixels :: proc(t: ^testing.T) {
	frame := painted_frame()
	defer delete(frame.instance)
	scene := make_annotation_scene(m.mat4(1))
	defer destroy_annotation_scene(&scene)

	anns := annotations_derive(frame, &scene)
	defer delete(anns)

	testing.expect_value(t, len(anns), 2)
	if len(anns) != 2 {
		return
	}

	testing.expect_value(t, anns[0].instance_id, 1)
	testing.expect_value(t, anns[0].bbox, [4]i32{2, 1, 3, 2})
	testing.expect_value(t, anns[0].pixel_area, 6)
	testing.expect_value(t, anns[0].semantic_id, 1)
	testing.expect_value(t, anns[0].class_name, "chair")
	testing.expect_value(t, anns[0].path, "/World/Chair")

	// The single edge pixel: a box of width 1, not 0, and not running past the
	// last column.
	testing.expect_value(t, anns[1].instance_id, 2)
	testing.expect_value(t, anns[1].bbox, [4]i32{7, 0, 1, 1})
	testing.expect_value(t, anns[1].pixel_area, 1)
	testing.expect_value(t, anns[1].class_name, "table")

	// The property, stated over every pixel rather than over the two boxes
	// above: containment in both directions.
	for a in anns {
		inside := 0
		for y in 0 ..< int(frame.height) {
			for x in 0 ..< int(frame.width) {
				id := frame.instance[y * int(frame.width) + x]
				in_box :=
					i32(x) >= a.bbox[0] && i32(x) < a.bbox[0] + a.bbox[2] &&
					i32(y) >= a.bbox[1] && i32(y) < a.bbox[1] + a.bbox[3]
				if id == a.instance_id {
					testing.expect(t, in_box, "a pixel with this id falls outside its box")
					inside += 1
				}
			}
		}
		testing.expect_value(t, i32(inside), a.pixel_area)
	}
}

// Background is never an object, however much of the frame it covers.
@(test)
test_background_is_not_annotated :: proc(t: ^testing.T) {
	frame := Label_Frame{width = 4, height = 4, instance = make([]u32, 16)}
	defer delete(frame.instance)
	scene := make_annotation_scene(m.mat4(1))
	defer destroy_annotation_scene(&scene)

	anns := annotations_derive(frame, &scene)
	defer delete(anns)
	testing.expect_value(t, len(anns), 0)
}

@(test)
test_world_bounds_follow_the_node :: proc(t: ^testing.T) {
	// 45° about Z. Every vertex lands at y = √2/2 or below, while the local
	// AABB's unoccupied (1, 1) corner swings up to y = √2 — so a bound built
	// from the corners reports a box twice as tall as the triangle.
	angle := f32(math.PI / 4)
	rot := m.mat4Rotate({0, 0, 1}, angle)

	frame := painted_frame()
	defer delete(frame.instance)
	scene := make_annotation_scene(rot)
	defer destroy_annotation_scene(&scene)

	anns := annotations_derive(frame, &scene)
	defer delete(anns)
	if len(anns) == 0 {
		testing.fail_now(t, "no annotations")
	}

	a := anns[0]
	testing.expect(t, a.has_pose, "a node-backed instance should carry a pose")

	eps :: 1e-5
	root2 := math.sqrt(f32(2))
	testing.expect(t, abs(a.bounds_min.x - (-root2 / 2)) < eps, "min x")
	testing.expect(t, abs(a.bounds_max.x - (root2 / 2)) < eps, "max x")
	testing.expect(t, abs(a.bounds_min.y - 0) < eps, "min y")
	testing.expect(t, abs(a.bounds_max.y - root2 / 2) < eps, "max y")
}

// The intrinsics must describe the frame that was actually rendered: half the
// vertical extent subtends half the field of view.
@(test)
test_intrinsics_match_the_projection :: proc(t: ^testing.T) {
	cam := lc.make_camera(
		lookfrom = {0, 0, 5},
		lookat = {0, 0, 0},
		vup = {0, 1, 0},
		vfov = 60.0,
		aspect_ratio = 16.0 / 9.0,
		aperture = 0.0,
		focus_dist = 5.0,
	)
	k := camera_intrinsics(cam, 1920, 1080)

	eps :: 1e-2
	expected_fy := f32(1080) * 0.5 / math.tan(f32(math.PI / 6))
	testing.expect(t, abs(k.fy - expected_fy) < eps, "fy from the vertical fov")
	// Square pixels: the projection scales x by the aspect, and the frame is
	// rendered at the aspect the camera was built for.
	testing.expect(t, abs(k.fx - k.fy) < eps, "fx and fy should agree")
	testing.expect_value(t, k.cx, 960)
	testing.expect_value(t, k.cy, 540)
}
