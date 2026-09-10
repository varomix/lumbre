package lumbre_core

// The shared vocabulary for ground-truth labels.
//
// These types live here, next to `Scene` and `Camera`, because they are the
// handoff between two packages that must not know about each other: `realtime`
// produces them on the GPU and has no business writing files, while `output`
// writes them and has no business linking SDL. Nothing in this file renders,
// allocates a texture, or touches a path — it is plain CPU data, which is what
// makes it safe for `core`, and it keeps the Houdini bridge's dependency
// surface unchanged.

// One frame of ground truth, top-row-first — the order the GPU hands textures
// back, and the opposite of the path tracer's bottom-row-first beauty buffer.
// Whoever writes it to a file owns the flip.
Label_Frame :: struct {
	width:    i32,
	height:   i32,
	instance: []u32,
	semantic: []u32,
	depth:    []f32,    // metres along the view axis; 0 is background
	normal:   [][4]f32, // world space, raw [-1, 1]
}

// One labelled object in one frame.
Annotation :: struct {
	instance_id: u32,
	semantic_id: i32,
	// Prim path and class name, resolved so a writer needs no scene access.
	// Empty for geometry with neither (spheres, OBJ meshes).
	path:        string,
	class_name:  string,

	// COCO's convention: [x, y, width, height] in pixels, origin at the top
	// left, matching the label frame's own row order.
	bbox:        [4]i32,
	pixel_area:  i32,

	// World-space axis-aligned bounds and the object's world transform.
	// `has_pose` is false for geometry that is not a scene node — spheres —
	// which have an id and pixels but no transform to report.
	bounds_min:  [3]f32,
	bounds_max:  [3]f32,
	pose:        matrix[4, 4]f32,
	has_pose:    bool,
}

// The pinhole model of the camera that rendered a frame, in pixels.
Camera_Intrinsics :: struct {
	fx, fy: f32,
	cx, cy: f32,
	width:  i32,
	height: i32,
}
