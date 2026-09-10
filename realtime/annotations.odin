package lumbre_realtime

// Turning a label frame into annotations.
//
// Almost everything a dataset needs falls out of two sources that already
// exist: one pass over the instance-id buffer gives the 2D extent and pixel
// area of every object actually visible, and the scene graph gives the 3D
// extent and pose of the object that produced it. Nothing here renders, and
// nothing here touches the GPU.
//
// Only *visible* instances are annotated. An annotation file describes a
// frame, and an object behind the camera or fully occluded contributes no
// pixels to label; emitting a box for it would teach a detector to find
// something that is not there. Occlusion percentage — which needs a second,
// unoccluded pass — is deliberately out of this phase, so a partly hidden
// object here carries the box of its *visible* pixels, not of its silhouette.

import "core:math"

import lc "../core"

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

// The pinhole model of the camera that rendered the frame, in pixels.
//
// Derived from the same `Camera_Frame` the projection matrix uses, so the
// intrinsics describe the image that was actually rendered rather than a
// nominal sensor. Real focal lengths and apertures do survive USD import
// (`importers/usd_camera.odin`) but are collapsed to a vfov there; recovering
// them is a separate job from this one.
Camera_Intrinsics :: struct {
	fx, fy: f32,
	cx, cy: f32,
	width:  i32,
	height: i32,
}

camera_intrinsics :: proc(cam: lc.Camera, width, height: i32) -> Camera_Intrinsics {
	f := camera_frame(cam)
	// Pixels are square: the projection derives its horizontal scale from the
	// vertical one and the aspect, so fx and fy differ only if the frame is
	// rendered at an aspect the camera was not built for.
	fy := f32(height) * 0.5 / math.tan(f.vfov * 0.5)
	fx := f32(width) * 0.5 / (f.aspect * math.tan(f.vfov * 0.5))
	return Camera_Intrinsics {
		fx = fx,
		fy = fy,
		cx = f32(width) * 0.5,
		cy = f32(height) * 0.5,
		width = width,
		height = height,
	}
}

// Derives one annotation per visible instance, ordered by instance id.
//
// `scene` must be the scene the frame was rendered from: instance ids are
// scene node indices, so an annotation set outlives its scene by exactly zero
// edits. The returned strings are borrowed from the scene, not copied.
annotations_derive :: proc(
	frame: Label_Frame,
	scene: ^lc.Scene,
	allocator := context.allocator,
) -> []Annotation {
	texels := int(frame.width) * int(frame.height)
	if texels <= 0 || len(frame.instance) < texels {
		return nil
	}

	// One entry per id the vertex stream can carry, indexed directly. A map
	// would be smaller for a sparse frame and slower for every frame: this is
	// one pass over a few million texels, and the id is already the index.
	id_count := len(scene.nodes) + len(scene.spheres)
	if id_count == 0 {
		return nil
	}

	Extent :: struct {
		min_x, min_y, max_x, max_y: i32,
		area:                       i32,
	}
	extents := make([]Extent, id_count, context.temp_allocator)
	for &e in extents {
		e = {min_x = max(i32), min_y = max(i32), max_x = min(i32), max_y = min(i32)}
	}

	width := int(frame.width)
	for y in 0 ..< int(frame.height) {
		row := y * width
		for x in 0 ..< width {
			id := int(frame.instance[row + x])
			// 0 is the unlabelled background, and an id past the table is a
			// frame from a different scene — neither is an object here.
			if id == 0 || id >= id_count {
				continue
			}
			e := &extents[id]
			e.min_x = min(e.min_x, i32(x))
			e.min_y = min(e.min_y, i32(y))
			e.max_x = max(e.max_x, i32(x))
			e.max_y = max(e.max_y, i32(y))
			e.area += 1
		}
	}

	// World transforms are needed for the 3D bounds and are recomputed rather
	// than assumed: the caller may have edited the graph since the last
	// flatten, and this is a handful of matrix multiplies.
	lc.compute_world_transforms(scene.nodes)

	out := make([dynamic]Annotation, allocator)
	for e, id in extents {
		if e.area == 0 {
			continue
		}
		a := Annotation {
			instance_id = u32(id),
			bbox = {e.min_x, e.min_y, e.max_x - e.min_x + 1, e.max_y - e.min_y + 1},
			pixel_area = e.area,
		}

		if id < len(scene.nodes) {
			node := scene.nodes[id]
			a.pose = node.world_transform
			a.has_pose = true
			if node.mesh_idx >= 0 && int(node.mesh_idx) < len(scene.meshes) {
				mesh := &scene.meshes[node.mesh_idx]
				a.semantic_id = mesh.semantic_class_id
				a.path = mesh.path
				if int(a.semantic_id) < len(scene.semantic_classes) {
					a.class_name = scene.semantic_classes[a.semantic_id]
				}
				a.bounds_min, a.bounds_max = mesh_world_bounds(mesh, node.world_transform)
			}
		}
		append(&out, a)
	}
	return out[:]
}

// The world-space AABB of a mesh under a transform.
//
// Transforms every vertex rather than the eight corners of the local AABB: the
// corner shortcut is exact only for axis-aligned box bounds under an affine
// map, and produces a box that is too large under rotation — which is a wrong
// 3D label, not a conservative one.
@(private = "file")
mesh_world_bounds :: proc(mesh: ^lc.Mesh, xform: matrix[4, 4]f32) -> (lo, hi: [3]f32) {
	if len(mesh.triangles) == 0 {
		return {}, {}
	}
	lo = {max(f32), max(f32), max(f32)}
	hi = {min(f32), min(f32), min(f32)}

	for tri in mesh.triangles {
		for v in ([3]lc.Point3{tri.v0, tri.v1, tri.v2}) {
			p := vec3f(lc.transform_point(v, xform))
			lo = {min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z)}
			hi = {max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z)}
		}
	}
	return lo, hi
}
