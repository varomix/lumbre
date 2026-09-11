package lumbre_realtime

// Conservative homogeneous clip test, shared by beauty, labels and cascades.
// Keep an AABB unless all eight corners lie outside the same clip plane.
bounds_visible :: proc(lo, hi: [3]f32, view_proj: matrix[4, 4]f32) -> bool {
	outside := [6]bool{true, true, true, true, true, true}
	for i in 0 ..< 8 {
		p := view_proj * [4]f32{i & 1 == 0 ? lo.x : hi.x, i & 2 == 0 ? lo.y : hi.y, i & 4 == 0 ? lo.z : hi.z, 1}
		outside[0] &&= p.x < -p.w
		outside[1] &&= p.x > p.w
		outside[2] &&= p.y < -p.w
		outside[3] &&= p.y > p.w
		outside[4] &&= p.z < 0
		outside[5] &&= p.z > p.w
	}
	for rejected in outside { if rejected { return false } }
	return true
}

// Coalesce adjacent surviving ranges. An entirely visible scene retains one
// draw per material (or one depth/label draw), rather than paying a draw per
// object merely to make culling possible.
visible_range :: proc(batches: []Draw_Batch, start: int, vp: matrix[4, 4]f32, same_material: bool) -> (first, next: int, vertices: u32) {
	first = start
	for first < len(batches) && !bounds_visible(batches[first].bounds_min, batches[first].bounds_max, vp) { first += 1 }
	if first == len(batches) { return first, first, 0 }
	vertices = batches[first].vertex_count
	next = first + 1
	for next < len(batches) {
		b := batches[next]
		if same_material && b.material_index != batches[first].material_index { break }
		if !bounds_visible(b.bounds_min, b.bounds_max, vp) { break }
		vertices += b.vertex_count
		next += 1
	}
	return
}
