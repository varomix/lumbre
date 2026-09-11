package lumbre_realtime

import sdl "vendor:sdl3"

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

// Culls every instance against one view, once per pass. A mesh drawn with
// several materials reads the same result for each of its batches.
scene_mark_visible :: proc(s: ^Scene_GPU, view_proj: matrix[4, 4]f32) {
	resize(&s.visible, len(s.instance_bounds))
	for b, i in s.instance_bounds {
		s.visible[i] = bounds_visible(b.lo, b.hi, view_proj)
	}
}

// The next run of consecutive visible instances in [start, end), which draws as
// one instanced call. `count` is 0 when none remain.
instance_run :: proc(visible: []bool, start, end: int) -> (first, count: int) {
	first = start
	for first < end && !visible[first] { first += 1 }
	next := first
	for next < end && visible[next] { next += 1 }
	return first, next - first
}

// Vertices on slot 0, instances on slot 1, as `vertex_input_state` declares.
scene_bind_vertex_buffers :: proc(pass: ^sdl.GPURenderPass, s: ^Scene_GPU) {
	bindings := [2]sdl.GPUBufferBinding{{buffer = s.vertices}, {buffer = s.instances}}
	sdl.BindGPUVertexBuffers(pass, 0, raw_data(&bindings), len(bindings))
}

// Draws the visible instances for a pass that ignores material -- shadow depth
// and labels. A mesh's material runs are contiguous and share its instances, so
// they merge into one vertex range. Call `scene_mark_visible` first.
scene_draw_ignoring_material :: proc(pass: ^sdl.GPURenderPass, s: ^Scene_GPU) {
	for bi := 0; bi < len(s.batches); {
		b := s.batches[bi]
		vertex_count := b.vertex_count
		bi += 1
		for bi < len(s.batches) &&
		    s.batches[bi].first_instance == b.first_instance &&
		    s.batches[bi].first_vertex == b.first_vertex + vertex_count {
			vertex_count += s.batches[bi].vertex_count
			bi += 1
		}

		start := int(b.first_instance)
		end := start + int(b.instance_count)
		for cursor := start; cursor < end; {
			first, count := instance_run(s.visible[:], cursor, end)
			if count == 0 { break }
			cursor = first + count
			sdl.DrawGPUPrimitives(pass, vertex_count, u32(count), b.first_vertex, u32(first))
		}
	}
}
