package lumbre_realtime

import sdl "vendor:sdl3"

// The six clip planes of `view_proj` in world space, as (normal, offset) with
// the inside positive: -w <= x, y <= w and 0 <= z <= w, SDL_GPU's clip volume.
Frustum :: [6][4]f32

frustum_planes :: proc(view_proj: matrix[4, 4]f32) -> Frustum {
	row :: proc(m: matrix[4, 4]f32, i: int) -> [4]f32 {
		return {m[i, 0], m[i, 1], m[i, 2], m[i, 3]}
	}
	x, y, z, w := row(view_proj, 0), row(view_proj, 1), row(view_proj, 2), row(view_proj, 3)
	return {w + x, w - x, w + y, w - y, z, w - z}
}

// Conservative clip test, shared by beauty, labels and cascades. Rejects an
// AABB only when it lies wholly outside one plane: exactly the old test of
// whether all eight corners fail the same clip inequality, for one dot product
// per plane instead of eight matrix products.
bounds_visible_planes :: proc(lo, hi: [3]f32, planes: ^Frustum) -> bool {
	for p in planes {
		// The corner furthest along the plane normal.
		far := [3]f32{p.x >= 0 ? hi.x : lo.x, p.y >= 0 ? hi.y : lo.y, p.z >= 0 ? hi.z : lo.z}
		if p.x * far.x + p.y * far.y + p.z * far.z + p.w < 0 {
			return false
		}
	}
	return true
}

bounds_visible :: proc(lo, hi: [3]f32, view_proj: matrix[4, 4]f32) -> bool {
	planes := frustum_planes(view_proj)
	return bounds_visible_planes(lo, hi, &planes)
}

// Culls every instance against one view. A mesh drawn with several materials
// reads the same result for each of its batches, and consecutive passes over
// one view -- labels, G-buffer, transparency -- share one result: it is kept
// until a different view or an instance update replaces it.
scene_mark_visible :: proc(s: ^Scene_GPU, view_proj: matrix[4, 4]f32) {
	if s.visible_valid && s.visible_view_proj == view_proj && len(s.visible) == len(s.instance_bounds) {
		return
	}
	resize(&s.visible, len(s.instance_bounds))
	planes := frustum_planes(view_proj)
	for b, i in s.instance_bounds {
		s.visible[i] = bounds_visible_planes(b.lo, b.hi, &planes)
	}
	s.visible_view_proj = view_proj
	s.visible_valid = true
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

// Vertices on slot 0, instances on slot 1, as `vertex_input_state` declares,
// and the index buffer every scene draw reads.
scene_bind_vertex_buffers :: proc(pass: ^sdl.GPURenderPass, s: ^Scene_GPU) {
	bindings := [2]sdl.GPUBufferBinding{{buffer = s.vertices}, {buffer = s.instances}}
	sdl.BindGPUVertexBuffers(pass, 0, raw_data(&bindings), len(bindings))
	index_binding := sdl.GPUBufferBinding{buffer = s.indices}
	sdl.BindGPUIndexBuffer(pass, index_binding, ._32BIT)
}

// Draws the visible instances for a pass that ignores material -- shadow depth
// and labels. A mesh's material runs are contiguous and share its instances, so
// they merge into one index range. Call `scene_mark_visible` first.
scene_draw_ignoring_material :: proc(pass: ^sdl.GPURenderPass, s: ^Scene_GPU) {
	for bi := 0; bi < len(s.batches); {
		b := s.batches[bi]
		index_count := b.index_count
		bi += 1
		for bi < len(s.batches) &&
		    s.batches[bi].first_instance == b.first_instance &&
		    s.batches[bi].base_vertex == b.base_vertex &&
		    s.batches[bi].first_index == b.first_index + index_count {
			index_count += s.batches[bi].index_count
			bi += 1
		}

		start := int(b.first_instance)
		end := start + int(b.instance_count)
		for cursor := start; cursor < end; {
			first, count := instance_run(s.visible[:], cursor, end)
			if count == 0 { break }
			cursor = first + count
			sdl.DrawGPUIndexedPrimitives(pass, index_count, u32(count), b.first_index, b.base_vertex, u32(first))
		}
	}
}
