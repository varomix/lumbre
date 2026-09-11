package lumbre_realtime

// Clustered light assignment.
//
// The deferred pass used to test every light at every pixel, which is fine for
// the handful of lights a test scene carries and hopeless for a room with
// fifty. The view frustum is divided into froxels -- screen tiles in x and y,
// logarithmic slices in depth -- and each froxel gets the list of lights that
// can reach it. A pixel then looks up its own froxel and iterates that list.
//
// Assignment SCATTERS: each light writes itself into the cells it covers,
// rather than each cell testing every light. Cell-major was the obvious way to
// write it and was measured at twice the cost of no clustering at all -- 3,072
// cells times every light, every frame, which for 200 lights outweighed the
// shading it saved. Light-major touches only the cells a light reaches.
//
// Assignment runs on the CPU. A compute pass would scale further, but there is
// no compute pipeline in this renderer yet, and the shape of the data is the
// same either way, so moving it later changes no shader.
//
// Lights are scattered in ascending index order, so a pixel accumulates its
// lights in exactly the order the unclustered loop did: when every light
// reaches every cluster the image is bit-identical, which is what makes this
// testable against the previous renderer.

import "core:fmt"
import "core:math"
import "core:os"

import sdl "vendor:sdl3"

// LUMBRE_CLUSTER_STATS=1 reports what the grid actually holds, which is the
// only way to tell culling from busywork: a scene whose lights all reach
// everything pays for assignment and saves nothing.
@(private = "file")
stats_enabled := false
@(private = "file")
stats_checked := false

// Froxel counts. 16x8 tiles matches the usual 80-120 px tile at viewport sizes,
// and 24 depth slices is the common choice -- enough that a slice is thin near
// the camera, few enough that the grid stays small.
CLUSTER_X :: 16
CLUSTER_Y :: 8
CLUSTER_Z :: 24
CLUSTER_COUNT :: CLUSTER_X * CLUSTER_Y * CLUSTER_Z

// A cell keeps at most this many lights, which bounds the index buffer at
// 3 MB. Past it the highest-indexed lights are dropped; that needs more than
// 256 lights overlapping one froxel, and the alternative -- keeping the
// strongest -- costs a sort per cell and makes neighbouring cells disagree
// about which lights exist, which reads as a visible tile edge.
MAX_LIGHTS_PER_CLUSTER :: 256

// A light is dropped from a cell once the radiance it could deliver there falls
// below this. Chosen well under one 8-bit display step (1/255 ~ 0.004) so
// culling cannot remove anything the display could show.
CLUSTER_CUTOFF :: 0.001

// Where one cell's lights live in `Cluster_Grid.indices`.
Cluster_Range :: struct {
	offset: u32,
	count:  u32,
}

Cluster_Grid :: struct {
	ranges:  [dynamic]Cluster_Range,
	indices: [dynamic]u32,

	range_buffer:   ^sdl.GPUBuffer,
	index_buffer:   ^sdl.GPUBuffer,
	range_capacity: u32,
	index_capacity: u32,
}

// Near and far distances the slices span, matching the projection the pass
// renders with (camera_projection in camera.odin).
cluster_near_far :: proc(f: Camera_Frame) -> (near, far: f32) {
	return f.focus * NEAR_SCALE, f.focus * FAR_SCALE
}

// near, far, the log scale that maps a view depth to a slice, unused.
cluster_depth_params :: proc(f: Camera_Frame) -> [4]f32 {
	near, far := cluster_near_far(f)
	scale := f32(CLUSTER_Z) / math.ln(max(far / near, 1.0001))
	return {near, far, scale, 0}
}

cluster_dims :: proc() -> [4]f32 {
	return {f32(CLUSTER_X), f32(CLUSTER_Y), f32(CLUSTER_Z), f32(MAX_LIGHTS_PER_CLUSTER)}
}

// A point relative to the camera, as (right, up, forward) distances. Forward is
// positive in front of the camera, which is the same convention the shader's
// `view_depth` uses.
@(private = "file")
to_view :: proc(f: Camera_Frame, p: [3]f32) -> [3]f32 {
	d := [3]f32{p.x - f.eye.x, p.y - f.eye.y, p.z - f.eye.z}
	return {dot(d, f.right), dot(d, f.up), dot(d, f.forward)}
}

@(private = "file")
dot :: proc(a, b: [3]f32) -> f32 {
	return a.x * b.x + a.y * b.y + a.z * b.z
}

// The view-space box of one cell. The cell is a frustum slab; its axis-aligned
// bounds are taken from the corners at both depths, which is conservative --
// it can only keep a light a tighter test would drop.
@(private = "file")
cell_bounds :: proc(f: Camera_Frame, x, y: int, z0, z1: f32) -> (lo, hi: [3]f32) {
	tan_half := math.tan(f.vfov * 0.5)
	// NDC edges of the tile. y is flipped because the shader's uv runs top-down.
	nx0 := f32(x) / f32(CLUSTER_X) * 2 - 1
	nx1 := f32(x + 1) / f32(CLUSTER_X) * 2 - 1
	ny1 := 1 - f32(y) / f32(CLUSTER_Y) * 2
	ny0 := 1 - f32(y + 1) / f32(CLUSTER_Y) * 2

	lo = {max(f32), max(f32), z0}
	hi = {min(f32), min(f32), z1}
	for z in ([2]f32{z0, z1}) {
		h := z * tan_half
		w := h * f.aspect
		for nx in ([2]f32{nx0, nx1}) {
			lo.x = min(lo.x, nx * w)
			hi.x = max(hi.x, nx * w)
		}
		for ny in ([2]f32{ny0, ny1}) {
			lo.y = min(lo.y, ny * h)
			hi.y = max(hi.y, ny * h)
		}
	}
	return
}

// The distance past which a light contributes less than CLUSTER_CUTOFF, and the
// point it radiates from. Mirrors `light_sample` in lighting_fs.slang: point and
// spot lights carry radiant intensity and fall off as 1/d^2; the area shapes
// carry radiance and are integrated as radiance * area / d^2.
//
// `infinite` is true for a light with no position -- the sun -- which reaches
// every cell.
cluster_light_reach :: proc(l: Light_GPU) -> (centre: [3]f32, radius: f32, infinite: bool) {
	kind := Light_Kind_GPU(i32(l.params.x))
	if kind == .Distant {
		return {}, 0, true
	}
	power := max(l.emission.x, max(l.emission.y, l.emission.z))
	if kind != .Point && kind != .Spot {
		// Area shapes: emission is radiance, emission.w the area.
		power *= l.emission.w
	}
	centre = {l.position.x, l.position.y, l.position.z}
	if power <= 0 {
		return centre, 0, false
	}
	// A sphere or cylinder light also emits from anywhere on its surface, so
	// its own extent widens the reach.
	extent := l.position.w
	if kind == .Cylinder {
		extent += l.direction.w * 0.5
	}
	return centre, math.sqrt(power / CLUSTER_CUTOFF) + extent, false
}

@(private = "file")
box_distance_squared :: proc(centre: [3]f32, lo, hi: [3]f32) -> f32 {
	d2: f32 = 0
	for k in 0 ..< 3 {
		v := centre[k]
		if v < lo[k] {
			d2 += (lo[k] - v) * (lo[k] - v)
		} else if v > hi[k] {
			d2 += (v - hi[k]) * (v - hi[k])
		}
	}
	return d2
}

// The cells one light covers, appended to `out` as flat cell indices. Cells are
// visited in ascending order, and a light is only listed once per cell.
@(private = "file")
light_cells :: proc(l: Light_GPU, f: Camera_Frame, out: ^[dynamic]u32) {
	centre_world, radius, infinite := cluster_light_reach(l)
	if infinite {
		for c in 0 ..< u32(CLUSTER_COUNT) {
			append(out, c)
		}
		return
	}
	if radius <= 0 {
		return
	}

	near, far := cluster_near_far(f)
	ratio := max(far / near, 1.0001)
	log_ratio := math.ln(ratio)
	tan_half := math.tan(f.vfov * 0.5)

	centre := to_view(f, centre_world)
	z_lo := max(centre.z - radius, near)
	z_hi := min(centre.z + radius, far)
	if z_lo > z_hi {
		return // entirely behind the camera or past the far plane
	}

	slice_of :: proc(z, near, scale: f32) -> int {
		return int(math.ln(max(z / near, 1)) * scale)
	}
	scale := f32(CLUSTER_Z) / log_ratio
	sz0 := clamp(slice_of(z_lo, near, scale), 0, CLUSTER_Z - 1)
	sz1 := clamp(slice_of(z_hi, near, scale), 0, CLUSTER_Z - 1)

	for z in sz0 ..= sz1 {
		zc0 := near * math.pow(ratio, f32(z) / f32(CLUSTER_Z))
		zc1 := near * math.pow(ratio, f32(z + 1) / f32(CLUSTER_Z))

		// Tiles the sphere can touch in this slice. The near edge of the slice
		// has the smallest world extent per tile, so mapping the sphere's x/y
		// span through it covers every tile a farther depth could reach.
		half_h := max(zc0 * tan_half, 1e-6)
		half_w := max(half_h * f.aspect, 1e-6)
		nx_lo := (centre.x - radius) / half_w
		nx_hi := (centre.x + radius) / half_w
		ny_lo := (centre.y - radius) / half_h
		ny_hi := (centre.y + radius) / half_h

		tx0 := clamp(int((nx_lo + 1) * 0.5 * f32(CLUSTER_X)), 0, CLUSTER_X - 1)
		tx1 := clamp(int((nx_hi + 1) * 0.5 * f32(CLUSTER_X)), 0, CLUSTER_X - 1)
		// uv y runs top-down, so the upper bound in view space is the lower tile.
		ty0 := clamp(int((1 - ny_hi) * 0.5 * f32(CLUSTER_Y)), 0, CLUSTER_Y - 1)
		ty1 := clamp(int((1 - ny_lo) * 0.5 * f32(CLUSTER_Y)), 0, CLUSTER_Y - 1)
		if tx1 < tx0 || ty1 < ty0 {
			continue
		}

		for y in ty0 ..= ty1 {
			for x in tx0 ..= tx1 {
				lo, hi := cell_bounds(f, x, y, zc0, zc1)
				if box_distance_squared(centre, lo, hi) > radius * radius {
					continue
				}
				append(out, u32((z * CLUSTER_Y + y) * CLUSTER_X + x))
			}
		}
	}
}

// Fills `ranges` and `indices` for one view. Two passes over the coverage:
// count per cell to lay out the ranges, then write. Lights are visited in
// ascending index order in both, so each cell's list comes out ascending.
clusters_build :: proc(g: ^Cluster_Grid, lights: []Light_GPU, f: Camera_Frame) {
	resize(&g.ranges, CLUSTER_COUNT)
	clear(&g.indices)
	for &r in g.ranges {
		r = {}
	}
	if len(lights) == 0 {
		return
	}

	coverage := make([dynamic]u32, 0, CLUSTER_COUNT, context.temp_allocator)
	defer delete(coverage)
	spans := make([][2]int, len(lights), context.temp_allocator)

	for l, li in lights {
		start := len(coverage)
		light_cells(l, f, &coverage)
		spans[li] = {start, len(coverage) - start}
	}

	// Pass 1: how many lights each cell keeps, capped.
	counts := make([]u32, CLUSTER_COUNT, context.temp_allocator)
	for cell in coverage {
		if counts[cell] < u32(MAX_LIGHTS_PER_CLUSTER) {
			counts[cell] += 1
		}
	}
	total: u32 = 0
	for c, cell in counts {
		g.ranges[cell] = {offset = total, count = 0}
		total += c
	}
	resize(&g.indices, int(total))

	// Pass 2: write, in ascending light order.
	for span, li in spans {
		for k in span[0] ..< span[0] + span[1] {
			cell := coverage[k]
			r := &g.ranges[cell]
			if r.count >= counts[cell] {
				continue // cell full
			}
			g.indices[r.offset + r.count] = u32(li)
			r.count += 1
		}
	}

	clusters_report(g, len(lights))
}

// Average and worst-case occupancy, against the light count a pass would test
// without the grid.
@(private = "file")
clusters_report :: proc(g: ^Cluster_Grid, light_count: int) {
	if !stats_checked {
		stats_checked = true
		stats_enabled = os.get_env("LUMBRE_CLUSTER_STATS", context.temp_allocator) == "1"
	}
	if !stats_enabled {
		return
	}
	worst, empty: u32
	for r in g.ranges {
		worst = max(worst, r.count)
		if r.count == 0 {
			empty += 1
		}
	}
	fmt.printfln(
		"clusters: %d lights, %.1f per cell on average, %d worst, %d of %d cells empty",
		light_count, f64(len(g.indices)) / f64(CLUSTER_COUNT), worst, empty, CLUSTER_COUNT,
	)
}

// Uploads the grid, growing either buffer when it no longer fits. The index
// list is rebuilt every frame, so the buffers are written whole rather than
// patched.
clusters_upload :: proc(gpu: ^sdl.GPUDevice, g: ^Cluster_Grid) -> bool {
	range_bytes := u32(max(len(g.ranges), 1) * size_of(Cluster_Range))
	index_bytes := u32(max(len(g.indices), 1) * size_of(u32))

	if g.range_buffer == nil || g.range_capacity < range_bytes {
		if g.range_buffer != nil {
			sdl.ReleaseGPUBuffer(gpu, g.range_buffer)
		}
		g.range_buffer = sdl.CreateGPUBuffer(
			gpu, sdl.GPUBufferCreateInfo{usage = {.GRAPHICS_STORAGE_READ}, size = range_bytes},
		)
		g.range_capacity = g.range_buffer != nil ? range_bytes : 0
		if g.range_buffer == nil {
			return false
		}
	}
	if g.index_buffer == nil || g.index_capacity < index_bytes {
		if g.index_buffer != nil {
			sdl.ReleaseGPUBuffer(gpu, g.index_buffer)
		}
		g.index_buffer = sdl.CreateGPUBuffer(
			gpu, sdl.GPUBufferCreateInfo{usage = {.GRAPHICS_STORAGE_READ}, size = index_bytes},
		)
		g.index_capacity = g.index_buffer != nil ? index_bytes : 0
		if g.index_buffer == nil {
			return false
		}
	}

	if len(g.ranges) > 0 {
		bytes := (([^]u8)(raw_data(g.ranges)))[:len(g.ranges) * size_of(Cluster_Range)]
		if !upload_bytes(gpu, g.range_buffer, bytes) {
			return false
		}
	}
	if len(g.indices) > 0 {
		bytes := (([^]u8)(raw_data(g.indices)))[:len(g.indices) * size_of(u32)]
		if !upload_bytes(gpu, g.index_buffer, bytes) {
			return false
		}
	}
	return true
}

clusters_destroy :: proc(gpu: ^sdl.GPUDevice, g: ^Cluster_Grid) {
	if gpu != nil {
		if g.range_buffer != nil {
			sdl.ReleaseGPUBuffer(gpu, g.range_buffer)
		}
		if g.index_buffer != nil {
			sdl.ReleaseGPUBuffer(gpu, g.index_buffer)
		}
	}
	delete(g.ranges)
	delete(g.indices)
	g^ = {}
}
