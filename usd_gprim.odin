package main

// Tessellation of USD's parametric gprims (Cube, Sphere, Cylinder, Cone,
// Capsule) into the same polygon soup a UsdGeomMesh hands over, so the rest of
// the USD path -- material binding, triangulation, UV flip, BVH -- treats them
// as ordinary meshes.
//
// The shim reports only the schema's numbers (see usd_shim_get_gprim_data);
// building geometry from them lives here, next to usd_triangulate_mesh, for
// the same reason: the shim stays thin marshalling.
//
// Everything is built in a canonical frame with the spine along +Z, then
// permuted onto the prim's `axis`. Vertices are emitted unwelded (one point
// per face corner) so seams and poles need no special indexing, and every
// corner carries its own analytic normal -- a sphere still shades smooth.

import "core:c"
import "core:math"
import m "core:math/linalg/glsl"

USD_GPRIM_SEGMENTS :: 32       // radial divisions around the spine
USD_GPRIM_SPHERE_STACKS :: 16  // pole-to-pole divisions of a Sphere
USD_GPRIM_CAP_STACKS :: 8      // divisions of one Capsule hemisphere

Usd_Gprim_Vertex :: struct {
	pos: [3]f64,
	nrm: [3]f64,
	uv:  [2]f64,
}

// Unwelded polygon soup, laid out to match Usd_Shim_Mesh_Data's vertex
// interpolation: one point, normal and uv per face corner.
Usd_Gprim_Mesh :: struct {
	points:  [dynamic]f32,
	normals: [dynamic]f32,
	uvs:     [dynamic]f32,
	indices: [dynamic]c.int,
	counts:  [dynamic]c.int,
}

usd_gprim_mesh_destroy :: proc(g: ^Usd_Gprim_Mesh) {
	delete(g.points)
	delete(g.normals)
	delete(g.uvs)
	delete(g.indices)
	delete(g.counts)
}

// Borrows the builder's buffers. Valid only while `g` is alive and unmodified.
usd_gprim_mesh_view :: proc(g: ^Usd_Gprim_Mesh) -> Usd_Shim_Mesh_Data {
	return Usd_Shim_Mesh_Data{
		points              = raw_data(g.points),
		point_count         = c.int(len(g.points) / 3),
		face_vertex_indices = raw_data(g.indices),
		index_count         = c.int(len(g.indices)),
		face_vertex_counts  = raw_data(g.counts),
		face_count          = c.int(len(g.counts)),
		normals             = raw_data(g.normals),
		normal_count        = c.int(len(g.normals) / 3),
		normal_interp       = .Vertex,
		uvs                 = raw_data(g.uvs),
		uv_count            = c.int(len(g.uvs) / 2),
		uv_interp           = .Vertex,
	}
}

usd_emit_gprim :: proc(
	prim: Usd_Shim_Prim,
	gp: Usd_Shim_Gprim_Data,
	transform: m.mat4,
	meshes: ^[dynamic]Mesh,
	state: ^usd_load_state,
) {
	g := usd_tessellate_gprim(gp)
	defer usd_gprim_mesh_destroy(&g)
	if len(g.counts) == 0 {
		return
	}
	// A gprim is one surface with one binding; GeomSubsets are a Mesh thing.
	usd_build_mesh(prim, usd_gprim_mesh_view(&g), transform, meshes, state, use_subsets = false)
}

usd_tessellate_gprim :: proc(gp: Usd_Shim_Gprim_Data) -> Usd_Gprim_Mesh {
	g: Usd_Gprim_Mesh
	switch gp.type {
	case .Cube:
		usd_gprim_cube(&g, gp.size)
	case .Sphere:
		usd_gprim_sphere(&g, gp.radius)
	case .Cylinder, .Cone:
		// One shape: a frustum. A Cone is the radius_top == 0 case, its top
		// ring collapsed onto the apex.
		usd_gprim_frustum(&g, gp.axis, gp.radius_bottom, gp.radius_top, gp.height)
	case .Capsule:
		usd_gprim_capsule(&g, gp.axis, gp.radius, gp.height)
	case .None:
	}
	return g
}

// ---------------------------------------------------------------------------
// Frame handling and primitive emission
// ---------------------------------------------------------------------------

// Canonical (+Z spine) to the prim's axis. Both are cyclic permutations, so
// each is a rotation (determinant +1) -- no handedness flip, and the winding
// established below survives.
usd_gprim_axis_map :: proc(v: [3]f64, axis: Usd_Axis) -> [3]f64 {
	switch axis {
	case .X: return {v.z, v.x, v.y}
	case .Y: return {v.y, v.z, v.x}
	case .Z: return v
	}
	return v
}

usd_gprim_push :: proc(g: ^Usd_Gprim_Mesh, axis: Usd_Axis, v: Usd_Gprim_Vertex) {
	p := usd_gprim_axis_map(v.pos, axis)
	n := usd_gprim_axis_map(v.nrm, axis)
	append(&g.points, f32(p.x), f32(p.y), f32(p.z))
	append(&g.normals, f32(n.x), f32(n.y), f32(n.z))
	append(&g.uvs, f32(v.uv.x), f32(v.uv.y))
	append(&g.indices, c.int(len(g.points) / 3 - 1))
}

usd_gprim_face :: proc(g: ^Usd_Gprim_Mesh, axis: Usd_Axis, verts: ..Usd_Gprim_Vertex) {
	for v in verts {
		usd_gprim_push(g, axis, v)
	}
	append(&g.counts, c.int(len(verts)))
}

// A ring is one closed loop of USD_GPRIM_SEGMENTS+1 vertices; the last repeats
// the first's position with u = 1 instead of 0, which is what keeps the UV
// seam from wrapping.
usd_gprim_ring_make :: proc() -> []Usd_Gprim_Vertex {
	return make([]Usd_Gprim_Vertex, USD_GPRIM_SEGMENTS + 1)
}

// A ring whose vertices all sit on one point: a sphere pole, a cone apex, or
// the centre of a cap disc. Banding against one produces triangles, not quads.
usd_gprim_ring_degenerate :: proc(ring: []Usd_Gprim_Vertex) -> bool {
	if len(ring) == 0 {
		return true
	}
	p0 := ring[0].pos
	// Poles come out of cos(+/-pi/2), so they miss the axis by an epsilon that
	// scales with the radius rather than landing on exactly zero.
	tol := 1e-9 * max(1.0, abs(p0.x) + abs(p0.y) + abs(p0.z))
	for v in ring[1:] {
		d := v.pos - p0
		if abs(d.x) > tol || abs(d.y) > tol || abs(d.z) > tol {
			return false
		}
	}
	return true
}

// Stitches two rings into a strip. Corner order runs +phi along `lo`, then up
// to `hi`, so the face normal is cross(d_phi, d_lo_to_hi) -- outward for every
// caller below. Reverse the two arguments to flip a face outward the other way
// (the two cap discs differ only in that).
usd_gprim_band :: proc(g: ^Usd_Gprim_Mesh, axis: Usd_Axis, lo, hi: []Usd_Gprim_Vertex) {
	lo_deg := usd_gprim_ring_degenerate(lo)
	hi_deg := usd_gprim_ring_degenerate(hi)
	if lo_deg && hi_deg {
		return
	}
	for j in 0 ..< len(lo) - 1 {
		a, b := lo[j], lo[j + 1]
		cc, d := hi[j + 1], hi[j]
		switch {
		// Drop whichever corner is a duplicate of the point the ring
		// collapsed to; the remaining cycle keeps its winding.
		case lo_deg: usd_gprim_face(g, axis, a, cc, d)
		case hi_deg: usd_gprim_face(g, axis, a, b, d)
		case:        usd_gprim_face(g, axis, a, b, cc, d)
		}
	}
}

usd_gprim_bands :: proc(g: ^Usd_Gprim_Mesh, axis: Usd_Axis, rings: [][]Usd_Gprim_Vertex) {
	for i in 0 ..< len(rings) - 1 {
		usd_gprim_band(g, axis, rings[i], rings[i + 1])
	}
}

// ---------------------------------------------------------------------------
// The shapes
// ---------------------------------------------------------------------------

// UsdGeomCube: `size` is the edge length, centred on the origin. It has no
// axis, so it is built directly in world space.
usd_gprim_cube :: proc(g: ^Usd_Gprim_Mesh, size: f64) {
	h := size * 0.5
	if h <= 0 {
		return
	}
	// (normal, u, v) per face, with cross(u, v) == normal, so walking the
	// corners counter-clockwise in (u, v) faces the quad outward.
	faces := [6][3][3]f64{
		{{ 1,  0,  0}, { 0,  0, -1}, {0, 1,  0}},
		{{-1,  0,  0}, { 0,  0,  1}, {0, 1,  0}},
		{{ 0,  1,  0}, { 1,  0,  0}, {0, 0, -1}},
		{{ 0, -1,  0}, { 1,  0,  0}, {0, 0,  1}},
		{{ 0,  0,  1}, { 1,  0,  0}, {0, 1,  0}},
		{{ 0,  0, -1}, {-1,  0,  0}, {0, 1,  0}},
	}
	corners := [4][2]f64{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}

	for face in faces {
		n, u, v := face[0], face[1], face[2]
		quad: [4]Usd_Gprim_Vertex
		for corner, i in corners {
			su, sv := corner[0], corner[1]
			quad[i] = Usd_Gprim_Vertex{
				pos = h * (n + su * u + sv * v),
				nrm = n,
				uv  = {(su + 1) * 0.5, (sv + 1) * 0.5},
			}
		}
		usd_gprim_face(g, .Z, quad[0], quad[1], quad[2], quad[3])
	}
}

// UsdGeomSphere: `radius`, centred on the origin, no axis. Poles are placed on
// Z to match the rest of the gprim family; USD defines no UVs for a sphere, so
// the choice is ours.
usd_gprim_sphere :: proc(g: ^Usd_Gprim_Mesh, radius: f64) {
	if radius <= 0 {
		return
	}
	prev, cur := usd_gprim_ring_make(), usd_gprim_ring_make()
	defer delete(prev)
	defer delete(cur)

	fill :: proc(ring: []Usd_Gprim_Vertex, radius, theta, v: f64) {
		ct, st := math.cos(theta), math.sin(theta)
		for j in 0 ..= USD_GPRIM_SEGMENTS {
			phi := math.TAU * f64(j) / f64(USD_GPRIM_SEGMENTS)
			dir := [3]f64{ct * math.cos(phi), ct * math.sin(phi), st}
			ring[j] = Usd_Gprim_Vertex{
				pos = radius * dir,
				nrm = dir,
				uv  = {f64(j) / f64(USD_GPRIM_SEGMENTS), v},
			}
		}
	}

	for i in 0 ..= USD_GPRIM_SPHERE_STACKS {
		t := f64(i) / f64(USD_GPRIM_SPHERE_STACKS)
		fill(cur, radius, -math.PI * 0.5 + math.PI * t, t)
		if i > 0 {
			usd_gprim_band(g, .Z, prev, cur)
		}
		prev, cur = cur, prev
	}
}

// UsdGeomCylinder / UsdGeomCone / UsdGeomCylinder_1, which are one shape: a
// frustum spanning +/- height/2 along the spine, capped by discs. A Cone is
// radius_top == 0, so its top disc vanishes and its side collapses to an apex.
usd_gprim_frustum :: proc(g: ^Usd_Gprim_Mesh, axis: Usd_Axis, radius_bottom, radius_top, height: f64) {
	if height <= 0 || (radius_bottom <= 0 && radius_top <= 0) {
		return
	}
	z0, z1 := -height * 0.5, height * 0.5

	// Outward normal of the slanted side: the surface radius varies as
	// dr/dz = (rt - rb)/h, so the normal leans by the negative of that.
	slope := (radius_bottom - radius_top) / height
	side_len := math.sqrt(1 + slope * slope)

	side_lo, side_hi := usd_gprim_ring_make(), usd_gprim_ring_make()
	defer delete(side_lo)
	defer delete(side_hi)

	for j in 0 ..= USD_GPRIM_SEGMENTS {
		u := f64(j) / f64(USD_GPRIM_SEGMENTS)
		phi := math.TAU * u
		cp, sp := math.cos(phi), math.sin(phi)
		n := [3]f64{cp / side_len, sp / side_len, slope / side_len}
		side_lo[j] = {pos = {radius_bottom * cp, radius_bottom * sp, z0}, nrm = n, uv = {u, 0}}
		side_hi[j] = {pos = {radius_top * cp, radius_top * sp, z1}, nrm = n, uv = {u, 1}}
	}
	usd_gprim_band(g, axis, side_lo, side_hi)

	// Cap discs, each a fan from a degenerate centre ring. Banding centre->rim
	// faces -Z and rim->centre faces +Z, so the argument order picks the side.
	disc :: proc(ring: []Usd_Gprim_Vertex, radius, z, nz: f64, centre: bool) {
		r := 0.0 if centre else radius
		uv_r := 0.0 if centre else 0.5
		for j in 0 ..= USD_GPRIM_SEGMENTS {
			phi := math.TAU * f64(j) / f64(USD_GPRIM_SEGMENTS)
			cp, sp := math.cos(phi), math.sin(phi)
			ring[j] = Usd_Gprim_Vertex{
				pos = {r * cp, r * sp, z},
				nrm = {0, 0, nz},
				uv  = {0.5 + uv_r * cp, 0.5 + uv_r * sp},
			}
		}
	}

	centre, rim := usd_gprim_ring_make(), usd_gprim_ring_make()
	defer delete(centre)
	defer delete(rim)

	if radius_bottom > 0 {
		disc(centre, radius_bottom, z0, -1, true)
		disc(rim, radius_bottom, z0, -1, false)
		usd_gprim_band(g, axis, centre, rim)
	}
	if radius_top > 0 {
		disc(centre, radius_top, z1, 1, true)
		disc(rim, radius_top, z1, 1, false)
		usd_gprim_band(g, axis, rim, centre)
	}
}

// UsdGeomCapsule: a cylinder of `height` capped by two hemispheres of
// `radius`, so the shape spans height + 2*radius along the spine. The
// hemisphere normal at the equator is already the cylinder's side normal, so
// the three sections band together as one continuous run of rings.
usd_gprim_capsule :: proc(g: ^Usd_Gprim_Mesh, axis: Usd_Axis, radius, height: f64) {
	if radius <= 0 {
		return
	}
	half := height * 0.5
	total := height + 2 * radius

	rings: [dynamic][]Usd_Gprim_Vertex
	defer {
		for r in rings {
			delete(r)
		}
		delete(rings)
	}

	// One ring on the sphere of `centre`, at latitude `theta`. v runs over the
	// capsule's full length so the caps and the barrel share one map.
	hemi_ring :: proc(radius, centre_z, theta, half, total: f64) -> []Usd_Gprim_Vertex {
		ring := usd_gprim_ring_make()
		ct, st := math.cos(theta), math.sin(theta)
		z := centre_z + radius * st
		for j in 0 ..= USD_GPRIM_SEGMENTS {
			u := f64(j) / f64(USD_GPRIM_SEGMENTS)
			phi := math.TAU * u
			dir := [3]f64{ct * math.cos(phi), ct * math.sin(phi), st}
			ring[j] = Usd_Gprim_Vertex{
				pos = {radius * dir.x, radius * dir.y, z},
				nrm = dir,
				uv  = {u, (z + half + radius) / total},
			}
		}
		return ring
	}

	// Bottom hemisphere, south pole up to the equator at z = -height/2.
	for i in 0 ..= USD_GPRIM_CAP_STACKS {
		theta := -math.PI * 0.5 * (1 - f64(i) / f64(USD_GPRIM_CAP_STACKS))
		append(&rings, hemi_ring(radius, -half, theta, half, total))
	}
	// Top hemisphere, equator at z = +height/2 up to the north pole. Its first
	// ring is the barrel's top, so the cylindrical section needs no ring of
	// its own -- the band between the two equators is the barrel.
	for i in 0 ..= USD_GPRIM_CAP_STACKS {
		theta := math.PI * 0.5 * f64(i) / f64(USD_GPRIM_CAP_STACKS)
		append(&rings, hemi_ring(radius, half, theta, half, total))
	}

	usd_gprim_bands(g, axis, rings[:])
}
