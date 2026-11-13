package lumbre_core

// Irradiance probes for the realtime renderer's diffuse ambient term, baked by
// the CPU path tracer (Phase 04 step 12).
//
// An axis-aligned grid of probes over the scene's bounds. Each probe traces a
// fixed set of directions with `ray_color` and keeps two octahedral maps:
//
//   irradiance  per texel direction n, the incoming radiance convolved with
//               the clamped cosine about n, divided by pi: the cosine-weighted
//               mean radiance. The same quantity the realtime renderer's
//               environment irradiance map holds, so it stands in for it
//               directly.
//   depth       the mean and mean square of the distance to the first hit.
//               The realtime lookup uses them for a Chebyshev visibility test
//               (as in DDGI), so a probe on the far side of a wall does not
//               light the near side.
//   radiance    the incoming radiance itself, at 16x16 under a narrow lobe:
//               what a reflection sees along a direction. The realtime
//               renderer reflects it where screen-space reflections find
//               nothing on screen, instead of the sky.
//
// Not spherical harmonics, though the phase plan proposed L2 SH. L2 rings: the
// fit to a small bright source -- the Cornell box's ceiling panel, seen by the
// probes beside it -- goes negative about 120 degrees away from it, which put
// dark, green-tinted blots on the ceiling around the light. The octahedral map
// is convolved from the rays themselves, so it is never negative, and shares
// the depth map's layout and lookup.
//
// What a probe sees is radiance leaving surfaces and the environment, never an
// analytic light's own emission: the path tracer's rays do not intersect lights.
// That is exactly what the realtime renderer lacks -- it shades analytic lights
// directly, so probes add the bounce and emissive-mesh light without counting
// any light twice.
//
// Deterministic by construction: the directions are a fixed Fibonacci set, each
// probe draws from its own seeded generator, and threads only decide which
// probe is traced when. Two bakes of the same scene are byte-identical.

import "core:encoding/endian"
import "core:hash/xxhash"
import "core:math"
import m "core:math/linalg/glsl"
import "core:os"
import "core:slice"
import "core:strings"
import "core:sync"
import "core:sys/info"
import "core:thread"

PROBE_FILE_MAGIC :: "LUMBREPROBES"
PROBE_FILE_VERSION :: u32(3)
// Interior texels per side of a probe's octahedral maps, and the tile with its
// one-texel border, which lets a bilinear lookup cross an edge of the
// octahedron without special cases. Irradiance is smooth enough for 8x8; the
// depth maps share it so one lookup serves both.
PROBE_TILE_RES :: 8
PROBE_TILE :: PROBE_TILE_RES + 2
PROBE_TILE_TEXELS :: PROBE_TILE * PROBE_TILE
// The radiance map is sharper: reflections need more than 25-degree texels.
// Its lobe, cos^64, is about the texel's own width at 16x16.
PROBE_RADIANCE_RES :: 16
PROBE_RADIANCE_TILE :: PROBE_RADIANCE_RES + 2
PROBE_RADIANCE_TEXELS :: PROBE_RADIANCE_TILE * PROBE_RADIANCE_TILE
PROBE_RADIANCE_SHARPNESS :: 64.0
// Exponent of the cosine lobe that spreads each ray's distance over nearby
// texels. DDGI's default; at 8x8 a lobe this narrow still covers ~14 rays per
// texel at 2048 rays.
PROBE_DEPTH_SHARPNESS :: 50.0
// A flat scene still gets a grid with some depth: the flattest axis counts as
// at least this fraction of the largest. Matches the fog's rule.
PROBE_MIN_EXTENT_FRACTION :: 0.1

Probe_Grid :: struct {
	// `probe_scene_revision` of the scene this was baked from.
	revision:     u64,
	dims:         [3]i32,
	bounds_min:   [3]f32,
	bounds_max:   [3]f32,
	// Distances in `depth` are divided by this, and clamped to it.
	max_distance: f32,
	rays:         i32,
	// probe_count * PROBE_TILE_TEXELS texels each, probes x fastest, then y,
	// then z; texels row by row within a bordered tile.
	irradiance:   [][3]f32,
	// (mean, mean square) of distance / max_distance.
	depth:        [][2]f32,
	// probe_count * PROBE_RADIANCE_TEXELS, laid out the same way.
	radiance:     [][3]f32,
}

Probe_Bake_Settings :: struct {
	grid:         i32, // probes along the scene's longest axis
	rays:         i32, // directions per probe
	max_depth:    i32,
	// The path tracer's own biases, passed through so a bake sees what a
	// render of the same settings would.
	max_radiance:     f64,
	roughness_cutoff: f64,
	glossy_bias:      f64,
}

PROBE_DEFAULT_GRID :: 8
PROBE_DEFAULT_RAYS :: 2048

probe_count :: proc(g: Probe_Grid) -> int {
	return int(g.dims.x) * int(g.dims.y) * int(g.dims.z)
}

probe_grid_destroy :: proc(g: ^Probe_Grid) {
	delete(g.irradiance)
	delete(g.depth)
	delete(g.radiance)
	g^ = {}
}

// Probes sit at cell centres, never on the bounds: a box scene's walls are its
// bounds, and a probe exactly on a wall sees half of everything as a hit at
// distance zero.
probe_position :: proc(g: Probe_Grid, i, j, k: i32) -> [3]f32 {
	idx := [3]i32{i, j, k}
	p: [3]f32
	for a in 0 ..< 3 {
		cell := (g.bounds_max[a] - g.bounds_min[a]) / f32(g.dims[a])
		p[a] = g.bounds_min[a] + (f32(idx[a]) + 0.5) * cell
	}
	return p
}

// Grid dimensions: `n` along the longest axis, the others in proportion, at
// least one each.
probe_grid_dims :: proc(extent: [3]f32, n: i32) -> [3]i32 {
	longest := max(extent.x, extent.y, extent.z)
	dims: [3]i32
	for a in 0 ..< 3 {
		dims[a] = longest > 0 ? max(1, i32(math.ceil(f32(n) * extent[a] / longest - 1e-4))) : 1
	}
	return dims
}

// The world bounds of every triangle and sphere, with a flat axis widened to
// PROBE_MIN_EXTENT_FRACTION of the largest about its centre.
probe_scene_bounds :: proc(triangles: []Triangle, spheres: []Sphere) -> (lo, hi: [3]f32, ok: bool) {
	lo = {max(f32), max(f32), max(f32)}
	hi = {-max(f32), -max(f32), -max(f32)}
	grow :: proc(lo, hi: ^[3]f32, p: Vec3) {
		for a in 0 ..< 3 {
			lo[a] = min(lo[a], f32(p[a]))
			hi[a] = max(hi[a], f32(p[a]))
		}
	}
	for t in triangles {
		grow(&lo, &hi, t.v0); grow(&lo, &hi, t.v1); grow(&lo, &hi, t.v2)
		ok = true
	}
	for s in spheres {
		r := Vec3{s.radius, s.radius, s.radius}
		grow(&lo, &hi, s.center - r); grow(&lo, &hi, s.center + r)
		ok = true
	}
	if !ok { return {}, {}, false }
	extent := hi - lo
	floor_extent := max(extent.x, extent.y, extent.z) * PROBE_MIN_EXTENT_FRACTION
	if floor_extent <= 0 { floor_extent = 1 }
	for a in 0 ..< 3 {
		if extent[a] < floor_extent {
			c := (lo[a] + hi[a]) * 0.5
			lo[a] = c - floor_extent * 0.5
			hi[a] = c + floor_extent * 0.5
		}
	}
	return lo, hi, true
}

// ── octahedral mapping ───────────────────────────────────────────────────────

@(private = "file")
sign_not_zero :: proc(v: f64) -> f64 { return v >= 0 ? 1 : -1 }

// Unit direction to [0,1]^2. Must match `probe_oct_encode` in shading.slang.
probe_oct_encode :: proc(d: [3]f64) -> [2]f64 {
	l1 := abs(d.x) + abs(d.y) + abs(d.z)
	p := [2]f64{d.x / l1, d.y / l1}
	if d.z < 0 {
		p = {(1 - abs(p.y)) * sign_not_zero(p.x), (1 - abs(p.x)) * sign_not_zero(p.y)}
	}
	return p * 0.5 + 0.5
}

probe_oct_decode :: proc(uv: [2]f64) -> [3]f64 {
	p := uv * 2 - 1
	d := [3]f64{p.x, p.y, 1 - abs(p.x) - abs(p.y)}
	if d.z < 0 {
		d.x, d.y = (1 - abs(p.y)) * sign_not_zero(p.x), (1 - abs(p.x)) * sign_not_zero(p.y)
	}
	return m.normalize(d)
}

// Where border texel (x, y) of a tile copies from, so bilinear filtering
// across the octahedron's edges reads the texel on the other side of the fold.
// `R` is the tile's interior size.
@(private = "file")
probe_border_source :: proc(x, y: int, R := PROBE_TILE_RES) -> (int, int) {
	switch {
	case x == 0 && y == 0:         return R, R
	case x == R + 1 && y == 0:     return 1, R
	case x == 0 && y == R + 1:     return R, 1
	case x == R + 1 && y == R + 1: return 1, 1
	case y == 0:                   return R + 1 - x, 1
	case y == R + 1:               return R + 1 - x, R
	case x == 0:                   return 1, R + 1 - y
	case x == R + 1:               return R, R + 1 - y
	}
	return x, y
}

// N points spread evenly over the sphere. Fixed, so a bake is reproducible and
// every probe samples the same directions.
probe_ray_direction :: proc(i, n: i32) -> [3]f64 {
	GOLDEN_ANGLE :: 2.399963229728653
	z := 1 - (2 * f64(i) + 1) / f64(n)
	r := math.sqrt(max(0, 1 - z * z))
	phi := GOLDEN_ANGLE * f64(i)
	return {r * math.cos(phi), r * math.sin(phi), z}
}

// ── bake ─────────────────────────────────────────────────────────────────────

@(private = "file")
Bake_Shared :: struct {
	grid:            ^Probe_Grid,
	settings:        Probe_Bake_Settings,
	spheres:         []Sphere,
	sphere_nodes:    []BVH_Node,
	sphere_bvh_root: i32,
	triangles:       []Triangle,
	tri_nodes:       []BVH_Node,
	tri_bvh_root:    i32,
	materials:       []Material,
	lights:          []Light,
	env:             ^Environment,
	texel_dirs:      [][3]f64,
	radiance_dirs:   [][3]f64,
	dirs:            [][3]f64,
	next:            int, // next probe to trace, taken atomically
	done:            int,
	progress:        proc(done, total: int),
}

// Fills one probe's two bordered tiles from its rays: the direction of each,
// the radiance it brought back, and its first-hit distance / max distance.
probe_fill_tiles :: proc(
	dirs: [][3]f64, radiance: [][3]f64, distance: []f64, texel_dirs: [][3]f64,
	irradiance_tile: [][3]f32, depth_tile: [][2]f32,
) {
	n := len(dirs)
	for n_t, t in texel_dirs {
		// Uniform directions: each ray stands for 4pi / n of solid angle, so
		// the cosine-weighted integral over pi is 4 / n times the sum.
		irr: [3]f64
		moments: [2]f64
		weight: f64
		for d, i in dirs {
			c := m.dot(n_t, d)
			if c <= 0 { continue }
			irr += radiance[i] * c
			// cos^50 is below 1e-6 under 0.76; skip the pow there.
			if c < 0.76 { continue }
			w := math.pow(c, PROBE_DEPTH_SHARPNESS)
			moments += w * [2]f64{distance[i], distance[i] * distance[i]}
			weight += w
		}
		irr *= 4 / f64(n)
		// A texel no ray came near (only with very few rays) reads as open space.
		mo := weight > 0 ? moments / weight : [2]f64{1, 1}
		x, y := t % PROBE_TILE_RES + 1, t / PROBE_TILE_RES + 1
		irradiance_tile[y * PROBE_TILE + x] = {f32(irr.x), f32(irr.y), f32(irr.z)}
		depth_tile[y * PROBE_TILE + x] = {f32(mo.x), f32(mo.y)}
	}
	for y in 0 ..< PROBE_TILE {
		for x in 0 ..< PROBE_TILE {
			if x > 0 && x <= PROBE_TILE_RES && y > 0 && y <= PROBE_TILE_RES { continue }
			sx, sy := probe_border_source(x, y)
			irradiance_tile[y * PROBE_TILE + x] = irradiance_tile[sy * PROBE_TILE + sx]
			depth_tile[y * PROBE_TILE + x] = depth_tile[sy * PROBE_TILE + sx]
		}
	}
}

// Fills one probe's bordered radiance tile: per texel, the rays' radiance
// averaged under a cos^64 lobe about the texel's direction.
probe_fill_radiance :: proc(
	dirs: [][3]f64, radiance: [][3]f64, texel_dirs: [][3]f64, tile: [][3]f32,
) {
	// What a texel no ray came near falls back to: the mean over the sphere.
	// Only a bake with very few rays leaves one empty.
	mean: [3]f64
	for L in radiance { mean += L }
	mean /= f64(max(len(radiance), 1))
	for n_t, t in texel_dirs {
		sum: [3]f64
		weight: f64
		for d, i in dirs {
			c := m.dot(n_t, d)
			// cos^64 is below 3e-4 under 0.88.
			if c < 0.88 { continue }
			w := math.pow(c, PROBE_RADIANCE_SHARPNESS)
			sum += w * radiance[i]
			weight += w
		}
		v := weight > 0 ? sum / weight : mean
		x, y := t % PROBE_RADIANCE_RES + 1, t / PROBE_RADIANCE_RES + 1
		tile[y * PROBE_RADIANCE_TILE + x] = {f32(v.x), f32(v.y), f32(v.z)}
	}
	for y in 0 ..< PROBE_RADIANCE_TILE {
		for x in 0 ..< PROBE_RADIANCE_TILE {
			if x > 0 && x <= PROBE_RADIANCE_RES && y > 0 && y <= PROBE_RADIANCE_RES { continue }
			sx, sy := probe_border_source(x, y, PROBE_RADIANCE_RES)
			tile[y * PROBE_RADIANCE_TILE + x] = tile[sy * PROBE_RADIANCE_TILE + sx]
		}
	}
}

// Interior texel t's direction in the radiance tile, row by row.
probe_radiance_texel_direction :: proc(t: int) -> [3]f64 {
	R :: f64(PROBE_RADIANCE_RES)
	uv := [2]f64{(f64(t % PROBE_RADIANCE_RES) + 0.5) / R, (f64(t / PROBE_RADIANCE_RES) + 0.5) / R}
	return probe_oct_decode(uv)
}

// Interior texel t's direction, row by row.
probe_texel_direction :: proc(t: int) -> [3]f64 {
	uv := [2]f64{(f64(t % PROBE_TILE_RES) + 0.5) / PROBE_TILE_RES, (f64(t / PROBE_TILE_RES) + 0.5) / PROBE_TILE_RES}
	return probe_oct_decode(uv)
}

@(private = "file")
bake_probe :: proc(s: ^Bake_Shared, index: int) {
	g := s.grid
	nx, ny := int(g.dims.x), int(g.dims.y)
	i, j, k := i32(index % nx), i32((index / nx) % ny), i32(index / (nx * ny))
	p := probe_position(g^, i, j, k)
	origin := Point3{f64(p.x), f64(p.y), f64(p.z)}

	rng := Rng{state = u64(index + 1) * 0x9e3779b97f4a7c15}
	rays := int(s.settings.rays)
	max_dist := f64(g.max_distance)

	radiance := make([][3]f64, rays)
	defer delete(radiance)
	distance := make([]f64, rays)
	defer delete(distance)

	for r in 0 ..< rays {
		ray := Ray{origin, Vec3(s.dirs[r])}
		// The first hit's distance for the visibility map. `ray_color` traces
		// the same ray again; the duplicate intersection is a small cost next
		// to the path behind it.
		rec, hit := hit_scene(s.spheres, s.sphere_nodes, s.sphere_bvh_root, s.triangles, s.tri_nodes, s.tri_bvh_root, s.materials, ray)
		distance[r] = (hit ? min(rec.t, max_dist) : max_dist) / max_dist
		L := ray_color(s.spheres, s.sphere_nodes, s.sphere_bvh_root, s.triangles, s.tri_nodes, s.tri_bvh_root,
			s.materials, s.lights, s.env, ray, s.settings.max_depth, s.settings.max_radiance, &rng,
			s.settings.roughness_cutoff, s.settings.glossy_bias)
		radiance[r] = {L.x, L.y, L.z}
	}

	probe_fill_tiles(s.dirs, radiance, distance, s.texel_dirs,
		g.irradiance[index * PROBE_TILE_TEXELS:][:PROBE_TILE_TEXELS],
		g.depth[index * PROBE_TILE_TEXELS:][:PROBE_TILE_TEXELS])
	probe_fill_radiance(s.dirs, radiance, s.radiance_dirs,
		g.radiance[index * PROBE_RADIANCE_TEXELS:][:PROBE_RADIANCE_TEXELS])
}

@(private = "file")
bake_worker :: proc(data: rawptr) {
	s := cast(^Bake_Shared)data
	total := probe_count(s.grid^)
	for {
		index := sync.atomic_add(&s.next, 1)
		if index >= total { return }
		bake_probe(s, index)
		done := sync.atomic_add(&s.done, 1) + 1
		if s.progress != nil { s.progress(done, total) }
	}
}

// Bakes a probe grid for `scene`. `progress`, when given, is called from the
// worker threads after each probe.
probe_bake :: proc(
	scene: ^Scene,
	settings: Probe_Bake_Settings,
	progress: proc(done, total: int) = nil,
) -> (grid: Probe_Grid, ok: bool) {
	flattened := flatten_scene_graph(scene)
	defer destroy_flattened_scene(flattened)

	lo, hi := probe_scene_bounds(flattened.triangles, scene.spheres) or_return
	settings := settings
	settings.grid = max(settings.grid, 1)
	settings.rays = max(settings.rays, 16)

	grid.revision = probe_scene_revision(scene)
	grid.bounds_min, grid.bounds_max = lo, hi
	grid.dims = probe_grid_dims(hi - lo, settings.grid)
	grid.rays = settings.rays
	cell := (hi - lo) / [3]f32{f32(grid.dims.x), f32(grid.dims.y), f32(grid.dims.z)}
	// 1.5 cell diagonals: a shaded point is never farther than one diagonal
	// from any of its eight probes, and the margin keeps a hit just past it
	// distinguishable from open space.
	grid.max_distance = 1.5 * math.sqrt(cell.x * cell.x + cell.y * cell.y + cell.z * cell.z)

	n := probe_count(grid)
	grid.irradiance = make([][3]f32, n * PROBE_TILE_TEXELS)
	grid.depth = make([][2]f32, n * PROBE_TILE_TEXELS)
	grid.radiance = make([][3]f32, n * PROBE_RADIANCE_TEXELS)

	sphere_nodes := make([]BVH_Node, bvh_node_capacity(len(scene.spheres)))
	defer delete(sphere_nodes)
	sphere_count: i32
	sphere_root := i32(-1)
	if len(scene.spheres) > 0 {
		sphere_root = build_bvh(scene.spheres, sphere_nodes, &sphere_count, 0, i32(len(scene.spheres)))
	}
	tri_nodes := make([]BVH_Node, bvh_node_capacity(len(flattened.triangles)))
	defer delete(tri_nodes)
	tri_count: i32
	tri_root := i32(-1)
	if len(flattened.triangles) > 0 {
		tri_root = build_triangle_bvh(flattened.triangles, tri_nodes, &tri_count, 0, i32(len(flattened.triangles)))
	}

	texel_dirs := make([][3]f64, PROBE_TILE_RES * PROBE_TILE_RES)
	defer delete(texel_dirs)
	for &d, t in texel_dirs { d = probe_texel_direction(t) }
	radiance_dirs := make([][3]f64, PROBE_RADIANCE_RES * PROBE_RADIANCE_RES)
	defer delete(radiance_dirs)
	for &d, t in radiance_dirs { d = probe_radiance_texel_direction(t) }
	dirs := make([][3]f64, settings.rays)
	defer delete(dirs)
	for &d, r in dirs { d = probe_ray_direction(i32(r), settings.rays) }

	shared := Bake_Shared {
		grid            = &grid,
		settings        = settings,
		spheres         = scene.spheres,
		sphere_nodes    = sphere_nodes[:sphere_count],
		sphere_bvh_root = sphere_root,
		triangles       = flattened.triangles,
		tri_nodes       = tri_nodes[:tri_count],
		tri_bvh_root    = tri_root,
		materials       = flattened.materials,
		lights          = flattened.lights,
		env             = &scene.environment,
		texel_dirs      = texel_dirs,
		radiance_dirs   = radiance_dirs,
		dirs            = dirs,
		progress        = progress,
	}

	_, logical, cpu_ok := info.cpu_core_count()
	workers := clamp(logical if cpu_ok else 4, 1, 64)
	threads := make([]^thread.Thread, workers)
	defer delete(threads)
	for &t in threads {
		t = thread.create_and_start_with_data(&shared, bake_worker)
	}
	for t in threads {
		thread.join(t)
		thread.destroy(t)
	}
	return grid, true
}

// ── revision ─────────────────────────────────────────────────────────────────

@(private = "file")
hash_value :: proc(h: u64, value: $T) -> u64 {
	v := value
	return xxhash.XXH64(([^]u8)(rawptr(&v))[:size_of(T)], h)
}

@(private = "file")
hash_texture :: proc(h: u64, tex: TextureMap) -> u64 {
	if !tex.has_data { return hash_value(h, 0) }
	h := hash_value(h, [3]i32{tex.width, tex.height, tex.srgb ? 1 : 0})
	return xxhash.XXH64(tex.pixels, h)
}

// A material's parameters, not its textures: cheap enough to hash on every
// edit. `probe_heavy_revision` covers the texels.
@(private = "file")
hash_material_params :: proc(h: u64, mat: Material) -> u64 {
	h := hash_value(h, i32(mat.kind))
	for v in ([?]f64{
		mat.fuzz, mat.ir, mat.emission_strength, mat.roughness, mat.metallic,
		mat.specular, mat.clearcoat, mat.clearcoat_roughness, mat.sheen,
		mat.anisotropic, mat.spec_trans, mat.subsurface, mat.subsurface_scale,
		mat.normal_scale,
	}) {
		h = hash_value(h, v)
	}
	for c in ([?]Color{mat.albedo, mat.emission, mat.specular_tint, mat.sheen_tint, mat.subsurface_color, mat.subsurface_radius}) {
		h = hash_value(h, c)
	}
	return h
}

@(private = "file")
hash_material_textures :: proc(h: u64, mat: Material) -> u64 {
	h := h
	for tex in ([?]TextureMap{mat.albedo_tex, mat.metallic_roughness_tex, mat.normal_tex, mat.emissive_tex}) {
		h = hash_texture(h, tex)
	}
	return h
}

// Everything a bake depends on: geometry and its placement, materials, lights
// and the environment. Not the camera. Field by field rather than whole
// structs, so padding bytes and texture pointers never reach the hash.
//
// In two halves so an interactive frontend can recheck cheaply: the heavy one
// (triangles, texels, environment pixels) changes with the scene key, the
// light one (material parameters, lights, environment rotation and intensity)
// with every slider drag.
probe_scene_revision :: proc(scene: ^Scene) -> u64 {
	return probe_revision_combine(probe_heavy_revision(scene), probe_light_revision(scene))
}

probe_revision_combine :: proc(heavy, light: u64) -> u64 {
	return hash_value(heavy, light)
}

probe_heavy_revision :: proc(scene: ^Scene) -> u64 {
	h := hash_value(0, PROBE_FILE_VERSION)
	h = hash_value(h, len(scene.nodes))
	for node in scene.nodes {
		h = hash_value(h, node.local_transform)
		h = hash_value(h, [3]i32{node.mesh_idx, node.material_override_idx, node.parent})
	}
	h = hash_value(h, len(scene.meshes))
	for mesh in scene.meshes {
		h = hash_value(h, len(mesh.triangles))
		// Triangle is nine dvec3s, a b32 and an i32: no padding, so its bytes
		// can be hashed whole.
		#assert(size_of(Triangle) == 9 * size_of(Vec3) + 8)
		h = xxhash.XXH64(slice.to_bytes(mesh.triangles), h)
		h = hash_material_textures(h, mesh.material)
	}
	h = hash_value(h, len(scene.materials))
	for mat in scene.materials { h = hash_material_textures(h, mat) }
	h = hash_value(h, len(scene.spheres))
	for s in scene.spheres {
		h = hash_value(h, s.center)
		h = hash_value(h, s.radius)
		h = hash_material_textures(h, s.material)
	}
	env := scene.environment
	h = hash_value(h, env.has_data ? 1 : 0)
	if env.has_data {
		h = hash_value(h, [2]i32{env.width, env.height})
		h = xxhash.XXH64(slice.to_bytes(env.pixels), h)
	}
	return h
}

probe_light_revision :: proc(scene: ^Scene) -> u64 {
	h := hash_value(0, len(scene.meshes))
	for mesh in scene.meshes { h = hash_material_params(h, mesh.material) }
	h = hash_value(h, len(scene.materials))
	for mat in scene.materials { h = hash_material_params(h, mat) }
	h = hash_value(h, len(scene.spheres))
	for s in scene.spheres { h = hash_material_params(h, s.material) }
	h = hash_value(h, len(scene.lights))
	for l in scene.lights {
		h = hash_value(h, i32(l.kind))
		h = hash_value(h, [5]Vec3{l.position, l.u, l.v, l.intensity, l.direction})
		h = hash_value(h, [6]f64{l.area, l.radius, l.cos_inner, l.cos_outer, l.angular_radius, l.height})
		h = hash_value(h, l.two_sided ? 1 : 0)
	}
	env := scene.environment
	if env.has_data {
		h = hash_value(h, [2]f64{env.rotation, env.intensity})
	}
	return h
}

// ── file ─────────────────────────────────────────────────────────────────────
//
// Little-endian throughout:
//   "LUMBREPROBES"  u32 version  u64 revision  i32 dims[3]
//   f32 bounds_min[3]  f32 bounds_max[3]  f32 max_distance  i32 rays
//   i32 tile (texels per side, with the border)  i32 radiance_tile
//   f32 irradiance[probe_count][tile^2][3]
//   f32 depth[probe_count][tile^2][2]
//   f32 radiance[probe_count][radiance_tile^2][3]

// `scene.usda` -> `scene.probes`, next to it.
probe_sidecar_path :: proc(scene_file: string, allocator := context.allocator) -> string {
	dot := strings.last_index_byte(scene_file, '.')
	slash := max(strings.last_index_byte(scene_file, '/'), strings.last_index_byte(scene_file, '\\'))
	stem := scene_file[:dot] if dot > slash + 1 else scene_file
	return strings.concatenate({stem, ".probes"}, allocator)
}

probe_grid_encode :: proc(g: Probe_Grid, allocator := context.allocator) -> []u8 {
	buf := make([dynamic]u8, allocator)
	put_u32 :: proc(b: ^[dynamic]u8, v: u32) {
		bytes: [4]u8
		endian.put_u32(bytes[:], .Little, v)
		append(b, ..bytes[:])
	}
	put_f32 :: proc(b: ^[dynamic]u8, v: f32) { put_u32(b, transmute(u32)v) }
	put_i32 :: proc(b: ^[dynamic]u8, v: i32) { put_u32(b, u32(v)) }

	append(&buf, PROBE_FILE_MAGIC)
	put_u32(&buf, PROBE_FILE_VERSION)
	put_u32(&buf, u32(g.revision))
	put_u32(&buf, u32(g.revision >> 32))
	for a in 0 ..< 3 { put_i32(&buf, g.dims[a]) }
	for a in 0 ..< 3 { put_f32(&buf, g.bounds_min[a]) }
	for a in 0 ..< 3 { put_f32(&buf, g.bounds_max[a]) }
	put_f32(&buf, g.max_distance)
	put_i32(&buf, g.rays)
	put_i32(&buf, PROBE_TILE)
	put_i32(&buf, PROBE_RADIANCE_TILE)
	for t in g.irradiance { for v in t { put_f32(&buf, v) } }
	for t in g.depth { put_f32(&buf, t.x); put_f32(&buf, t.y) }
	for t in g.radiance { for v in t { put_f32(&buf, v) } }
	return buf[:]
}

probe_grid_decode :: proc(data: []u8, allocator := context.allocator) -> (g: Probe_Grid, ok: bool) {
	pos := 0
	get_u32 :: proc(data: []u8, pos: ^int) -> (u32, bool) {
		if pos^ + 4 > len(data) { return 0, false }
		v, _ := endian.get_u32(data[pos^:][:4], .Little)
		pos^ += 4
		return v, true
	}
	get_f32 :: proc(data: []u8, pos: ^int) -> (f32, bool) {
		v, ok := get_u32(data, pos)
		return transmute(f32)v, ok
	}

	if len(data) < len(PROBE_FILE_MAGIC) || string(data[:len(PROBE_FILE_MAGIC)]) != PROBE_FILE_MAGIC { return }
	pos = len(PROBE_FILE_MAGIC)
	version := get_u32(data, &pos) or_return
	if version != PROBE_FILE_VERSION { return }
	lo := get_u32(data, &pos) or_return
	hi := get_u32(data, &pos) or_return
	g.revision = u64(lo) | u64(hi) << 32
	for a in 0 ..< 3 { g.dims[a] = i32(get_u32(data, &pos) or_return) }
	for a in 0 ..< 3 { g.bounds_min[a] = get_f32(data, &pos) or_return }
	for a in 0 ..< 3 { g.bounds_max[a] = get_f32(data, &pos) or_return }
	g.max_distance = get_f32(data, &pos) or_return
	g.rays = i32(get_u32(data, &pos) or_return)
	if (get_u32(data, &pos) or_return) != PROBE_TILE { return }
	if (get_u32(data, &pos) or_return) != PROBE_RADIANCE_TILE { return }
	for a in 0 ..< 3 { if g.dims[a] <= 0 || g.dims[a] > 4096 { return } }

	n := probe_count(g)
	texels := n * PROBE_TILE_TEXELS
	radiance_texels := n * PROBE_RADIANCE_TEXELS
	if len(data) - pos != (texels * 5 + radiance_texels * 3) * 4 { return }
	g.irradiance = make([][3]f32, texels, allocator)
	g.depth = make([][2]f32, texels, allocator)
	g.radiance = make([][3]f32, radiance_texels, allocator)
	for &t in g.irradiance { for &v in t { v, _ = get_f32(data, &pos) } }
	for &t in g.depth { t.x, _ = get_f32(data, &pos); t.y, _ = get_f32(data, &pos) }
	for &t in g.radiance { for &v in t { v, _ = get_f32(data, &pos) } }
	return g, true
}

probe_sidecar_exists :: proc(path: string) -> bool {
	return os.exists(path)
}

probe_grid_write :: proc(path: string, g: Probe_Grid) -> bool {
	data := probe_grid_encode(g)
	defer delete(data)
	return os.write_entire_file(path, data) == nil
}

probe_grid_read :: proc(path: string, allocator := context.allocator) -> (Probe_Grid, bool) {
	data, err := os.read_entire_file(path, context.allocator)
	if err != nil { return {}, false }
	defer delete(data)
	return probe_grid_decode(data, allocator)
}
