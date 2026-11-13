package lumbre_core

import "core:bytes"
import "core:math"
import m "core:math/linalg/glsl"
import "core:testing"

// Fills a tile from `radiance` over the bake's own direction set, exactly as
// `bake_probe` does, so these tests check the convolution, not a tracer.
@(private = "file")
convolve :: proc(radiance: proc(d: [3]f64) -> f64, rays: i32) -> (irr: [PROBE_TILE_TEXELS][3]f32) {
	dirs := make([][3]f64, rays)
	defer delete(dirs)
	L := make([][3]f64, rays)
	defer delete(L)
	dist := make([]f64, rays)
	defer delete(dist)
	for &d, r in dirs {
		d = probe_ray_direction(i32(r), rays)
		v := radiance(d)
		L[r] = {v, v, v}
		dist[r] = 1
	}
	texel_dirs: [PROBE_TILE_RES * PROBE_TILE_RES][3]f64
	for &d, t in texel_dirs { d = probe_texel_direction(t) }
	depth: [PROBE_TILE_TEXELS][2]f32
	probe_fill_tiles(dirs, L, dist, texel_dirs[:], irr[:], depth[:])
	return
}

// The interior texel whose direction is nearest `d`, and that direction.
@(private = "file")
nearest_texel :: proc(d: [3]f64) -> (index: int, dir: [3]f64) {
	uv := probe_oct_encode(d)
	x := clamp(int(uv.x * PROBE_TILE_RES), 0, PROBE_TILE_RES - 1)
	y := clamp(int(uv.y * PROBE_TILE_RES), 0, PROBE_TILE_RES - 1)
	return (y + 1) * PROBE_TILE + x + 1, probe_texel_direction(y * PROBE_TILE_RES + x)
}

// A uniform environment of radiance L gives L back in every texel, borders
// included: the cosine-weighted mean of a constant is the constant.
@(test)
test_probe_irradiance_constant :: proc(t: ^testing.T) {
	irr := convolve(proc(d: [3]f64) -> f64 { return 0.7 }, 2048)
	for v, i in irr {
		testing.expectf(t, abs(v.x - 0.7) < 2e-3, "texel %d: %v", i, v.x)
	}
}

// L(w) = 1 + w.z: its cosine-weighted mean about n is 1 + (2/3) n.z. And a
// bright narrow source never drives any texel negative, where L2 SH would.
@(test)
test_probe_irradiance_linear_and_no_ringing :: proc(t: ^testing.T) {
	irr := convolve(proc(d: [3]f64) -> f64 { return 1 + d.z }, 2048)
	for i in 0 ..< PROBE_TILE_RES * PROBE_TILE_RES {
		n := probe_texel_direction(i)
		x, y := i % PROBE_TILE_RES + 1, i / PROBE_TILE_RES + 1
		got := f64(irr[y * PROBE_TILE + x].x)
		want := 1 + 2.0 / 3.0 * n.z
		testing.expectf(t, abs(got - want) < 5e-3, "along %v: got %v, want %v", n, got, want)
	}
	spot := convolve(proc(d: [3]f64) -> f64 { return d.y > 0.98 ? 500 : 0 }, 2048)
	for v, i in spot {
		testing.expectf(t, v.x >= 0, "texel %d negative: %v", i, v.x)
	}
	// Straight away from the source it is exactly zero; L2 SH gives about
	// -0.16 of the source's mean there.
	away, _ := nearest_texel({0, -1, 0})
	testing.expectf(t, spot[away].x == 0, "facing away from the source: %v", spot[away].x)
}

// The radiance tile: a constant field reads back constant everywhere, borders
// included, and a linear one L = 1 + w.z reads back close to itself (the lobe
// averages a little of the slope away: about 1.5% of w.z at cos^64).
@(test)
test_probe_radiance_tile :: proc(t: ^testing.T) {
	RAYS :: 4096
	dirs := make([][3]f64, RAYS)
	defer delete(dirs)
	L := make([][3]f64, RAYS)
	defer delete(L)
	texel_dirs := make([][3]f64, PROBE_RADIANCE_RES * PROBE_RADIANCE_RES)
	defer delete(texel_dirs)
	for &d, i in texel_dirs { d = probe_radiance_texel_direction(i) }
	tile: [PROBE_RADIANCE_TEXELS][3]f32

	for &d, r in dirs { d = probe_ray_direction(i32(r), RAYS); L[r] = {0.3, 0.3, 0.3} }
	probe_fill_radiance(dirs, L, texel_dirs, tile[:])
	for v, i in tile {
		testing.expectf(t, abs(v.x - 0.3) < 1e-5, "constant field, texel %d: %v", i, v.x)
	}

	for d, r in dirs { v := 1 + d.z; L[r] = {v, v, v} }
	probe_fill_radiance(dirs, L, texel_dirs, tile[:])
	for n, i in texel_dirs {
		x, y := i % PROBE_RADIANCE_RES + 1, i / PROBE_RADIANCE_RES + 1
		got := f64(tile[y * PROBE_RADIANCE_TILE + x].x)
		testing.expectf(t, abs(got - (1 + n.z)) < 0.03, "linear field along %v: %v, want %v", n, got, 1 + n.z)
	}
}

@(test)
test_probe_octahedral_round_trip :: proc(t: ^testing.T) {
	for i in 0 ..< i32(200) {
		d := probe_ray_direction(i, 200)
		uv := probe_oct_encode(d)
		testing.expectf(t, uv.x >= 0 && uv.x <= 1 && uv.y >= 0 && uv.y <= 1, "uv out of range: %v", uv)
		back := probe_oct_decode(uv)
		testing.expectf(t, m.length(back - d) < 1e-9, "%v -> %v -> %v", d, uv, back)
	}
}

@(test)
test_probe_sidecar_path :: proc(t: ^testing.T) {
	cases := [?][2]string{
		{"scenes/box.usda", "scenes/box.probes"},
		{"box", "box.probes"},
		{"dir.v2/box", "dir.v2/box.probes"},
		{".hidden", ".hidden.probes"},
	}
	for c in cases {
		got := probe_sidecar_path(c[0])
		defer delete(got)
		testing.expectf(t, got == c[1], "%s -> %s, want %s", c[0], got, c[1])
	}
}

@(test)
test_probe_grid_dims :: proc(t: ^testing.T) {
	testing.expect_value(t, probe_grid_dims({10, 10, 10}, 4), [3]i32{4, 4, 4})
	testing.expect_value(t, probe_grid_dims({10, 5, 1}, 4), [3]i32{4, 2, 1})
	testing.expect_value(t, probe_grid_dims({10, 2.6, 1}, 8), [3]i32{8, 3, 1})
}

// A black floor under the default sky: the lowest probe, half a unit above it, sees sky above and
// nothing below. Two bakes are byte-identical, the file round-trips, and the
// revision moves when a material does.
@(test)
test_probe_bake_floor :: proc(t: ^testing.T) {
	tris := make([]Triangle, 3)
	up := Vec3{0, 1, 0}
	tris[0] = {v0 = {-1, 0, -1}, v1 = {1, 0, -1}, v2 = {1, 0, 1}, n0 = up, n1 = up, n2 = up}
	tris[1] = {v0 = {-1, 0, -1}, v1 = {1, 0, 1}, v2 = {-1, 0, 1}, n0 = up, n1 = up, n2 = up}
	// A speck in the far corner at y = 2, to give the bounds height.
	tris[2] = {v0 = {0.95, 2, 0.95}, v1 = {1, 2, 0.95}, v2 = {1, 2, 1}, n0 = up, n1 = up, n2 = up}
	meshes := make([]Mesh, 1)
	meshes[0] = Mesh{triangles = tris, transform = m.mat4(1)}
	materials := make([]Material, 1)
	materials[0] = Material{kind = .Lambertian, albedo = {0, 0, 0}}
	scene := Scene{meshes = meshes, materials = materials}
	build_default_scene_graph(&scene)
	defer {
		delete(tris); delete(meshes); delete(materials); delete(scene.nodes)
	}

	settings := Probe_Bake_Settings{grid = 2, rays = 256, max_depth = 4, max_radiance = 1000}
	a, ok_a := probe_bake(&scene, settings)
	defer probe_grid_destroy(&a)
	b, ok_b := probe_bake(&scene, settings)
	defer probe_grid_destroy(&b)
	testing.expect(t, ok_a && ok_b)
	testing.expect_value(t, a.dims, [3]i32{2, 2, 2})

	ea := probe_grid_encode(a)
	defer delete(ea)
	eb := probe_grid_encode(b)
	defer delete(eb)
	testing.expect(t, bytes.equal(ea, eb), "two bakes differ")

	testing.expect_value(t, len(a.radiance), 8 * PROBE_RADIANCE_TEXELS)
	decoded, ok_d := probe_grid_decode(ea)
	defer probe_grid_destroy(&decoded)
	testing.expect(t, ok_d, "decode failed")
	ed := probe_grid_encode(decoded)
	defer delete(ed)
	testing.expect(t, bytes.equal(ea, ed), "round trip differs")
	_, ok_short := probe_grid_decode(ea[:len(ea) - 4])
	testing.expect(t, !ok_short, "truncated file accepted")

	// The default sky's cosine-weighted mean straight up is about 0.19 (blue
	// channel 0.25). Looking down, the black floor covers most of the view;
	// the sky past its edge, half a unit away, supplies the rest.
	up_texel, _ := nearest_texel({0, 1, 0})
	down, _ := nearest_texel({0, -1, 0})
	e_up := a.irradiance[up_texel]
	e_down := a.irradiance[down]
	testing.expectf(t, e_up.z > 0.18 && e_up.z < 0.27, "sky irradiance up: %v", e_up)
	testing.expectf(t, e_down.z < 0.4 * e_up.z, "irradiance down at a black floor: %v (up %v)", e_down, e_up)

	// Straight down, the floor is at the probe's height above it.
	// Straight down, the floor is at the probe's height, along the texel's
	// direction a little farther.
	p := probe_position(a, 0, 0, 0)
	down_dir: [3]f64
	_, down_dir = nearest_texel({0, -1, 0})
	mean := f64(a.depth[down].x * a.max_distance)
	want := f64(p.y) / -down_dir.y
	testing.expectf(t, abs(mean - want) < 0.05 * f64(a.max_distance), "mean depth down %v, want about %v", mean, want)

	rev := probe_scene_revision(&scene)
	testing.expect_value(t, rev, a.revision)
	materials[0].albedo = {0.5, 0.5, 0.5}
	testing.expect(t, probe_scene_revision(&scene) != rev, "albedo change kept the revision")
}
