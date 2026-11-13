package lumbre_realtime

// Baked irradiance probes (Phase 04 step 13): the diffuse ambient term from a
// grid the path tracer baked (core/probes.odin), in place of the environment's
// irradiance map.
//
// The environment map knows nothing of the scene: every surface facing up sees
// the whole sky, a floor under a table included, and no surface receives light
// bounced off another. The probes carry both -- occlusion and one-or-more
// bounces, emissive meshes included -- at the grid's resolution. Specular
// ambient stays with the prefiltered environment and SSR.
//
// Two storage buffers, appended after the lighting pass's other buffers so no
// sampler slot moves, one entry per texel of each probe's bordered octahedral
// tile: irradiance as float4 (rgb, unused) and depth moments as float2.
// Bilinear filtering is done by hand, as storage buffers do not filter. The
// forward (glass) pass binds the same two after its own buffers.
//
// Fog takes a third, one float4 per probe: the mean radiance arriving at it
// from all directions, which is what a particle of fog scatters. The fog march
// evaluates ambient at 32 steps per pixel, so it reads these eight values per
// step, trilinear, with no visibility test -- the full surface lookup there
// would be thousands of reads a pixel.
// The lookup is DDGI's: trilinear over the eight surrounding probes, weighted by
// whether each probe faces the surface and by a Chebyshev test against its
// depth moments, so a probe behind a wall does not light the near side.
//
// A grid is used only while its revision matches the scene's. Any change it
// was baked from -- geometry, placement, materials, lights, environment --
// leaves it loaded but inactive, and lighting falls back to the environment,
// with a message saying so.

import "core:fmt"
import "core:slice"

import lc "../core"
import sdl "vendor:sdl3"

// The lookup point moves this fraction of the smallest cell size off the
// surface (mostly toward the viewer), so a probe's depth at the surface itself
// is not compared against the surface. DDGI's normal and view bias, folded.
PROBE_BIAS_FRACTION :: 0.25

Probes_GPU :: struct {
	irradiance:   ^sdl.GPUBuffer,
	depth:        ^sdl.GPUBuffer,
	mean:         ^sdl.GPUBuffer,
	revision:     u64,
	dims:         [3]i32,
	bounds_min:   [3]f32,
	bounds_max:   [3]f32,
	max_distance: f32,
	// Buffers hold a grid.
	loaded:       bool,
	// Loaded, and baked from the scene currently uploaded.
	active:       bool,
	// Scene key `active` was last decided for, and the heavy half of the
	// scene's revision under it; see `probes_check_scene`.
	checked:      bool,
	checked_key:  u64,
	heavy:        u64,
	// The decision has been printed since this grid was loaded.
	announced:    bool,
}

// Uploads `grid`, replacing any grid before it; nil unloads. Whether it is used
// is decided against the scene on the next `renderer_set_scene`, or now if a
// scene is already uploaded and `scene` is given.
renderer_set_probes :: proc(r: ^Renderer, grid: ^lc.Probe_Grid, scene: ^lc.Scene = nil) -> bool {
	probes_release(r.gpu, &r.probes)
	if grid == nil {
		return true
	}
	texels := lc.probe_count(grid^) * lc.PROBE_TILE_TEXELS
	if texels == 0 || len(grid.irradiance) != texels || len(grid.depth) != texels {
		return false
	}

	// float3 has no portable storage-buffer stride; pad to float4.
	packed := make([][4]f32, texels, context.temp_allocator)
	for v, i in grid.irradiance { packed[i] = {v.x, v.y, v.z, 0} }

	// The mean of the interior texels: each is a cosine-weighted mean
	// radiance about its direction, and averaged over all directions that
	// is the mean radiance itself (exactly so for a uniform field).
	count := lc.probe_count(grid^)
	mean := make([][4]f32, count, context.temp_allocator)
	for probe in 0 ..< count {
		sum: [3]f32
		for y in 1 ..= lc.PROBE_TILE_RES {
			for x in 1 ..= lc.PROBE_TILE_RES {
				sum += grid.irradiance[probe * lc.PROBE_TILE_TEXELS + y * lc.PROBE_TILE + x]
			}
		}
		sum /= f32(lc.PROBE_TILE_RES * lc.PROBE_TILE_RES)
		mean[probe] = {sum.x, sum.y, sum.z, 0}
	}

	p := &r.probes
	p.irradiance = make_storage_buffer(r.gpu, slice.to_bytes(packed))
	p.depth = make_storage_buffer(r.gpu, slice.to_bytes(grid.depth))
	p.mean = make_storage_buffer(r.gpu, slice.to_bytes(mean))
	if p.irradiance == nil || p.depth == nil || p.mean == nil {
		probes_release(r.gpu, p)
		return false
	}
	p.revision = grid.revision
	p.dims = grid.dims
	p.bounds_min = grid.bounds_min
	p.bounds_max = grid.bounds_max
	p.max_distance = grid.max_distance
	p.loaded = true
	if scene != nil && r.has_scene {
		probes_check_scene(r, scene, force = true)
	}
	return true
}

// Decides whether the loaded grid belongs to `scene`, printing when that
// changes. `renderer_set_scene` calls it for each new scene key; a frontend that
// edits materials or lights in place, without a new key, calls it with
// `force` after `renderer_refresh_scene`. The expensive half of the revision
// (triangles, texels) is hashed once per key, the rest on every call.
probes_check_scene :: proc(r: ^Renderer, scene: ^lc.Scene, force := false) {
	p := &r.probes
	if !p.loaded {
		p.active = false
		return
	}
	if !force && p.checked && p.checked_key == r.scene.key {
		return
	}
	if !p.checked || p.checked_key != r.scene.key {
		p.heavy = lc.probe_heavy_revision(scene)
	}
	p.checked = true
	p.checked_key = r.scene.key
	was := p.active
	p.active = lc.probe_revision_combine(p.heavy, lc.probe_light_revision(scene)) == p.revision
	if p.active != was || !p.announced {
		p.announced = true
		if p.active {
			fmt.printfln("realtime: baked GI on (%d x %d x %d probes)", p.dims.x, p.dims.y, p.dims.z)
		} else {
			fmt.println("realtime: the baked probes do not match this scene (rebake with --bake-probes); diffuse ambient falls back to the environment")
		}
	}
}

// True when the shaded view is using probes this frame.
probes_in_use :: proc(r: ^Renderer) -> bool {
	return r.probes.active && r.settings.gi_probes
}

// The three uniform vectors the shader reads; all zero when probes are off.
probes_uniforms :: proc(r: ^Renderer) -> (lo, cell, dims: [4]f32) {
	p := r.probes
	if !probes_in_use(r) {
		return
	}
	c: [3]f32
	for a in 0 ..< 3 {
		c[a] = (p.bounds_max[a] - p.bounds_min[a]) / f32(p.dims[a])
	}
	lo = {p.bounds_min.x, p.bounds_min.y, p.bounds_min.z, 1}
	cell = {c.x, c.y, c.z, p.max_distance}
	dims = {f32(p.dims.x), f32(p.dims.y), f32(p.dims.z), PROBE_BIAS_FRACTION * min(c.x, c.y, c.z)}
	return
}

// The buffers to bind: the grid's, or one-element stand-ins, since a slot
// cannot be left empty. The shader never reads them when probes are off.
probes_buffers :: proc(r: ^Renderer) -> (irradiance, depth, mean: ^sdl.GPUBuffer) {
	if r.probes.loaded {
		return r.probes.irradiance, r.probes.depth, r.probes.mean
	}
	if r.probes_dummy == nil {
		zero: [4][4]f32
		r.probes_dummy = make_storage_buffer(r.gpu, slice.to_bytes(zero[:]))
	}
	return r.probes_dummy, r.probes_dummy, r.probes_dummy
}

probes_release :: proc(gpu: ^sdl.GPUDevice, p: ^Probes_GPU) {
	if gpu != nil {
		if p.irradiance != nil { sdl.ReleaseGPUBuffer(gpu, p.irradiance) }
		if p.depth != nil { sdl.ReleaseGPUBuffer(gpu, p.depth) }
		if p.mean != nil { sdl.ReleaseGPUBuffer(gpu, p.mean) }
	}
	p^ = {}
}

@(private = "file")
make_storage_buffer :: proc(gpu: ^sdl.GPUDevice, data: []u8) -> ^sdl.GPUBuffer {
	buffer := sdl.CreateGPUBuffer(
		gpu,
		sdl.GPUBufferCreateInfo{usage = {.GRAPHICS_STORAGE_READ}, size = u32(len(data))},
	)
	if buffer == nil {
		fmt.eprintln("realtime: probe buffer failed:", sdl.GetError())
		return nil
	}
	if !upload_bytes(gpu, buffer, data) {
		sdl.ReleaseGPUBuffer(gpu, buffer)
		return nil
	}
	return buffer
}

// Loads the probe file at `path` into the renderer, replacing any grid before
// it. A missing file is not an error and prints nothing -- most scenes have no
// bake -- but one that cannot be read says so. Either way the renderer is left
// with no grid, and lighting uses the environment.
renderer_load_probes :: proc(r: ^Renderer, path: string, scene: ^lc.Scene = nil) -> bool {
	if path == "" {
		renderer_set_probes(r, nil)
		return false
	}
	grid, ok := lc.probe_grid_read(path)
	if !ok {
		if lc.probe_sidecar_exists(path) {
			fmt.eprintln("realtime: could not read", path, "-- diffuse ambient falls back to the environment")
		}
		renderer_set_probes(r, nil)
		return false
	}
	defer lc.probe_grid_destroy(&grid)
	if !renderer_set_probes(r, &grid, scene) {
		fmt.eprintln("realtime: could not upload", path)
		return false
	}
	return true
}

// `renderer_load_probes` on the sidecar next to `scene_file`.
renderer_load_probes_for :: proc(r: ^Renderer, scene_file: string, scene: ^lc.Scene = nil) -> bool {
	if scene_file == "" {
		renderer_set_probes(r, nil)
		return false
	}
	return renderer_load_probes(r, lc.probe_sidecar_path(scene_file, context.temp_allocator), scene)
}
