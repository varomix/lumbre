package lumbre_core

import "core:fmt"
import "core:os"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"
import m "core:math/linalg/glsl"
import "core:time"

// ── GPU data structs (packed for Metal) ──────────────────────────────────────

GPUMaterial :: struct {
	albedo:    [4]f32,
	emission:  [4]f32,
	params0:   [4]f32, // kind, fuzz, ir, roughness
	params1:   [4]f32, // metallic, emission_strength, specular, clearcoat
	params2:   [4]f32, // clearcoat_roughness, sheen, normal_scale, anisotropic
	spec_tint: [4]f32, // rgb = specular_tint
	sheen_tint: [4]f32, // rgb = sheen_tint
	// Each *_info is {pixel_offset, width, height, has_tex} into `tex_pixels`.
	tex_info:  [4]f32, // base color (sRGB)
	mr_info:   [4]f32, // metallic-roughness: G = rough, B = metal (linear)
	nrm_info:  [4]f32, // tangent-space normal map (linear)
	emis_info: [4]f32, // emissive (sRGB)
	params3:   [4]f32, // spec_trans, unused, unused, unused
	params4:   [4]f32, // rgb = SSS albedo, w = SSS weight
	params5:   [4]f32, // rgb = SSS mean free path (radius * scale)
}

GPULightTriangle :: struct {
	p0:       [4]f32,
	p1:       [4]f32,
	p2:       [4]f32,
	emission: [4]f32,
}

GPUQuadLight :: struct {
	position:  [4]f32,
	u:         [4]f32,
	v:         [4]f32,
	emission:  [4]f32,
}

GPUSphereLight :: struct {
	position:  [4]f32,
	emission:  [4]f32,
	radius:    f32,
	_pad:      [3]f32,
}

GPUDiscLight :: struct {
	position:  [4]f32, // xyz = center, w = radius
	normal:    [4]f32, // xyz = disc normal
	emission:  [4]f32,
}

GPUCylinderLight :: struct {
	position:  [4]f32, // xyz = base center, w = radius
	axis:      [4]f32, // xyz = axis (normalized), w = height
	emission:  [4]f32,
}

// point / spot / distant packed into one buffer.
// params = (kind, cos_inner, cos_outer, angular_radius); kind 0=point 1=spot 2=distant.
GPUPunctualLight :: struct {
	position:  [4]f32,
	direction: [4]f32,
	emission:  [4]f32,
	params:    [4]f32,
}

GICachePoint :: struct {
	position:  [4]f32,
	normal:    [4]f32,
	irradiance: [4]f32,
}

Photon :: struct {
	position: [4]f32,
	incident: [4]f32,
	power:    [4]f32,
}

GPUSceneData :: struct {
	origin:            [4]f32,
	lower_left:        [4]f32,
	horizontal:        [4]f32,
	vertical:          [4]f32,
	u:                 [4]f32,
	v:                 [4]f32,
	lens_radius:       f32,
	image_width:       i32,
	image_height:      i32,
	samples_per_pixel: i32,
	max_depth:         i32,
	max_radiance:      f32,
	debug_mode:        i32,
	tri_light_count:   i32,
	primitive_count:   i32,
	seed:              u32,
	quad_light_count:   i32,
	sphere_light_count: i32,
	roughness_cutoff:   f32,
	glossy_bias:        f32,
	gi_cache_enabled:   i32,
	gi_cache_distance:  f32,
	gi_cache_normal_angle: f32,
	gi_cache_num_points: i32,
	photon_enabled:     i32,
	photon_count:       i32,
	photon_radius:      f32,
	photon_max_bounces: i32,
	disc_light_count:     i32,
	cylinder_light_count: i32,
	punctual_light_count: i32,
	// HDRI environment (dome light).
	has_env:        i32,
	env_width:      i32,
	env_height:     i32,
	env_rotation:   f32,
	env_intensity:  f32,
	env_func_int:   f32,
	hide_default_sky: i32,
	// Progressive accumulation: how many samples are already in the accum
	// buffer. 0 means this dispatch starts a fresh image, which is what every
	// one-shot render does. Must stay in lockstep with GPUSceneData in
	// shaders/raytrace.metal.
	sample_offset:  i32,
}

GPUSphere :: struct {
	center:        [4]f32,
	radius:        f32,
	material_kind: i32,
	_fuzz:         f32,
	_ir:           f32,
	albedo:        [4]f32,
}

AxisAlignedBoundingBox :: struct {
	min: [3]f32,
	max: [3]f32,
}

// ── Sphere → mesh converter ─────────────────────────────────────────────────

// Picks a sensible photon-search / GI-cache radius from photon *density*
// rather than geometry. Photons land on surfaces, so the mean spacing between
// stored photons is ~sqrt(total_surface_area / photon_count). Keying off that
// (a) is independent of tessellation — a Cornell wall is one huge quad, a
// helmet panel is thousands of tiny tris, but both want the same gather scale;
// and (b) shrugs off a lone giant primitive (the test scene's radius-1000
// ground "sphere") far better than a bounding-box extent, which would inflate
// the radius into the hundreds and gather the whole map every lookup.
//
// Too-small a radius is as bad as too-large: the hash grid's cells shrink,
// buckets collide, and every gather scans a bloated bucket list — so we clamp
// the result to a fraction of the scene extent on both ends.
auto_gather_radius :: proc(tris: []Triangle, scene_extent: f64, photon_count: int) -> f64 {
	if len(tris) == 0 || photon_count <= 0 {
		return 0.05
	}
	area := 0.0
	for t in tris {
		area += 0.5 * m.length(m.cross(t.v1 - t.v0, t.v2 - t.v0))
	}
	spacing := m.sqrt(area / f64(photon_count))
	// A few multiples of the spacing gathers enough neighbours to smooth
	// indirect light without bleeding across features.
	K :: 4.0
	return clamp(K * spacing, 0.02, scene_extent * 0.1)
}

build_icosphere :: proc(center: Vec3, radius: f64, material: Material, allocator := context.allocator) -> []Triangle {
	// Simple UV sphere with 32×16 segments — good enough for display
	segments_u := 32
	segments_v := 16
	tri_count := segments_u * segments_v * 2
	tris := make([]Triangle, tri_count, allocator)
	idx := 0

	for j in 0 ..< segments_v {
		lat0 := m.PI * f64(j) / f64(segments_v)
		lat1 := m.PI * f64(j + 1) / f64(segments_v)
		for i in 0 ..< segments_u {
			lon0 := 2.0 * m.PI * f64(i) / f64(segments_u)
			lon1 := 2.0 * m.PI * f64(i + 1) / f64(segments_u)

			v0 := center + radius * Vec3{m.sin(lat0) * m.cos(lon0), m.cos(lat0), m.sin(lat0) * m.sin(lon0)}
			v1 := center + radius * Vec3{m.sin(lat1) * m.cos(lon0), m.cos(lat1), m.sin(lat1) * m.sin(lon0)}
			v2 := center + radius * Vec3{m.sin(lat1) * m.cos(lon1), m.cos(lat1), m.sin(lat1) * m.sin(lon1)}
			v3 := center + radius * Vec3{m.sin(lat0) * m.cos(lon1), m.cos(lat0), m.sin(lat0) * m.sin(lon1)}

			n0 := m.normalize(v0 - center)
			n1 := m.normalize(v1 - center)
			n2 := m.normalize(v2 - center)
			n3 := m.normalize(v3 - center)

			tris[idx] = Triangle{v0, v2, v1, n0, n2, n1, {}, {}, {}, false, 0}
			idx += 1
			tris[idx] = Triangle{v0, v3, v2, n0, n3, n2, {}, {}, {}, false, 0}
			idx += 1
		}
	}
	return tris
}

// ── Main GPU render function ────────────────────────────────────────────────

Render_GPU_Buffers :: struct {
	device:             ^MTL.Device,
	cmd_queue:          ^MTL.CommandQueue,
	vertex_buffer:      ^MTL.Buffer,
	index_buffer:       ^MTL.Buffer,
	material_buffer:    ^MTL.Buffer,
	scene_buffer:       ^MTL.Buffer,
	output_buffer:      ^MTL.Buffer,
	accel_struct:       ^MTL.AccelerationStructure,
	pipeline:           ^MTL.ComputePipelineState,
	ift:                ^MTL.IntersectionFunctionTable,
	num_triangles:      i32,
}

// Runs the full Metal ray-tracing pipeline (beauty + optional AOVs + optional
// denoise) and returns the result as an in-memory frame. File output (PNG/EXR)
// lives in the CLI adapter; the Houdini bridge consumes the beauty buffer.
gpu_render_frame :: proc(
	scene: ^Scene,
	image_width, image_height: i32,
	samples_per_pixel, max_depth: i32,
	max_radiance: f64,
	debug_mode: i32 = 0,
	roughness_cutoff: f64 = 0.95,
	glossy_bias: f64 = 0.0,
	gi_cache_enabled: b32 = true,
	gi_cache_distance: f32 = 0.0,
	gi_cache_normal_angle: f32 = 0.5,
	photon_enabled: b32 = true,
	photon_count: i32 = 200000,
	photon_radius: f32 = 0.0,
	photon_bounces: i32 = 8,
	enable_aovs: b32 = false,
	exr_compress: b32 = false,
	denoise_enabled: b32 = false,
	hide_default_sky: bool = false,
	// Optional persistent Metal state. When nil, one is created for this call
	// and thrown away, which is what the CLI and the Houdini bridge do. The
	// interactive viewport passes a long-lived renderer so the ~46 ms of device
	// setup and shader compilation is paid once per session, not per batch.
	renderer: ^GPU_Renderer = nil,
	// Samples already accumulated by previous dispatches. 0 starts a fresh
	// image; anything else adds to the renderer's accumulation buffer, which
	// requires a persistent `renderer`.
	sample_offset: i32 = 0,
	// Identifies the scene, so its GPU resources can be reused across calls.
	// Bump it whenever geometry, materials, or lights change; camera moves and
	// sample-count changes must not.
	scene_key: u64 = 0,
	// Whether the caller needs `beauty_linear`, the float copy of the beauty
	// image that EXR output and the Houdini bridge consume. The interactive
	// viewport displays the 8-bit sRGB buffer and nothing else, and building
	// the float copy costs an allocation and a full-image copy — per batch, at
	// several batches a second.
	want_linear: bool = true,
) -> GPU_Frame {
	total_start := time.tick_now()

	owned_renderer: GPU_Renderer
	rnd := renderer
	if rnd == nil {
		created, ok := gpu_renderer_create()
		if !ok {
			return {}
		}
		owned_renderer = created
		rnd = &owned_renderer
	}
	defer if renderer == nil {
		gpu_renderer_destroy(&owned_renderer)
	}

	device := rnd.device
	cmd_queue := rnd.queue

	// Scene-dependent GPU resources come from the renderer's cache; see
	// core/gpu_scene_cache.odin. A one-shot render still rebuilds them, because
	// its throwaway renderer starts with an empty cache.
	if !gpu_scene_cache_ensure(rnd, scene, scene_key, photon_count, gi_cache_distance, photon_radius) {
		return {}
	}
	sc := &rnd.cache

	vertex_buffer := sc.vertex_buffer
	index_buffer := sc.index_buffer
	material_buffer := sc.material_buffer
	mat_index_buffer := sc.mat_index_buffer
	tex_buffer := sc.tex_buffer
	as := sc.as
	tri_light_buffer := sc.tri_light_buffer
	quad_light_buffer := sc.quad_light_buffer
	sphere_light_buffer := sc.sphere_light_buffer
	disc_light_buffer := sc.disc_light_buffer
	cylinder_light_buffer := sc.cylinder_light_buffer
	punctual_light_buffer := sc.punctual_light_buffer
	env_pixels_buffer := sc.env_pixels_buffer
	env_marginal_buffer := sc.env_marginal_buffer
	env_conditional_buffer := sc.env_conditional_buffer
	gi_cache_buffer := sc.gi_cache_buffer
	gi_counter_buffer := sc.gi_counter_buffer
	gi_grid_cells_buffer := sc.gi_grid_cells_buffer
	gi_grid_counts_buffer := sc.gi_grid_counts_buffer
	photons_buffer := sc.photons_buffer
	photon_counter_buffer := sc.photon_counter_buffer
	photon_cell_buffer := sc.photon_cell_buffer
	photon_grid_counts_buffer := sc.photon_grid_counts_buffer
	photon_grid_offsets_buffer := sc.photon_grid_offsets_buffer
	photon_grid_fill_buffer := sc.photon_grid_fill_buffer
	photon_grid_sorted_buffer := sc.photon_grid_sorted_buffer
	num_tris := sc.num_tris
	effective_gi_cache_distance := sc.effective_gi_cache_distance
	effective_photon_radius := sc.effective_photon_radius

	cam := scene.camera
	scene_data := GPUSceneData {
		origin            = {f32(cam.origin.x), f32(cam.origin.y), f32(cam.origin.z), 0},
		lower_left        = {f32(cam.lower_left_corner.x), f32(cam.lower_left_corner.y), f32(cam.lower_left_corner.z), 0},
		horizontal        = {f32(cam.horizontal.x), f32(cam.horizontal.y), f32(cam.horizontal.z), 0},
		vertical          = {f32(cam.vertical.x), f32(cam.vertical.y), f32(cam.vertical.z), 0},
		u                 = {f32(cam.u.x), f32(cam.u.y), f32(cam.u.z), 0},
		v                 = {f32(cam.v.x), f32(cam.v.y), f32(cam.v.z), 0},
		lens_radius       = f32(cam.lens_radius),
		image_width       = image_width,
		image_height      = image_height,
		samples_per_pixel = samples_per_pixel,
		max_depth         = max_depth,
		max_radiance      = f32(max_radiance),
		debug_mode        = debug_mode,
		tri_light_count   = sc.tri_light_count,
		primitive_count   = num_tris,
		seed              = 42,
		sample_offset     = sample_offset,
		quad_light_count  = sc.quad_light_count,
		sphere_light_count = sc.sphere_light_count,
		roughness_cutoff   = f32(roughness_cutoff),
		glossy_bias        = f32(glossy_bias),
		gi_cache_enabled   = i32(gi_cache_enabled),
		gi_cache_distance  = effective_gi_cache_distance,
		gi_cache_normal_angle = gi_cache_normal_angle,
		photon_enabled     = i32(photon_enabled),
		photon_count       = photon_count,
		photon_radius      = effective_photon_radius,
		photon_max_bounces = photon_bounces,
		gi_cache_num_points = GI_CACHE_MAX_POINTS,
		disc_light_count     = sc.disc_light_count,
		cylinder_light_count = sc.cylinder_light_count,
		punctual_light_count = sc.punctual_light_count,
		has_env        = i32(sc.has_env),
		env_width      = sc.env_width,
		env_height     = sc.env_height,
		env_rotation   = sc.env_rotation,
		env_intensity  = sc.env_intensity,
		env_func_int   = sc.env_func_int,
		hide_default_sky = i32(hide_default_sky),
	}
	

	scene_slice := ([^]byte)(&scene_data)[:size_of(GPUSceneData)]
	scene_buffer := device->newBufferWithBytes(scene_slice, MTL.ResourceStorageModeShared)

	pixel_count := int(image_width) * int(image_height)
	output_buffer := device->newBufferWithLength(
		NS.UInteger(pixel_count * size_of([4]f32)),
		MTL.ResourceStorageModeShared,
	)

	// Running sample total for progressive rendering. A persistent renderer owns
	// this so batches accumulate across calls; a one-shot render gets a scratch
	// buffer, and since it starts at sample_offset 0 the kernel overwrites
	// rather than reads it, so its initial contents do not matter.
	accum_buffer: ^MTL.Buffer
	if renderer != nil {
		accum_buffer = gpu_renderer_ensure_accum(rnd, image_width, image_height)
	} else {
		accum_buffer = device->newBufferWithLength(
			NS.UInteger(pixel_count * size_of([4]f32)),
			MTL.ResourceStorageModeShared,
		)
	}

	// Shader and pipelines live on the renderer; see core/gpu_renderer.odin.
	pipeline := rnd.pipeline
	photon_pipeline := rnd.photon_emit_pipeline
	photon_count_pipeline := rnd.photon_count_pipeline
	photon_scatter_pipeline := rnd.photon_scatter_pipeline

	// Dispatch compute
	fmt.println("Rendering...")

	// Photon map build: emit → count → [CPU prefix sum] → scatter.
	// The count and scatter passes are split across two command buffers so
	// the CPU can compute the exclusive prefix sum of the per-bucket counts
	// in between (16384 buckets — trivial on the CPU, and it avoids a GPU
	// scan kernel).
	// The photon map is emitted from the lights and is camera-independent, so a
	// progressive viewport builds it once for the scene and reuses it across
	// every batch and every camera move. `photons_built` is cleared by
	// gpu_scene_cache_reset_gi when the lighting it was built from changes.
	if scene_data.photon_enabled != 0 && scene_data.photon_count > 0 && !sc.photons_built {
		sc.photons_built = true
		photon_start := time.tick_now()
		photon_n := min(scene_data.photon_count, PHOTON_MAX_COUNT)
		ph_tg := MTL.Size{width = 64, height = 1, depth = 1}
		ph_gs := MTL.Size{width = NS.Integer(photon_n), height = 1, depth = 1}

		// The emit and count passes *append* into these buffers:
		// photonEmitKernel atomic-increments photon_counter, and
		// photonCountKernel atomic-adds into photon_grid_counts. They start
		// zeroed at cache-build time, but gpu_scene_cache_reset_photons only
		// flips `photons_built` — so on a re-emit (after a material or light
		// edit settles) they still hold the previous build's totals. Left
		// stale, photon_counter climbs toward PHOTON_MAX_COUNT and every grid
		// cell's photon range grows with each rebuild, so the gather in the
		// beauty pass scans ever more entries: convergence slows and camera
		// navigation goes laggy, permanently and worse per edit. Zero them here
		// so each build starts from a clean map.
		{
			zero: i32 = 0
			copy(photon_counter_buffer->contents()[:size_of(i32)], ([^]byte)(&zero)[:size_of(i32)])
			grid_counts := photon_grid_counts_buffer->contentsAsSlice([]i32)
			for i in 0 ..< len(grid_counts) {
				grid_counts[i] = 0
			}
		}

		build_buf := cmd_queue->commandBuffer()

		// Emit
		emit_enc := build_buf->computeCommandEncoder()
		emit_enc->setComputePipelineState(photon_pipeline)
		emit_enc->setBuffer(scene_buffer, 0, 0)
		emit_enc->setBuffer(material_buffer, 0, 1)
		emit_enc->setAccelerationStructure(as, 3)
		emit_enc->setBuffer(vertex_buffer, 0, 4)
		emit_enc->setBuffer(index_buffer, 0, 5)
		emit_enc->setBuffer(tri_light_buffer, 0, 6)
		emit_enc->setBuffer(quad_light_buffer, 0, 7)
		emit_enc->setBuffer(sphere_light_buffer, 0, 8)
		emit_enc->setBuffer(mat_index_buffer, 0, 9)
		emit_enc->setBuffer(photons_buffer, 0, 14)
		emit_enc->setBuffer(photon_counter_buffer, 0, 15)
		emit_enc->setBuffer(disc_light_buffer, 0, 20)
		emit_enc->setBuffer(cylinder_light_buffer, 0, 21)
		emit_enc->setBuffer(punctual_light_buffer, 0, 22)
		emit_enc->setBuffer(env_pixels_buffer, 0, 23)
		emit_enc->setBuffer(env_marginal_buffer, 0, 24)
		emit_enc->setBuffer(env_conditional_buffer, 0, 25)
		emit_enc->dispatchThreads(ph_gs, ph_tg)
		emit_enc->endEncoding()

		// Count photons per bucket
		cnt_enc := build_buf->computeCommandEncoder()
		cnt_enc->setComputePipelineState(photon_count_pipeline)
		cnt_enc->setBuffer(photons_buffer, 0, 0)
		cnt_enc->setBuffer(photon_counter_buffer, 0, 1)
		cnt_enc->setBuffer(photon_cell_buffer, 0, 2)
		cnt_enc->setBuffer(photon_grid_counts_buffer, 0, 3)
		cnt_enc->setBuffer(scene_buffer, 0, 4)
		cnt_enc->dispatchThreads(ph_gs, ph_tg)
		cnt_enc->endEncoding()

		build_buf->commit()
		build_buf->waitUntilCompleted()

		// CPU exclusive prefix sum: counts → offsets, zero the fill cursors.
		counts := photon_grid_counts_buffer->contentsAsSlice([]i32)
		offsets := photon_grid_offsets_buffer->contentsAsSlice([]i32)
		fill := photon_grid_fill_buffer->contentsAsSlice([]i32)
		running: i32 = 0
		for i in 0 ..< PHOTON_GRID_SIZE {
			offsets[i] = running
			running += counts[i]
			fill[i] = 0
		}
		fmt.println("  Photons stored:", running)

		// Scatter photon indices into the sorted array.
		scatter_buf := cmd_queue->commandBuffer()
		sc_enc := scatter_buf->computeCommandEncoder()
		sc_enc->setComputePipelineState(photon_scatter_pipeline)
		sc_enc->setBuffer(photon_counter_buffer, 0, 1)
		sc_enc->setBuffer(photon_cell_buffer, 0, 2)
		sc_enc->setBuffer(photon_grid_offsets_buffer, 0, 3)
		sc_enc->setBuffer(photon_grid_fill_buffer, 0, 4)
		sc_enc->setBuffer(photon_grid_sorted_buffer, 0, 5)
		sc_enc->setBuffer(scene_buffer, 0, 6)
		sc_enc->dispatchThreads(ph_gs, ph_tg)
		sc_enc->endEncoding()
		scatter_buf->commit()
		scatter_buf->waitUntilCompleted()
		fmt.printfln("  Photon map build: [%.3f s]", time.duration_seconds(time.tick_since(photon_start)))
	}

	// Main raytrace pass
	trace_start := time.tick_now()
	dispatch_buf := cmd_queue->commandBuffer()
	enc := dispatch_buf->computeCommandEncoder()
	enc->setComputePipelineState(pipeline)
	enc->setBuffer(scene_buffer, 0, 0)
	enc->setBuffer(material_buffer, 0, 1)
	enc->setBuffer(output_buffer, 0, 2)
	enc->setAccelerationStructure(as, 3)
	enc->setBuffer(vertex_buffer, 0, 4)
	enc->setBuffer(index_buffer, 0, 5)
	enc->setBuffer(tri_light_buffer, 0, 6)
	enc->setBuffer(quad_light_buffer, 0, 7)
	enc->setBuffer(sphere_light_buffer, 0, 8)
	enc->setBuffer(mat_index_buffer, 0, 9)
	enc->setBuffer(gi_cache_buffer, 0, 10)
	enc->setBuffer(gi_counter_buffer, 0, 11)
	enc->setBuffer(gi_grid_cells_buffer, 0, 12)
	enc->setBuffer(gi_grid_counts_buffer, 0, 13)
	enc->setBuffer(photons_buffer, 0, 14)
	enc->setBuffer(photon_counter_buffer, 0, 15)
	enc->setBuffer(photon_grid_offsets_buffer, 0, 16)
	enc->setBuffer(photon_grid_counts_buffer, 0, 17)
	enc->setBuffer(tex_buffer, 0, 18)
	enc->setBuffer(photon_grid_sorted_buffer, 0, 19)
	enc->setBuffer(disc_light_buffer, 0, 20)
	enc->setBuffer(cylinder_light_buffer, 0, 21)
	enc->setBuffer(punctual_light_buffer, 0, 22)
	enc->setBuffer(env_pixels_buffer, 0, 23)
	enc->setBuffer(env_marginal_buffer, 0, 24)
	enc->setBuffer(env_conditional_buffer, 0, 25)
	enc->setBuffer(accum_buffer, 0, 26)

	tg_size := MTL.Size{width = 16, height = 8, depth = 1}
	grid_size := MTL.Size{
		width  = NS.Integer(image_width),
		height = NS.Integer(image_height),
		depth  = 1,
	}
	enc->dispatchThreads(grid_size, tg_size)
	enc->endEncoding()

	dispatch_buf->commit()
	dispatch_buf->waitUntilCompleted()
	fmt.printfln("  Done. [%.3f s]", time.duration_seconds(time.tick_since(trace_start)))

	// Snapshot the beauty result before the AOV/guide passes below overwrite
	// output_buffer. This is the pristine beauty image the final readback and
	// the denoiser both consume — so it is worth nothing when neither runs, and
	// the round trip out and back is two full-image copies the progressive
	// viewport was paying on every batch.
	overwrites_output := enable_aovs || denoise_enabled
	beauty_snapshot: [][4]f32
	defer delete(beauty_snapshot)
	if overwrites_output {
		beauty_snapshot = make([][4]f32, pixel_count)
		copy(beauty_snapshot, output_buffer->contentsAsSlice([][4]f32)[:pixel_count])
	}

	// AOV passes: re-run the kernel with a different debug_mode to
	// capture each AOV. We re-use the photon and GI cache state from
	// the main pass — the only thing that changes is the kernel's
	// `debug_mode` field in scene_data.
	aov_passes: []int
	if enable_aovs || denoise_enabled {
		aov_passes = []int{1, 2, 3, 5, 9} // albedo, normal, depth, direct, indirect
	} else {
		aov_passes = []int{}
	}
	aov_results := make(map[int][dynamic][4]f32)
	defer {
		for _, v in aov_results {
			delete(v)
		}
		delete(aov_results)
	}
	aov_start := time.tick_now()
	if enable_aovs || denoise_enabled {
		fmt.println("AOV passes:", len(aov_passes))
		// The GI cache and photon map built during the beauty pass persist on
		// the GPU and are reused for every AOV pass — the indirect/beauty AOVs
		// see the same biased-GI state the beauty image did.

		for aov_debug in aov_passes {
			// Update scene_data.debug_mode in the GPU buffer
			scene_data.debug_mode = i32(aov_debug)
			scene_slice := ([^]byte)(&scene_data)[:size_of(GPUSceneData)]
			dst := ([^]byte)(raw_data(scene_buffer->contents()))[:size_of(GPUSceneData)]
			copy(dst, scene_slice)

			aov_cmd := cmd_queue->commandBuffer()
			aov_enc := aov_cmd->computeCommandEncoder()
			aov_enc->setComputePipelineState(pipeline)
			aov_enc->setBuffer(scene_buffer, 0, 0)
			aov_enc->setBuffer(material_buffer, 0, 1)
			aov_enc->setBuffer(output_buffer, 0, 2)
			aov_enc->setAccelerationStructure(as, 3)
			aov_enc->setBuffer(vertex_buffer, 0, 4)
			aov_enc->setBuffer(index_buffer, 0, 5)
			aov_enc->setBuffer(tri_light_buffer, 0, 6)
			aov_enc->setBuffer(quad_light_buffer, 0, 7)
			aov_enc->setBuffer(sphere_light_buffer, 0, 8)
			aov_enc->setBuffer(mat_index_buffer, 0, 9)
			aov_enc->setBuffer(gi_cache_buffer, 0, 10)
			aov_enc->setBuffer(gi_counter_buffer, 0, 11)
			aov_enc->setBuffer(gi_grid_cells_buffer, 0, 12)
			aov_enc->setBuffer(gi_grid_counts_buffer, 0, 13)
			aov_enc->setBuffer(photons_buffer, 0, 14)
			aov_enc->setBuffer(photon_counter_buffer, 0, 15)
			aov_enc->setBuffer(photon_grid_offsets_buffer, 0, 16)
			aov_enc->setBuffer(photon_grid_counts_buffer, 0, 17)
			aov_enc->setBuffer(tex_buffer, 0, 18)
			aov_enc->setBuffer(photon_grid_sorted_buffer, 0, 19)
			aov_enc->setBuffer(disc_light_buffer, 0, 20)
			aov_enc->setBuffer(cylinder_light_buffer, 0, 21)
			aov_enc->setBuffer(punctual_light_buffer, 0, 22)
			aov_enc->setBuffer(env_pixels_buffer, 0, 23)
			aov_enc->setBuffer(env_marginal_buffer, 0, 24)
			aov_enc->setBuffer(env_conditional_buffer, 0, 25)
			aov_enc->setBuffer(accum_buffer, 0, 26)
			aov_enc->dispatchThreads(grid_size, tg_size)
			aov_enc->endEncoding()
			aov_cmd->commit()
			aov_cmd->waitUntilCompleted()

			// Read back the result for this AOV
			aov_data := output_buffer->contentsAsSlice([][4]f32)
			aov_results[aov_debug] = make([dynamic][4]f32, pixel_count)
			copy(aov_results[aov_debug][:], aov_data[:pixel_count])
		}
		// Reset debug_mode for the beauty readback
		scene_data.debug_mode = debug_mode
		fmt.printfln("  AOV passes: [%.3f s]", time.duration_seconds(time.tick_since(aov_start)))
	}

	// ── Stage 5: OpenImageDenoise ────────────────────────────────────────
	// OIDN's RT model is designed for Monte-Carlo path-traced HDR images and
	// uses albedo + normal feature buffers to keep reflective/refractive
	// details sharp. It operates on the CPU backend here: Apple Silicon's
	// unified memory makes the renderer's shared Metal readback directly usable.
	if denoise_enabled {
		fmt.println("Denoising (OpenImageDenoise RT, HDR)...")
		if !(oidn_denoise(beauty_snapshot, aov_results[1][:], aov_results[2][:], image_width, image_height, os.get_env("LUMBRE_OIDN_LIBRARY", context.temp_allocator))) {
			fmt.eprintln("OIDN denoising failed; writing the unfiltered beauty image.")
		}
	}

	// Retained temporarily as an isolated reference implementation while OIDN
	// replaces it. `when false` keeps it out of the binary and out of semantic
	// checking, so the old Metal pipeline and tuning parameters are gone.
	when false {
	// ── Previous edge-avoiding À-Trous implementation ────────────────────
	// Filters the beauty snapshot in place, guided by first-hit normal/depth
	// passes. The result is written back into output_buffer so both the PNG
	// and EXR beauty consume the denoised image.
	denoise_start := time.tick_now()
	if denoise_enabled {
		fmt.println("Denoising (A-Trous,", denoise_iterations, "iterations)...")

		buf_len := NS.UInteger(pixel_count * size_of([4]f32))
		normal_guide := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		depth_guide  := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		albedo_guide := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		emission_guide := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		color_a      := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)
		color_b      := device->newBufferWithLength(buf_len, MTL.ResourceStorageModeShared)

		// Render one raw geometry guide (first-hit-only debug mode) into
		// output_buffer, then copy it into `dst`.
		render_guide :: proc(
			cmd_queue: ^MTL.CommandQueue, pipeline: ^MTL.ComputePipelineState,
			scene_data: ^GPUSceneData, scene_buffer, material_buffer, output_buffer: ^MTL.Buffer,
			as: ^MTL.AccelerationStructure,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer,
			gi_grid_counts_buffer, photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer,
			photon_grid_counts_buffer, tex_buffer, photon_grid_sorted_buffer,
			disc_light_buffer, cylinder_light_buffer, punctual_light_buffer,
			env_pixels_buffer, env_marginal_buffer, env_conditional_buffer: ^MTL.Buffer,
			grid_size, tg_size: MTL.Size, mode: i32, dst: ^MTL.Buffer, pixel_count: int,
		) {
			scene_data.debug_mode = mode
			// Guides run at the beauty's full sample count so their primary-ray
			// jitter matches: an anti-aliased albedo/emission guide divides out
			// of the anti-aliased beauty cleanly. A 1-spp (aliased) guide leaves
			// an AA-texture/point-texture ratio in the demodulated signal that
			// the wavelet filter smears into flat posterized blocks.
			scene_slice := ([^]byte)(scene_data)[:size_of(GPUSceneData)]
			copy(([^]byte)(raw_data(scene_buffer->contents()))[:size_of(GPUSceneData)], scene_slice)

			cmd := cmd_queue->commandBuffer()
			e := cmd->computeCommandEncoder()
			e->setComputePipelineState(pipeline)
			e->setBuffer(scene_buffer, 0, 0)
			e->setBuffer(material_buffer, 0, 1)
			e->setBuffer(output_buffer, 0, 2)
			e->setAccelerationStructure(as, 3)
			e->setBuffer(vertex_buffer, 0, 4)
			e->setBuffer(index_buffer, 0, 5)
			e->setBuffer(tri_light_buffer, 0, 6)
			e->setBuffer(quad_light_buffer, 0, 7)
			e->setBuffer(sphere_light_buffer, 0, 8)
			e->setBuffer(mat_index_buffer, 0, 9)
			e->setBuffer(gi_cache_buffer, 0, 10)
			e->setBuffer(gi_counter_buffer, 0, 11)
			e->setBuffer(gi_grid_cells_buffer, 0, 12)
			e->setBuffer(gi_grid_counts_buffer, 0, 13)
			e->setBuffer(photons_buffer, 0, 14)
			e->setBuffer(photon_counter_buffer, 0, 15)
			e->setBuffer(photon_grid_offsets_buffer, 0, 16)
			e->setBuffer(photon_grid_counts_buffer, 0, 17)
			e->setBuffer(tex_buffer, 0, 18)
			e->setBuffer(photon_grid_sorted_buffer, 0, 19)
			e->setBuffer(disc_light_buffer, 0, 20)
			e->setBuffer(cylinder_light_buffer, 0, 21)
			e->setBuffer(punctual_light_buffer, 0, 22)
			e->setBuffer(env_pixels_buffer, 0, 23)
			e->setBuffer(env_marginal_buffer, 0, 24)
			e->setBuffer(env_conditional_buffer, 0, 25)
			e->dispatchThreads(grid_size, tg_size)
			e->endEncoding()
			cmd->commit()
			cmd->waitUntilCompleted()
			copy(dst->contentsAsSlice([][4]f32)[:pixel_count], output_buffer->contentsAsSlice([][4]f32)[:pixel_count])
		}

		render_guide(cmd_queue, pipeline, &scene_data, scene_buffer, material_buffer, output_buffer, as,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer, gi_grid_counts_buffer,
			photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer, photon_grid_counts_buffer,
			tex_buffer, photon_grid_sorted_buffer,
			disc_light_buffer, cylinder_light_buffer, punctual_light_buffer,
			env_pixels_buffer, env_marginal_buffer, env_conditional_buffer,
			grid_size, tg_size, 20, normal_guide, pixel_count)
		render_guide(cmd_queue, pipeline, &scene_data, scene_buffer, material_buffer, output_buffer, as,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer, gi_grid_counts_buffer,
			photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer, photon_grid_counts_buffer,
			tex_buffer, photon_grid_sorted_buffer,
			disc_light_buffer, cylinder_light_buffer, punctual_light_buffer,
			env_pixels_buffer, env_marginal_buffer, env_conditional_buffer,
			grid_size, tg_size, 21, depth_guide, pixel_count)
		render_guide(cmd_queue, pipeline, &scene_data, scene_buffer, material_buffer, output_buffer, as,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer, gi_grid_counts_buffer,
			photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer, photon_grid_counts_buffer,
			tex_buffer, photon_grid_sorted_buffer,
			disc_light_buffer, cylinder_light_buffer, punctual_light_buffer,
			env_pixels_buffer, env_marginal_buffer, env_conditional_buffer,
			grid_size, tg_size, 22, albedo_guide, pixel_count)
		render_guide(cmd_queue, pipeline, &scene_data, scene_buffer, material_buffer, output_buffer, as,
			vertex_buffer, index_buffer, tri_light_buffer, quad_light_buffer, sphere_light_buffer,
			mat_index_buffer, gi_cache_buffer, gi_counter_buffer, gi_grid_cells_buffer, gi_grid_counts_buffer,
			photons_buffer, photon_counter_buffer, photon_grid_offsets_buffer, photon_grid_counts_buffer,
			tex_buffer, photon_grid_sorted_buffer,
			disc_light_buffer, cylinder_light_buffer, punctual_light_buffer,
			env_pixels_buffer, env_marginal_buffer, env_conditional_buffer,
			grid_size, tg_size, 23, emission_guide, pixel_count)

		// Albedo demodulation: filter illumination, not texture detail. Divide
		// the beauty by a clamped first-hit albedo, filter that illumination
		// signal, then remultiply by the same clamped albedo. Using the *same*
		// clamped value for divide and multiply makes the transform an exact
		// inverse where albedo is near zero (background, emitters), so those
		// pixels pass through unharmed.
		// Albedo floor. Dividing by a near-zero albedo channel amplifies noise
		// into self-preserving outliers the wavelet can't remove. Texture
		// detail lives in the *bright* albedo channels (which stay above the
		// floor and demodulate fully); dark channels carry no visible detail,
		// so flooring them well above zero caps amplification with no loss.
		DEMOD_EPS :: f32(0.15)
		clamped_albedo := make([][4]f32, pixel_count)
		defer delete(clamped_albedo)
		albedo_src := albedo_guide->contentsAsSlice([][4]f32)
		emission_src := emission_guide->contentsAsSlice([][4]f32)
		color_a_dst := color_a->contentsAsSlice([][4]f32)
		for i in 0 ..< pixel_count {
			ar := max(albedo_src[i][0], DEMOD_EPS)
			ag := max(albedo_src[i][1], DEMOD_EPS)
			ab := max(albedo_src[i][2], DEMOD_EPS)
			clamped_albedo[i] = {ar, ag, ab, 1.0}
			// Remove the noise-free self-emission (HUD/screen glow, emissive
			// textures) before demodulating. Otherwise it lands in the
			// illumination signal and the wavelet filter smears it across the
			// surface. It is added back, untouched, after filtering.
			color_a_dst[i] = {
				(beauty_snapshot[i][0] - emission_src[i][0]) / ar,
				(beauty_snapshot[i][1] - emission_src[i][1]) / ag,
				(beauty_snapshot[i][2] - emission_src[i][2]) / ab,
				beauty_snapshot[i][3],
			}
		}

		dn_params := GPUDenoiseParams{
			width        = image_width,
			height       = image_height,
			step_width   = 1,
			sigma_color  = denoise_c_sigma,
			sigma_normal = denoise_n_sigma,
			sigma_depth  = denoise_d_sigma,
		}
		dn_params_buffer := device->newBufferWithLength(NS.UInteger(size_of(GPUDenoiseParams)), MTL.ResourceStorageModeShared)

		iters := int(denoise_iterations)
		if iters < 1 { iters = 1 }
		src, dst := color_a, color_b
		for it in 0 ..< iters {
			dn_params.step_width = i32(1 << uint(it))
			copy(([^]byte)(raw_data(dn_params_buffer->contents()))[:size_of(GPUDenoiseParams)],
				([^]byte)(&dn_params)[:size_of(GPUDenoiseParams)])

			dn_cmd := cmd_queue->commandBuffer()
			dn_enc := dn_cmd->computeCommandEncoder()
			dn_enc->setComputePipelineState(denoise_pipeline)
			dn_enc->setBuffer(dn_params_buffer, 0, 0)
			dn_enc->setBuffer(src, 0, 1)
			dn_enc->setBuffer(dst, 0, 2)
			dn_enc->setBuffer(normal_guide, 0, 3)
			dn_enc->setBuffer(depth_guide, 0, 4)
			dn_enc->dispatchThreads(grid_size, tg_size)
			dn_enc->endEncoding()
			dn_cmd->commit()
			dn_cmd->waitUntilCompleted()

			src, dst = dst, src
		}

		// Firefly pre-clamp on the demodulated illumination. Dividing a dark
		// beauty spike by a small albedo channel yields an extreme outlier,
		// and À-Trous self-preserves outliers (every neighbour reads as an
		// edge next to the spike), so they survive filtering. Clamp each
		// pixel's luminance to a multiple of its 3×3 neighbourhood mean before
		// the wavelet passes.
		{
			luminance :: proc(c: [4]f32) -> f32 {
				return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
			}
			FIREFLY_K :: f32(3.0)
			w := int(image_width)
			h := int(image_height)
			illum := color_a->contentsAsSlice([][4]f32)
			clamped := make([][4]f32, pixel_count)
			defer delete(clamped)
			copy(clamped, illum[:pixel_count])
			for y in 0 ..< h {
				for x in 0 ..< w {
					idx := y * w + x
					sum: f32 = 0
					n: f32 = 0
					for dy in -1 ..= 1 {
						for dx in -1 ..= 1 {
							sx := x + dx
							sy := y + dy
							if sx < 0 || sx >= w || sy < 0 || sy >= h {
								continue
							}
							sum += luminance(illum[sy * w + sx])
							n += 1
						}
					}
					mean := sum / n
					lum := luminance(illum[idx])
					limit := mean * FIREFLY_K
					if lum > limit && lum > 0 {
						scale := limit / lum
						clamped[idx] = {
							illum[idx][0] * scale,
							illum[idx][1] * scale,
							illum[idx][2] * scale,
							illum[idx][3],
						}
					}
				}
			}
			copy(illum[:pixel_count], clamped)
		}

		// `src` holds the final filtered illumination. Remodulate by the
		// clamped albedo to restore texture detail, and push it back into the
		// beauty snapshot and output_buffer.
		filtered := src->contentsAsSlice([][4]f32)
		for i in 0 ..< pixel_count {
			beauty_snapshot[i] = {
				filtered[i][0] * clamped_albedo[i][0] + emission_src[i][0],
				filtered[i][1] * clamped_albedo[i][1] + emission_src[i][1],
				filtered[i][2] * clamped_albedo[i][2] + emission_src[i][2],
				filtered[i][3],
			}
		}
		fmt.printfln("  Done. [%.3f s]", time.duration_seconds(time.tick_since(denoise_start)))
	}

	}

	// Restore the beauty image (denoised or raw) into output_buffer — the AOV
	// passes above leave the last AOV in it. With no such pass, output_buffer
	// still holds the beauty image and there is nothing to put back.
	if overwrites_output {
		copy(output_buffer->contentsAsSlice([][4]f32)[:pixel_count], beauty_snapshot)
	}

	// Readback: encode linear beauty into an 8-bit sRGB buffer and keep a
	// linear copy for EXR. File writing (PNG/EXR) lives in the CLI adapter.
	fmt.printfln("Total render time: %.3f s", time.duration_seconds(time.tick_since(total_start)))

	output_data := output_buffer->contentsAsSlice([][4]f32)
	pixels := make([]u8, pixel_count * 3)
	beauty_linear: [][4]f32
	if want_linear {
		beauty_linear = make([][4]f32, pixel_count)
		copy(beauty_linear, output_data[:pixel_count])
	}
	for i in 0 ..< pixel_count {
		lr := linear_to_srgb(clamp(f64(output_data[i][0]), 0.0, 1.0))
		lg := linear_to_srgb(clamp(f64(output_data[i][1]), 0.0, 1.0))
		lb := linear_to_srgb(clamp(f64(output_data[i][2]), 0.0, 1.0))
		pixels[i * 3 + 0] = u8(clamp(lr * 255.0, 0.0, 255.0))
		pixels[i * 3 + 1] = u8(clamp(lg * 255.0, 0.0, 255.0))
		pixels[i * 3 + 2] = u8(clamp(lb * 255.0, 0.0, 255.0))
	}

	frame := GPU_Frame{
		width         = image_width,
		height        = image_height,
		pixels        = pixels,
		beauty_linear = beauty_linear,
		aov_results   = aov_results,
	}
	// Ownership of aov_results transfers to the caller; neutralize the defer
	// above so it does not free what we just returned.
	aov_results = nil
	return frame
}
