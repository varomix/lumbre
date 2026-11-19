package lumbre_realtime

// The realtime rasterization renderer.
//
// Phase 01 of plans/REALTIME_PHASE01.md. Unlike the path tracer — Metal compute
// that returns CPU pixels — this draws through SDL_GPU straight into a texture
// ImGui already knows how to display, so a realtime frame never round-trips
// through system memory.
//
// SDL_GPU is the render hardware interface here, with no Lumbre-owned
// abstraction over it: the raster pipeline is new code rather than a port, so
// there was nothing to abstract, and SDL already covers Metal, Vulkan and
// D3D12 behind one API.
//
// Cascaded shadows and a G-buffer feed deferred HDR lighting, followed by
// display encoding. Debug and label passes retain their own data conventions.

import "core:fmt"
import "core:hash/xxhash"
import "core:math"
import "core:math/linalg"
import "core:slice"

import lc "../core"
import sdl "vendor:sdl3"

// What the viewport shows. Everything but `Shaded` reads one G-buffer channel
// raw, which is how a wrong normal or a mis-batched texture becomes visible
// rather than merely suspected.
Debug_View :: enum i32 {
	Shaded    = 0, // deferred lighting; falls back to albedo until it exists
	Albedo    = 1,
	Normal    = 2,
	Roughness = 3,
	Metallic  = 4,
	Emission  = 5,
	Depth     = 6,
	// Ground-truth channels from the label pass, colour-hashed for display.
	// The buffers themselves hold raw ids and metres; see labels.odin.
	Instance  = 7,
	Semantic  = 8,
	// What screen-space reflection found, alone: the reflected colour where a
	// ray hit, black where the environment is left in place. Drawn through the
	// shaded path, so exposure and the view transform apply.
	Reflection = 9,
	// Ambient occlusion: the visibility the lighting pass multiplies the
	// ambient light by, 1 where nothing occludes.
	AO         = 10,
	// Screen-space velocity from the motion pass: red/green are x/y pixels
	// (scaled, mid-grey is still), unjittered. See motion.odin.
	Motion     = 11,
}

// How linear HDR becomes display pixels. See shaders/display_fs.slang.
View_Transform :: enum i32 {
	Standard = 0, // clamp + sRGB, the path tracer's transform
	Neutral  = 1, // Khronos PBR Neutral
	AgX      = 2,
}

// Per-frame look controls. Changing any of them needs only a redraw, never a
// scene upload.
Render_Settings :: struct {
	exposure:       f32, // stops, applied before the view transform
	view_transform: View_Transform,
	fxaa:           bool, // post-process antialiasing of the shaded view
	// Temporal antialiasing of the shaded view. Replaces FXAA while on: the
	// two together only blur.
	taa:            bool,
	ao:             bool, // screen-space ambient occlusion
	// Multiplies the automatic AO radius, AO_RADIUS_FRACTION of the scene's
	// largest extent.
	ao_radius:      f32,
	ssr:            bool, // screen-space reflections on smooth surfaces
	// Height fog, lit by the sky, the sun and the local lights. Off by
	// default: unlike the passes above it is a look, not a correction.
	fog:            bool,
	// Optical depth of a path across the scene's largest extent at floor
	// level. 1 leaves about a third of what is that far away.
	fog_density:    f32,
	// Height over which the fog thins by e, as a fraction of the scene's
	// height -- or of FOG_FLAT_SCENE_HEIGHT of its largest extent, for a
	// scene too flat for its own height to mean anything. The floor is the
	// bottom of the scene's bounds.
	fog_height:     f32,
	// Henyey-Greenstein g: 0 scatters evenly, toward 1 glows around the sun.
	fog_anisotropy: f32,
	// Scattering albedo: the share of each channel the fog scatters rather
	// than absorbs. White fog is {1, 1, 1}; a dusty haze is warmer and darker.
	fog_colour:     [3]f32,
	// Light scattered more than once, approximated (see fog_fs). Thick fog
	// glows with it; without, it is only lit along each light's own rays.
	fog_multiple:   bool,
	// Diffuse ambient from baked irradiance probes when a matching grid is
	// loaded (probes.odin); otherwise from the environment.
	gi_probes:      bool,
}

DEFAULT_RENDER_SETTINGS :: Render_Settings {
	fxaa      = true,
	taa       = true,
	ao        = true,
	ao_radius = 1,
	ssr       = true,
	fog_density    = 0.5,
	fog_height     = 0.5,
	fog_anisotropy = 0.5,
	fog_colour     = {1, 1, 1},
	fog_multiple   = true,
	gi_probes      = true,
}

// The least a scene's height counts as when scaling the fog's falloff, as a
// fraction of its largest extent. A ground plane with nothing on it would
// otherwise get a layer of fog as thin as the plane.
FOG_FLAT_SCENE_HEIGHT :: 0.1

// The depth pyramid screen-space reflection marches. Eight levels let a ray
// cross 128 pixels of empty screen in a step; more would only be climbed by
// rays with nothing left to hit.
HIZ_FORMAT :: sdl.GPUTextureFormat.R32_FLOAT
HIZ_MAX_LEVELS :: 8
// How far behind a surface a ray may arrive and still count as hitting it, as
// a fraction of the scene's largest extent.
SSR_THICKNESS_FRACTION :: 0.005

// The AO radius as a fraction of the scene's largest extent: about 7 cm in a
// five-metre room, which is the scale of the contact shadows AO is for.
AO_RADIUS_FRACTION :: 0.015
// An exponent on the ambient visibility. 1 is GTAO's own answer.
AO_INTENSITY :: 1.0
AO_FORMAT :: sdl.GPUTextureFormat.R8_UNORM

// Temporal antialiasing. The jitter cycles through this many points of a
// Halton (2, 3) sequence: enough to cover a pixel evenly, few enough that a
// still view settles in about as many frames.
TAA_SAMPLES :: 16
// How many frames a caller should keep drawing after the last change for the
// history to settle. The viewport redraws this many times after the camera
// stops; a headless still renders this many and keeps the last.
TAA_SETTLE_FRAMES :: 16
// The current frame's weight while the camera moves: lower ghosts more,
// higher shimmers more.
TAA_ALPHA_MOVING :: 0.1
// The least it falls to while the camera is still and the history converges.
TAA_ALPHA_STILL :: 1.0 / 24.0

// The presented image. UNORM rather than an sRGB format because the debug pass
// encodes gamma itself.
COLOR_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM
HDR_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT

// G-buffer layout. See shaders/gbuffer_fs.slang for what each channel holds.
ALBEDO_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM
NORMAL_FORMAT :: sdl.GPUTextureFormat.R16G16_FLOAT
SURFACE_FORMAT :: sdl.GPUTextureFormat.R8G8B8A8_UNORM
EMISSION_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT
DEPTH_FORMAT :: sdl.GPUTextureFormat.D32_FLOAT

Renderer :: struct {
	gpu:              ^sdl.GPUDevice, // borrowed from the app; not owned
	settings:         Render_Settings,
	gbuffer_pipeline: ^sdl.GPUGraphicsPipeline,
	debug_pipeline:   ^sdl.GPUGraphicsPipeline,
	lighting_pipeline: ^sdl.GPUGraphicsPipeline,
	display_pipeline: ^sdl.GPUGraphicsPipeline,
	// FXAA reads the display-encoded image from `ldr` and writes `color`.
	fxaa_pipeline:    ^sdl.GPUGraphicsPipeline,
	ldr:              ^sdl.GPUTexture,
	linear_sampler:   ^sdl.GPUSampler,
	// Ambient occlusion: raw, then blurred into `ao`, which lighting reads.
	ao_pipeline:      ^sdl.GPUGraphicsPipeline,
	ao_blur_pipeline: ^sdl.GPUGraphicsPipeline,
	ao_raw:           ^sdl.GPUTexture,
	ao:               ^sdl.GPUTexture,
	// Screen-space reflections: `hiz` is the depth pyramid, one pass per level.
	// With reflections on, lighting draws into `refraction` and the reflection
	// pass composites from there into `hdr_color`.
	hiz_pipeline:     ^sdl.GPUGraphicsPipeline,
	ssr_pipeline:     ^sdl.GPUGraphicsPipeline, // one ray per pixel, into ssr_trace
	ssr_resolve_pipeline: ^sdl.GPUGraphicsPipeline, // shares them out, into hdr_color
	ssr_trace:        ^sdl.GPUTexture,
	hiz:              ^sdl.GPUTexture,
	hiz_levels:       u32,
	// Point-filtered like `target_sampler`, but allowed past level 0.
	hiz_sampler:      ^sdl.GPUSampler,
	// Trilinear and clamped: SSR reads `refraction`'s mips by the size of a
	// rough surface's reflection cone.
	mip_sampler:      ^sdl.GPUSampler,
	// Fog: marched into `fog`, then blurred and blended over `hdr_color`.
	fog_pipeline:       ^sdl.GPUGraphicsPipeline,
	fog_apply_pipeline: ^sdl.GPUGraphicsPipeline,
	fog:                ^sdl.GPUTexture,
	hdr_color: ^sdl.GPUTexture,
	shadow_pipeline:  ^sdl.GPUGraphicsPipeline,

	// Temporal antialiasing. The two history targets alternate: one is read
	// while the other is written, and the written one is what display reads.
	taa_pipeline:   ^sdl.GPUGraphicsPipeline,
	taa_history:    [2]^sdl.GPUTexture,
	taa_write:      int,  // the history target written last
	taa_valid:      bool, // whether that target holds a usable history
	taa_view:       Debug_View, // the view it was accumulated for
	taa_frame:      u32,  // advances the jitter sequence
	taa_still:      u32,  // consecutive frames from one camera
	taa_on:         bool, // whether the last frame was antialiased this way
	taa_jitter:     [2]f32, // this frame's offset in NDC; zero when off
	taa_prev_view_proj: matrix[4, 4]f32, // last frame's, unjittered

	// Per-pass GPU timing. Off unless asked for: it splits the frame into one
	// command buffer per pass and waits for each, which serialises the GPU
	// against the CPU. See profile.odin.
	profile:          bool,
	profiler:         Profiler,

	// Forward pass for transmissive surfaces, drawn after deferred lighting.
	// `refraction` is the lit opaque image, copied so the pass can read what is
	// behind a surface while writing to `hdr_color`. With reflections on, its
	// mips are a blur pyramid for glossy reflections.
	forward_pipeline: ^sdl.GPUGraphicsPipeline,
	refraction:       ^sdl.GPUTexture,
	refraction_levels: u32,
	transparent_scratch: [dynamic]Transparent_Draw,

	// Depth array, one layer per cascade, plus the comparison sampler that
	// does the depth test in hardware.
	shadow_map:       ^sdl.GPUTexture,
	shadow_sampler:   ^sdl.GPUSampler,
	// What each cascade layer currently holds: its matrix, and the scene
	// uploads it was drawn from. A layer whose matrix and scene are unchanged
	// is not redrawn -- the look controls redraw the frame without moving
	// anything, and texel snapping keeps far cascades still under small camera
	// moves.
	shadow_drawn:     [CASCADE_COUNT]matrix[4, 4]f32,
	shadow_valid:     [CASCADE_COUNT]bool,
	shadow_stamp:     [2]u64, // geometry_uploads, instance_uploads

	// Spot and omni (point and sphere) light shadows: a layer per spot, six
	// per omni light, each array created with the first scene that needs it.
	// Layers are cached the way the cascades are, but a local light's view
	// does not follow the camera, so a still scene never redraws one.
	spots:            Spot_Shadows,
	omnis:            Omni_Shadows,
	spot_layers:      Shadow_Layers,
	omni_layers:      Shadow_Layers,
	// Shadows through glass: the sun's cascades' glass maps (the spot and omni
	// ones live in their Shadow_Layers), the pipeline that draws them, a 1x1
	// stand-in for scenes without glass, and what the glass looked like when
	// they were drawn (see glass_key).
	glass_shadow_pipeline: ^sdl.GPUGraphicsPipeline,
	sun_glass:        ^sdl.GPUTexture,
	glass_none:       ^sdl.GPUTexture,
	has_glass:        bool,
	glass_drawn_key:  u64,

	// Image-based lighting. `brdf_lut` depends only on roughness and viewing
	// angle, so it is built once here rather than per scene.
	irradiance_pipeline: ^sdl.GPUGraphicsPipeline,
	specular_pipeline:   ^sdl.GPUGraphicsPipeline,
	brdf_lut:            ^sdl.GPUTexture,
	env:                 Environment_GPU,
	// Area lights' fitted lobes, built once; see area_lights.odin.
	ltc:                 Ltc_Textures,
	// Baked irradiance probes, when the frontend loaded a grid; see probes.odin.
	probes:              Probes_GPU,
	probes_dummy:        ^sdl.GPUBuffer,

	// Ground-truth label targets. Rendered only when a label view is showing
	// or a caller asks for them: a viewport doing lookdev has no use for
	// instance ids, and the pass is a full redraw of the scene.
	labels:              Labels,
	// Velocity target, drawn only for the motion view or a caller that asks.
	motion:              Motion,
	// Where the camera was one frame earlier, for the motion view; nil means
	// it has not moved, leaving object motion only. Set by the caller, since
	// only it knows (a stage's previous time code, or the last frame shown).
	motion_prev_camera:  Maybe(lc.Camera),

	// Which lights reach which froxel, rebuilt per frame because it follows the
	// camera as much as the lights. See clusters.odin.
	clusters:         Cluster_Grid,

	// Analytic lights, re-uploaded whenever the scene's light list changes.
	light_buffer:     ^sdl.GPUBuffer,
	light_capacity:   u32,
	light_count:      u32,
	light_scratch:    [dynamic]Light_GPU,

	// Reads G-buffer targets in the fullscreen passes. Point-filtered and
	// clamped: these are screen-aligned, so any filtering would only smear
	// neighbouring texels of data that is not a colour.
	target_sampler:   ^sdl.GPUSampler,

	// Render targets, resized to follow the viewport panel.
	color:            ^sdl.GPUTexture,
	albedo:           ^sdl.GPUTexture,
	normal:           ^sdl.GPUTexture,
	surface:          ^sdl.GPUTexture,
	emission:         ^sdl.GPUTexture,
	depth:            ^sdl.GPUTexture,
	width:            i32,
	height:           i32,

	// The uploaded scene. Content revisions avoid geometry work even when
	// a script publishes a new scene key with unchanged mesh data.
	scene:            Scene_GPU,
	has_scene:        bool,
	texture_cache: Texture_Cache,
	geometry_key, instance_key, texture_key, environment_key, lights_key, labels_key: u64,
	environment_ready, lights_ready, labels_ready: bool,
	geometry_uploads, instance_uploads, environment_builds: u64,
	shadow_layer_draws: u64,
}

renderer_create :: proc(gpu: ^sdl.GPUDevice) -> (result: Renderer, ok: bool) {
	r: Renderer
	defer if !ok { renderer_destroy(&r) }
	r.gpu = gpu
	r.settings = DEFAULT_RENDER_SETTINGS
	r.texture_cache = make(Texture_Cache)

	r.gbuffer_pipeline = make_gbuffer_pipeline(gpu) or_return
	r.debug_pipeline = make_debug_pipeline(gpu) or_return
	r.lighting_pipeline = make_lighting_pipeline(gpu) or_return
	r.display_pipeline = make_prefilter_pipeline(gpu, SHADER_DISPLAY_FS, COLOR_FORMAT) or_return
	r.fxaa_pipeline = make_prefilter_pipeline(gpu, SHADER_FXAA_FS, COLOR_FORMAT) or_return
	r.ao_pipeline = make_fullscreen_pipeline(gpu, SHADER_AO_FS, AO_FORMAT, {samplers = 2, uniform_buffers = 1}) or_return
	r.ao_blur_pipeline = make_fullscreen_pipeline(gpu, SHADER_AO_BLUR_FS, AO_FORMAT, {samplers = 2, uniform_buffers = 1}) or_return
	r.hiz_pipeline = make_fullscreen_pipeline(gpu, SHADER_HIZ_FS, HIZ_FORMAT, {samplers = 1, uniform_buffers = 1}) or_return
	r.ssr_pipeline = make_fullscreen_pipeline(gpu, SHADER_SSR_FS, HDR_FORMAT, {samplers = 4, uniform_buffers = 1}) or_return
	r.ssr_resolve_pipeline = make_fullscreen_pipeline(gpu, SHADER_SSR_RESOLVE_FS, HDR_FORMAT, {samplers = 9, storage_buffers = 4, uniform_buffers = 1}) or_return
	r.fog_pipeline = make_fullscreen_pipeline(gpu, SHADER_FOG_FS, HDR_FORMAT, {samplers = 5, storage_buffers = 4, uniform_buffers = 2}) or_return
	// Target becomes scattered + target * transmittance.
	r.fog_apply_pipeline = make_fullscreen_pipeline(
		gpu, SHADER_FOG_APPLY_FS, HDR_FORMAT, {samplers = 2, uniform_buffers = 1},
		sdl.GPUColorTargetBlendState {
			enable_blend = true,
			src_color_blendfactor = .ONE,
			dst_color_blendfactor = .SRC_ALPHA,
			color_blend_op = .ADD,
			src_alpha_blendfactor = .ZERO,
			dst_alpha_blendfactor = .ONE,
			alpha_blend_op = .ADD,
		},
	) or_return
	r.taa_pipeline = make_fullscreen_pipeline(gpu, SHADER_TAA_FS, HDR_FORMAT, {samplers = 3, uniform_buffers = 1}) or_return
	r.shadow_pipeline = make_shadow_pipeline(gpu) or_return
	r.glass_shadow_pipeline = make_glass_shadow_pipeline(gpu) or_return
	r.forward_pipeline = make_forward_pipeline(gpu) or_return
	r.irradiance_pipeline = make_prefilter_pipeline(gpu, SHADER_ENV_IRRADIANCE_FS, ENV_FORMAT) or_return
	r.specular_pipeline = make_prefilter_pipeline(gpu, SHADER_ENV_SPECULAR_FS, ENV_FORMAT) or_return
	r.labels = labels_create(gpu) or_return
	r.motion = motion_create(gpu) or_return
	r.ltc = ltc_create(gpu) or_return

	r.shadow_map = sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = .D2_ARRAY,
			format = DEPTH_FORMAT,
			usage = {.DEPTH_STENCIL_TARGET, .SAMPLER},
			width = SHADOW_RESOLUTION,
			height = SHADOW_RESOLUTION,
			layer_count_or_depth = CASCADE_COUNT,
			num_levels = 1,
			sample_count = ._1,
		},
	)
	if r.shadow_map == nil {
		fmt.eprintln("realtime: shadow map creation failed:", sdl.GetError())
		return {}, false
	}

	// A comparison sampler filters the RESULT of the depth test, not the
	// depths. Filtering depths and comparing afterwards gives wrong occlusion
	// along every shadow edge.
	r.shadow_sampler = sdl.CreateGPUSampler(
		gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .LINEAR,
			mag_filter = .LINEAR,
			mipmap_mode = .NEAREST,
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
			compare_op = .LESS_OR_EQUAL,
			enable_compare = true,
		},
	)
	if r.shadow_sampler == nil {
		fmt.eprintln("realtime: shadow sampler creation failed:", sdl.GetError())
		return {}, false
	}

	r.target_sampler = sdl.CreateGPUSampler(
		gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .NEAREST,
			mag_filter = .NEAREST,
			mipmap_mode = .NEAREST,
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
		},
	)
	if r.target_sampler == nil {
		fmt.eprintln("realtime: CreateGPUSampler failed:", sdl.GetError())
		return {}, false
	}

	// FXAA's edge walk samples between texels on purpose.
	r.linear_sampler = sdl.CreateGPUSampler(
		gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .LINEAR,
			mag_filter = .LINEAR,
			mipmap_mode = .NEAREST,
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
		},
	)
	if r.linear_sampler == nil {
		fmt.eprintln("realtime: CreateGPUSampler failed:", sdl.GetError())
		return {}, false
	}

	r.hiz_sampler = sdl.CreateGPUSampler(
		gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .NEAREST,
			mag_filter = .NEAREST,
			mipmap_mode = .NEAREST,
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
			max_lod = HIZ_MAX_LEVELS,
		},
	)
	if r.hiz_sampler == nil {
		fmt.eprintln("realtime: CreateGPUSampler failed:", sdl.GetError())
		return {}, false
	}

	r.mip_sampler = sdl.CreateGPUSampler(
		gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .LINEAR,
			mag_filter = .LINEAR,
			mipmap_mode = .LINEAR,
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
			max_lod = 1000,
		},
	)
	if r.mip_sampler == nil {
		fmt.eprintln("realtime: CreateGPUSampler failed:", sdl.GetError())
		return {}, false
	}

	if !build_brdf_lut(&r) {
		return {}, false
	}

	return r, true
}

renderer_destroy :: proc(r: ^Renderer) {
	if r.gpu == nil {
		return
	}
	scene_destroy(r.gpu, &r.scene)
	texture_cache_destroy(r.gpu, &r.texture_cache)
	release_targets(r)
	if r.target_sampler != nil {
		sdl.ReleaseGPUSampler(r.gpu, r.target_sampler)
	}
	if r.gbuffer_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.gbuffer_pipeline)
	}
	if r.debug_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.debug_pipeline)
	}
	if r.lighting_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.lighting_pipeline)
	}
	if r.display_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.display_pipeline) }
	if r.fxaa_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.fxaa_pipeline) }
	if r.ao_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.ao_pipeline) }
	if r.ao_blur_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.ao_blur_pipeline) }
	if r.hiz_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.hiz_pipeline) }
	if r.ssr_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.ssr_pipeline) }
	if r.ssr_resolve_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.ssr_resolve_pipeline) }
	if r.fog_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.fog_pipeline) }
	if r.fog_apply_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.fog_apply_pipeline) }
	if r.taa_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.taa_pipeline) }
	if r.linear_sampler != nil { sdl.ReleaseGPUSampler(r.gpu, r.linear_sampler) }
	if r.hiz_sampler != nil { sdl.ReleaseGPUSampler(r.gpu, r.hiz_sampler) }
	if r.mip_sampler != nil { sdl.ReleaseGPUSampler(r.gpu, r.mip_sampler) }
	if r.shadow_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.shadow_pipeline)
	}
	if r.forward_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.forward_pipeline)
	}
	delete(r.transparent_scratch)
	if r.shadow_map != nil {
		sdl.ReleaseGPUTexture(r.gpu, r.shadow_map)
	}
	for tex in ([]^sdl.GPUTexture{r.spot_layers.glass, r.omni_layers.glass, r.sun_glass, r.glass_none}) {
		if tex != nil { sdl.ReleaseGPUTexture(r.gpu, tex) }
	}
	if r.glass_shadow_pipeline != nil { sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.glass_shadow_pipeline) }
	if r.spot_layers.texture != nil {
		sdl.ReleaseGPUTexture(r.gpu, r.spot_layers.texture)
	}
	if r.omni_layers.texture != nil {
		sdl.ReleaseGPUTexture(r.gpu, r.omni_layers.texture)
	}
	if r.shadow_sampler != nil {
		sdl.ReleaseGPUSampler(r.gpu, r.shadow_sampler)
	}
	labels_destroy(r.gpu, &r.labels)
	motion_destroy(r.gpu, &r.motion)
	ltc_destroy(r.gpu, &r.ltc)
	env_destroy(r.gpu, &r.env)
	if r.brdf_lut != nil {
		sdl.ReleaseGPUTexture(r.gpu, r.brdf_lut)
	}
	if r.irradiance_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.irradiance_pipeline)
	}
	if r.specular_pipeline != nil {
		sdl.ReleaseGPUGraphicsPipeline(r.gpu, r.specular_pipeline)
	}
	if r.light_buffer != nil {
		sdl.ReleaseGPUBuffer(r.gpu, r.light_buffer)
	}
	clusters_destroy(r.gpu, &r.clusters)
	probes_release(r.gpu, &r.probes)
	if r.probes_dummy != nil {
		sdl.ReleaseGPUBuffer(r.gpu, r.probes_dummy)
	}
	delete(r.light_scratch)
	r^ = {}
}

// Publishes a scene revision. The frontend key skips unchanged scenes;
// content revisions then distinguish geometry, textures, lights, labels and
// environment updates when a fresh USD import owns new CPU allocations.
renderer_set_scene :: proc(r: ^Renderer, scene: ^lc.Scene, key: u64) -> (ok: bool) {
	defer if !ok { r.has_scene = false }
	if r.has_scene && r.scene.key == key {
		return true
	}
	geometry_key := geometry_revision(scene)
	instance_key := instance_revision(scene)
	texture_key := scene_texture_revision(scene)
	reupload := !r.has_scene || geometry_key != r.geometry_key
	// Moving nodes rewrites the instance buffer alone.
	if !reupload && instance_key != r.instance_key {
		if scene_update_instances(r.gpu, &r.scene, scene) {
			r.instance_uploads += 1
		} else {
			reupload = true
		}
	}
	if reupload {
		uploaded, ok := scene_upload(r.gpu, scene, key, &r.texture_cache)
		if !ok { return false }
		scene_destroy(r.gpu, &r.scene)
		r.scene = uploaded
		r.has_scene = true
		r.geometry_uploads += 1
	}
	r.geometry_key = geometry_key
	r.instance_key = instance_key
	if !reupload && texture_key != r.texture_key {
		if !scene_bind_textures(r.gpu, &r.scene, scene, &r.texture_cache) {
			r.has_scene = false
			return false
		}
	}
	texture_cache_prune(r.gpu, &r.texture_cache)
	r.texture_key = texture_key
	if !renderer_refresh_scene(r, scene) { return false }

	labels_build_semantic_table(&r.labels, scene)
	labels_key := xxhash.XXH64(slice.to_bytes(r.labels.semantic_scratch[:]))
	if !r.labels_ready || labels_key != r.labels_key {
		if !labels_upload_semantic_table(r.gpu, &r.labels) { return false }
		r.labels_key = labels_key
		r.labels_ready = true
	}
	env_key := environment_revision(scene.environment)
	if !r.environment_ready || env_key != r.environment_key {
		if !build_environment(r, scene) { return false }
		r.environment_key = env_key
		r.environment_ready = true
		r.environment_builds += 1
	}
	r.scene.key = key
	probes_check_scene(r, scene)
	return true
}

// Uploads the scene's environment and derives the two prefiltered maps.
//
// `hide_default_sky` is read from the core settings the same way the path
// tracer reads it, so a frontend that suppresses the procedural sky gets a
// black background in both modes rather than one each way.
@(private = "file")
build_environment :: proc(r: ^Renderer, scene: ^lc.Scene, hide_default_sky := false) -> bool {
	env_destroy(r.gpu, &r.env)

	pixels, width, height, ok := env_source_pixels(scene, hide_default_sky)
	if !ok {
		return true // no environment is a legitimate scene, not a failure
	}
	defer delete(pixels)

	r.env.rotation = f32(scene.environment.rotation)
	r.env.intensity = scene.environment.has_data ? f32(scene.environment.intensity) : 1.0

	r.env.source = env_upload_source(r.gpu, pixels, width, height)
	if r.env.source == nil {
		return false
	}

	r.env.sampler = sdl.CreateGPUSampler(
		r.gpu,
		sdl.GPUSamplerCreateInfo {
			min_filter = .LINEAR,
			mag_filter = .LINEAR,
			mipmap_mode = .LINEAR,
			// u wraps around the horizon; v must clamp or the poles bleed
			// across to the opposite one.
			address_mode_u = .REPEAT,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
			max_lod = 1000,
		},
	)
	if r.env.sampler == nil {
		return false
	}

	r.env.irradiance = make_render_target(r.gpu, ENV_FORMAT, IRRADIANCE_WIDTH, IRRADIANCE_HEIGHT, 1)
	r.env.specular = make_render_target(r.gpu, ENV_FORMAT, SPECULAR_WIDTH, SPECULAR_HEIGHT, SPECULAR_MIPS)
	if r.env.irradiance == nil || r.env.specular == nil {
		return false
	}

	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil {
		return false
	}

	// Diffuse convolution: an exact sum over a 64 x 32 grid of directions
	// (2048 taps per output texel), read from the source mip nearest that size.
	// The grid stays 64 x 32 for a smaller source too, such as a uniform sky's
	// few texels, which bilinear filtering stretches across it.
	irr_mip := max(0, i32(mip_levels(width, height)) - i32(mip_levels(IRRADIANCE_WIDTH, IRRADIANCE_HEIGHT)))
	prefilter_pass(
		cmd, r.irradiance_pipeline, r.env.irradiance, 0,
		r.env.source, r.env.sampler, {0, f32(irr_mip), IRRADIANCE_WIDTH, IRRADIANCE_HEIGHT},
	)

	// One specular mip per roughness step. Each tap picks its own source mip
	// from the lobe's width (filtered importance sampling), never finer than
	// the one matching the target's resolution.
	spec_floor := f32(max(0, i32(mip_levels(width, height)) - i32(mip_levels(SPECULAR_WIDTH, SPECULAR_HEIGHT))))
	for level in 0 ..< SPECULAR_MIPS {
		roughness := f32(level) / f32(SPECULAR_MIPS - 1)
		prefilter_pass(
			cmd, r.specular_pipeline, r.env.specular, u32(level),
			r.env.source, r.env.sampler, {roughness, 128, spec_floor, f32(width) * f32(height)},
		)
	}

	if !sdl.SubmitGPUCommandBuffer(cmd) {
		return false
	}

	r.env.has_env = true
	return true
}

// Runs one fullscreen prefilter pass into a specific mip of `dst`.
@(private = "file")
prefilter_pass :: proc(
	cmd: ^sdl.GPUCommandBuffer,
	pipeline: ^sdl.GPUGraphicsPipeline,
	dst: ^sdl.GPUTexture,
	mip: u32,
	src: ^sdl.GPUTexture,
	sampler: ^sdl.GPUSampler,
	params: [4]f32,
) {
	target := sdl.GPUColorTargetInfo {
		texture   = dst,
		mip_level = mip,
		load_op   = .DONT_CARE,
		store_op  = .STORE,
	}

	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, pipeline)

	binding := sdl.GPUTextureSamplerBinding{texture = src, sampler = sampler}
	sdl.BindGPUFragmentSamplers(pass, 0, &binding, 1)

	args := params
	sdl.PushGPUFragmentUniformData(cmd, 0, &args, size_of(args))
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
}

// The environment BRDF: independent of the scene, so built once.
@(private = "file")
build_brdf_lut :: proc(r: ^Renderer) -> bool {
	r.brdf_lut = make_render_target(r.gpu, LUT_FORMAT, BRDF_LUT_SIZE, BRDF_LUT_SIZE, 1)
	if r.brdf_lut == nil {
		return false
	}

	pipeline, ok := make_prefilter_pipeline(r.gpu, SHADER_BRDF_LUT_FS, LUT_FORMAT)
	if !ok {
		return false
	}
	defer sdl.ReleaseGPUGraphicsPipeline(r.gpu, pipeline)

	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil {
		return false
	}
	target := sdl.GPUColorTargetInfo{texture = r.brdf_lut, load_op = .DONT_CARE, store_op = .STORE}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, pipeline)
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
	return bool(sdl.SubmitGPUCommandBuffer(cmd))
}

@(private = "file")
make_render_target :: proc(
	gpu: ^sdl.GPUDevice, format: sdl.GPUTextureFormat, width, height: u32, levels: u32,
) -> ^sdl.GPUTexture {
	tex := sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = .D2,
			format = format,
			usage = {.COLOR_TARGET, .SAMPLER},
			width = width,
			height = height,
			layer_count_or_depth = 1,
			num_levels = levels,
			sample_count = ._1,
		},
	)
	if tex == nil {
		fmt.eprintln("realtime: render target creation failed:", sdl.GetError())
	}
	return tex
}

// Re-reads the scene's materials, lights and environment settings without
// touching geometry or textures.
//
// Material values reach the shader as per-draw uniforms rather than a buffer,
// so refreshing them is pure CPU work — no upload at all. Lights are one small
// storage buffer, and the environment's rotation and intensity are lookup-time
// scalars precisely so that changing them does not rebuild the prefiltered
// maps. Dragging a colour slider therefore costs nothing here, which is the
// same reasoning behind the path tracer's in-place material update.
renderer_refresh_scene :: proc(r: ^Renderer, scene: ^lc.Scene) -> bool {
	if !r.has_scene {
		return false
	}

	scene_refresh_materials(&r.scene, scene)
	lights_convert(scene.lights, &r.light_scratch)
	lights_key := xxhash.XXH64(slice.to_bytes(r.light_scratch[:]))
	if !r.lights_ready || lights_key != r.lights_key {
		if !lights_upload(r.gpu, &r.light_buffer, &r.light_capacity, r.light_scratch[:]) {
			r.lights_ready = false
			return false
		}
		r.light_count = u32(len(r.light_scratch))
		r.lights_key = lights_key
		r.lights_ready = true
	}

	if r.env.has_env && scene.environment.has_data {
		r.env.rotation = f32(scene.environment.rotation)
		r.env.intensity = f32(scene.environment.intensity)
	}
	return true
}

// Draws one frame and returns the texture holding it, or nil on failure.
//
// Called only when something changed — the app's redraw model is idle-driven
// and a realtime mode that repainted unconditionally would undo that.
renderer_render :: proc(
	r: ^Renderer,
	cam: lc.Camera,
	width, height: i32,
	view: Debug_View,
) -> ^sdl.GPUTexture {
	if width <= 0 || height <= 0 || !r.has_scene {
		return nil
	}
	if !ensure_targets(r, width, height) {
		return nil
	}

	// Shadow slots go to the lights that matter most to this view. The light
	// buffer always matches `light_scratch` here (refresh keeps it so), so it
	// needs uploading only when a slot moved.
	if shadow_slots_assign(r.light_scratch[:], camera_frame(cam)) {
		if !lights_upload(r.gpu, &r.light_buffer, &r.light_capacity, r.light_scratch[:]) {
			fmt.eprintln("realtime: light upload failed:", sdl.GetError())
			r.lights_ready = false
			return nil
		}
		r.lights_key = xxhash.XXH64(slice.to_bytes(r.light_scratch[:]))
	}

	cmd := sdl.AcquireGPUCommandBuffer(r.gpu)
	if cmd == nil {
		fmt.eprintln("realtime: AcquireGPUCommandBuffer failed:", sdl.GetError())
		return nil
	}
	// With profiling on, every `split` submits what has been recorded since the
	// last one, waits for it, and charges the time to that pass.
	profile_begin(r)
	split :: proc(r: ^Renderer, cmd: ^^sdl.GPUCommandBuffer, pass: Pass) -> bool {
		return !r.profile || profile_split(r, cmd, pass)
	}

	// Froxel assignment follows the camera, so it is rebuilt every frame. Its
	// copy pass leads the frame's command buffer, ahead of the lighting pass
	// that reads it.
	clusters_build(&r.clusters, r.light_scratch[:], camera_frame(cam))
	if !clusters_upload(r.gpu, cmd, &r.clusters) {
		fmt.eprintln("realtime: cluster upload failed:", sdl.GetError())
		_ = sdl.CancelGPUCommandBuffer(cmd)
		return nil
	}
	if !split(r, &cmd, .Clusters) { return nil }

	// The reflection view is the shaded frame with one pass's output swapped.
	shaded := view == .Shaded || view == .Reflection
	ssr := r.settings.ssr || view == .Reflection
	taa := shaded && r.settings.taa
	taa_begin_frame(r, cam, view, taa)
	if !glass_begin_frame(r, cmd) {
		_ = sdl.CancelGPUCommandBuffer(cmd)
		return nil
	}

	cascades: Cascades
	if dir, has_sun := sun_light_dir(r); has_sun && shaded {
		cascades = shadow_build_cascades(
			camera_frame(cam), dir, r.scene.bounds_min, r.scene.bounds_max,
		)
		if cascades.enabled {
			draw_shadows(r, cmd, cascades)
		}
	}
	r.spots = {}
	r.omnis = {}
	if shaded {
		r.spots = spot_shadows_build(r.light_scratch[:], r.scene.bounds_min, r.scene.bounds_max)
		r.omnis = omni_shadows_build(r.light_scratch[:], r.scene.bounds_min, r.scene.bounds_max)
		// One light position per face, for the glass maps.
		omni_eyes: [MAX_OMNI_SHADOWS * 6][3]f32
		for slot in 0 ..< r.omnis.count {
			for face in 0 ..< 6 { omni_eyes[slot * 6 + face] = r.omnis.eye[slot] }
		}
		ok := true
		if r.spots.count > 0 {
			ok = draw_shadow_layers(r, cmd, &r.spot_layers, r.spots.view_proj[:r.spots.count], r.spots.eye[:r.spots.count],
				SPOT_SHADOW_RESOLUTION, MAX_SPOT_SHADOWS, "Spot shadows")
		}
		if ok && r.omnis.count > 0 {
			ok = draw_shadow_layers(r, cmd, &r.omni_layers, r.omnis.view_proj[:r.omnis.count * 6], omni_eyes[:r.omnis.count * 6],
				OMNI_SHADOW_RESOLUTION, MAX_OMNI_SHADOWS * 6, "Point shadows")
		}
		if !ok {
			_ = sdl.CancelGPUCommandBuffer(cmd)
			return nil
		}
	}
	if !split(r, &cmd, .Shadows) { return nil }

	if view == .Instance || view == .Semantic {
		if !labels_ensure_targets(r.gpu, &r.labels, width, height) {
			_ = sdl.CancelGPUCommandBuffer(cmd)
			return nil
		}
		labels_draw(&r.labels, cmd, &r.scene, cam)
		if !split(r, &cmd, .Labels) { return nil }
	}
	if view == .Motion {
		if !motion_ensure_targets(r.gpu, &r.motion, width, height) {
			_ = sdl.CancelGPUCommandBuffer(cmd)
			return nil
		}
		prev_cam, moved := r.motion_prev_camera.?
		motion_draw(&r.motion, cmd, &r.scene, cam, moved ? prev_cam : cam)
		if !split(r, &cmd, .Labels) { return nil }
	}

	draw_gbuffer(r, cmd, cam)
	if !split(r, &cmd, .GBuffer) { return nil }
	if shaded {
		if r.settings.ao {
			draw_ao(r, cmd, cam)
			if !split(r, &cmd, .AO) { return nil }
		}
		if ssr {
			draw_lighting(r, cmd, cam, cascades, r.refraction)
			// The blur pyramid glossy reflections read.
			sdl.GenerateMipmapsForGPUTexture(cmd, r.refraction)
			if !split(r, &cmd, .Lighting) { return nil }
			draw_hiz(r, cmd)
			if !split(r, &cmd, .HiZ) { return nil }
			draw_ssr(r, cmd, cam, view == .Reflection)
			if !split(r, &cmd, .SSR) { return nil }
		} else {
			draw_lighting(r, cmd, cam, cascades, r.hdr_color)
			if !split(r, &cmd, .Lighting) { return nil }
		}
		// Glass would cover the one thing the reflection view is for, and fog
		// would dim it.
		if view != .Reflection {
			draw_forward(r, cmd, cam, cascades)
			if !split(r, &cmd, .Forward) { return nil }
			if r.settings.fog && r.settings.fog_density > 0 {
				draw_fog(r, cmd, cam, cascades)
				if !split(r, &cmd, .Fog) { return nil }
			}
		}
		source := r.hdr_color
		if taa {
			source = draw_taa(r, cmd, cam)
			if !split(r, &cmd, .TAA) { return nil }
		}
		fxaa := r.settings.fxaa && !taa
		draw_display(r, cmd, source, fxaa)
		if !split(r, &cmd, .Display) { return nil }
		if fxaa {
			draw_fxaa(r, cmd)
			if !split(r, &cmd, .FXAA) { return nil }
		}
	} else {
		if view == .AO {
			draw_ao(r, cmd, cam)
			if !split(r, &cmd, .AO) { return nil }
		}
		draw_debug(r, cmd, cam, view)
		if !split(r, &cmd, .Debug) { return nil }
	}

	if !sdl.SubmitGPUCommandBuffer(cmd) {
		fmt.eprintln("realtime: SubmitGPUCommandBuffer failed:", sdl.GetError())
		return nil
	}
	profile_end(r)
	return r.color
}

// Renders a still: with temporal antialiasing on, the history is cleared and
// TAA_SETTLE_FRAMES frames are drawn from the unmoving camera, so the result
// is the converged image and depends on nothing drawn before. Otherwise this
// is one ordinary frame.
renderer_render_still :: proc(
	r: ^Renderer,
	cam: lc.Camera,
	width, height: i32,
	view: Debug_View,
) -> ^sdl.GPUTexture {
	if !(r.settings.taa && (view == .Shaded || view == .Reflection)) {
		return renderer_render(r, cam, width, height, view)
	}
	renderer_reset_history(r)
	tex: ^sdl.GPUTexture
	for _ in 0 ..< TAA_SETTLE_FRAMES {
		tex = renderer_render(r, cam, width, height, view)
		if tex == nil { return nil }
	}
	return tex
}

// Forgets the temporal history, so the next frame starts the jitter sequence
// over and blends with nothing.
renderer_reset_history :: proc(r: ^Renderer) {
	r.taa_valid = false
}

// Whether the last frame was temporally antialiased, so a caller that wants a
// settled image should keep drawing for TAA_SETTLE_FRAMES after a change.
renderer_taa_active :: proc(r: ^Renderer) -> bool {
	return r.taa_on
}

// Picks this frame's jitter and decides whether the history still applies.
@(private = "file")
taa_begin_frame :: proc(r: ^Renderer, cam: lc.Camera, view: Debug_View, taa: bool) {
	r.taa_on = taa
	if !taa {
		r.taa_jitter = {}
		r.taa_valid = false
		return
	}
	// History from another view is a different image.
	if r.taa_valid && r.taa_view != view {
		r.taa_valid = false
	}
	if !r.taa_valid {
		r.taa_frame = 0
		r.taa_still = 0
	}
	r.taa_view = view

	f := camera_frame(cam)
	if r.taa_valid && camera_view_proj(f) == r.taa_prev_view_proj {
		r.taa_still += 1
	} else {
		r.taa_still = 0
	}

	// Halton (2, 3), skipping index 0, whose point is the corner (0, 0).
	i := r.taa_frame % TAA_SAMPLES + 1
	r.taa_frame += 1
	jitter := [2]f32{halton(i, 2) - 0.5, halton(i, 3) - 0.5}
	// Pixels to NDC: the screen spans 2 units across `width` pixels.
	r.taa_jitter = {2 * jitter.x / f32(r.width), 2 * jitter.y / f32(r.height)}
}

// The radical inverse of `index` in `base`: the Halton sequence's coordinate.
halton :: proc(index, base: u32) -> f32 {
	f := f32(1)
	result := f32(0)
	for i := index; i > 0; i /= base {
		f /= f32(base)
		result += f * f32(i % base)
	}
	return result
}

// TAA shader inputs; mirrors TaaUniforms in shaders/taa_fs.slang.
@(private = "file")
Taa_Uniforms :: struct {
	reproject: matrix[4, 4]f32,
	size:      [4]f32,
	params:    [4]f32,
}

// Blends `hdr_color` into the history and returns the target written, which
// is what the display pass then reads.
@(private = "file")
draw_taa :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera) -> ^sdl.GPUTexture {
	sdl.PushGPUDebugGroup(cmd, "Temporal antialiasing")
	defer sdl.PopGPUDebugGroup(cmd)

	f := camera_frame(cam)
	view_proj := camera_view_proj(f, r.taa_jitter)
	read := r.taa_write
	write := 1 - r.taa_write

	// Still, the history keeps averaging toward the full Halton cycle; moving,
	// it holds a fixed share so what was uncovered a moment ago fades in fast.
	alpha := f32(TAA_ALPHA_MOVING)
	if r.taa_still > 0 {
		alpha = max(1 / f32(r.taa_still + 10), TAA_ALPHA_STILL)
	}
	uniforms := Taa_Uniforms {
		reproject = r.taa_prev_view_proj * linalg.inverse(view_proj),
		size      = {f32(r.width), f32(r.height), 1 / f32(r.width), 1 / f32(r.height)},
		params    = {alpha, r.taa_valid ? 1 : 0, 0, 0},
	}

	target := sdl.GPUColorTargetInfo{texture = r.taa_history[write], load_op = .DONT_CARE, store_op = .STORE}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.taa_pipeline)
	samplers := [3]sdl.GPUTextureSamplerBinding {
		{texture = r.hdr_color, sampler = r.target_sampler},
		{texture = r.taa_history[read], sampler = r.linear_sampler},
		{texture = r.depth, sampler = r.target_sampler},
	}
	sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
	sdl.PushGPUFragmentUniformData(cmd, 0, &uniforms, size_of(uniforms))
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)

	r.taa_write = write
	r.taa_valid = true
	r.taa_prev_view_proj = camera_view_proj(f)
	return r.taa_history[write]
}

// ── passes ───────────────────────────────────────────────────────────────────

@(private = "file")
draw_gbuffer :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera) {
	sdl.PushGPUDebugGroup(cmd, "G-buffer")
	defer sdl.PopGPUDebugGroup(cmd)
	targets := [4]sdl.GPUColorTargetInfo {
		{texture = r.albedo, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
		{texture = r.normal, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
		{texture = r.surface, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
		{texture = r.emission, clear_color = {0, 0, 0, 0}, load_op = .CLEAR, store_op = .STORE},
	}
	depth := sdl.GPUDepthStencilTargetInfo {
		texture     = r.depth,
		clear_depth = DEPTH_CLEAR,
		load_op     = .CLEAR,
		store_op    = .STORE,
		cycle       = true,
		stencil_load_op = .DONT_CARE,
		stencil_store_op = .DONT_CARE,
	}

	pass := sdl.BeginGPURenderPass(cmd, raw_data(&targets), len(targets), &depth)
	sdl.BindGPUGraphicsPipeline(pass, r.gbuffer_pipeline)

	uniforms := camera_uniforms(cam, r.taa_jitter)
	sdl.PushGPUVertexUniformData(cmd, 0, &uniforms, size_of(uniforms))

	scene_bind_vertex_buffers(pass, &r.scene)
	scene_mark_visible(&r.scene, uniforms.view_proj)

	for b in r.scene.batches {
		// Transmissive surfaces cannot be expressed in a G-buffer; draw_forward
		// draws them after the lighting pass.
		if b.transparent {
			continue
		}
		start := int(b.first_instance)
		end := start + int(b.instance_count)
		bound := false
		for cursor := start; cursor < end; {
			first, count := instance_run(r.scene.visible[:], cursor, end)
			if count == 0 { break }
			cursor = first + count
			// Material state only for a batch that draws something.
			if !bound {
				samplers := [4]sdl.GPUTextureSamplerBinding {
					{texture = b.albedo, sampler = r.scene.sampler},
					{texture = b.mr, sampler = r.scene.sampler},
					{texture = b.normal, sampler = r.scene.sampler},
					{texture = b.emissive, sampler = r.scene.sampler},
				}
				sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
				mat := b.material
				sdl.PushGPUFragmentUniformData(cmd, 0, &mat, size_of(mat))
				bound = true
			}
			sdl.DrawGPUIndexedPrimitives(pass, b.index_count, u32(count), b.first_index, b.base_vertex, u32(first))
		}
	}

	sdl.EndGPURenderPass(pass)
}

@(private = "file")
draw_debug :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera, view: Debug_View) {
	target := sdl.GPUColorTargetInfo {
		texture     = r.color,
		clear_color = {0.05, 0.05, 0.06, 1.0},
		load_op     = .CLEAR,
		store_op    = .STORE,
	}

	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.debug_pipeline)

	samplers := [6]sdl.GPUTextureSamplerBinding {
		{texture = r.albedo, sampler = r.target_sampler},
		{texture = r.normal, sampler = r.target_sampler},
		{texture = r.surface, sampler = r.target_sampler},
		{texture = r.emission, sampler = r.target_sampler},
		{texture = r.depth, sampler = r.target_sampler},
		// Drawn this frame only for the AO view.
		{texture = r.ao, sampler = r.target_sampler},
	}
	sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))

	// The id targets are integer textures read with Load, not Sample, so they
	// carry no sampler and belong to SDL's storage-texture list rather than
	// the sampler list. Counting them as samplers binds sampler slots the
	// shader never declared, which the Metal backend does not survive.
	id_textures := [3]^sdl.GPUTexture {
		label_or(r, r.labels.instance),
		label_or(r, r.labels.semantic),
		r.motion.velocity if r.motion.velocity != nil else r.motion.fallback,
	}
	sdl.BindGPUFragmentStorageTextures(pass, 0, raw_data(&id_textures), len(id_textures))

	// The depth view needs the same near/far the projection used, or it
	// linearizes against the wrong range and reads as flat white. The shader
	// assumes camera_projection's reversed-Z.
	f := camera_frame(cam)
	// Scale the depth view against how far the SCENE reaches, not the focus
	// distance: anything extending past a couple of focus distances -- a ground
	// plane, most of an interior -- saturates to flat white otherwise.
	display_far := scene_view_far(f, r.scene.bounds_min, r.scene.bounds_max)
	params := [4]f32{f32(i32(view)), f.focus * NEAR_SCALE, f.focus * FAR_SCALE, display_far}
	sdl.PushGPUFragmentUniformData(cmd, 0, &params, size_of(params))

	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
}

@(private = "file")
draw_lighting :: proc(
	r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera, cascades: Cascades,
	output: ^sdl.GPUTexture,
) {
	sdl.PushGPUDebugGroup(cmd, "Deferred HDR lighting")
	defer sdl.PopGPUDebugGroup(cmd)
	target := sdl.GPUColorTargetInfo {
		texture     = output,
		clear_color = {0, 0, 0, 1},
		load_op     = .CLEAR,
		store_op    = .STORE,
	}

	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.lighting_pipeline)

	samplers := [16]sdl.GPUTextureSamplerBinding {
		{texture = r.albedo, sampler = r.target_sampler},
		{texture = r.normal, sampler = r.target_sampler},
		{texture = r.surface, sampler = r.target_sampler},
		{texture = r.emission, sampler = r.target_sampler},
		{texture = r.depth, sampler = r.target_sampler},
		{texture = r.shadow_map, sampler = r.shadow_sampler},
		// A scene with no environment still has to bind something; the shader
		// gates on `has_env` rather than on the binding.
		{texture = env_or(r, r.env.source), sampler = env_sampler(r)},
		{texture = env_or(r, r.env.irradiance), sampler = env_sampler(r)},
		{texture = env_or(r, r.env.specular), sampler = env_sampler(r)},
		{texture = r.brdf_lut, sampler = r.target_sampler},
		// Bound even with AO off; the shader gates on forward_params.y.
		{texture = r.ao, sampler = r.target_sampler},
		{texture = layers_or(r, r.spot_layers), sampler = r.shadow_sampler},
		{texture = layers_or(r, r.omni_layers), sampler = r.shadow_sampler},
		{texture = r.ltc.transform, sampler = r.linear_sampler},
		{texture = r.ltc.magnitude, sampler = r.linear_sampler},
		// The omni map read as plain depths, for soft shadows' blocker search.
		{texture = layers_or(r, r.omni_layers), sampler = r.target_sampler},
	}
	sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
	bind_glass_maps(r, pass)

	probe_irradiance, probe_depth, _, probe_radiance, probe_info := probes_buffers(r)
	buffers := [7]^sdl.GPUBuffer{r.light_buffer, r.clusters.range_buffer, r.clusters.index_buffer, probe_irradiance, probe_depth, probe_radiance, probe_info}
	sdl.BindGPUFragmentStorageBuffers(pass, 0, raw_data(&buffers), len(buffers))

	uniforms := frame_uniforms(r, cam, cascades)
	sdl.PushGPUFragmentUniformData(cmd, 0, &uniforms, size_of(uniforms))

	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
}

// Everything the lighting passes share: the camera, the shadow-casting light's
// index, the environment, and the froxel grid. Built once per pass so the
// deferred and forward passes cannot disagree about a frame.
@(private = "file")
frame_uniforms :: proc(r: ^Renderer, cam: lc.Camera, cascades: Cascades) -> Lighting_Uniforms {
	uniforms := lighting_uniforms(cam, r.light_count, cascades, r.taa_jitter)
	uniforms.eye.w = -1
	for light, index in r.light_scratch {
		if i32(light.params.x) == i32(Light_Kind_GPU.Distant) {
			uniforms.eye.w = f32(index)
			break
		}
	}
	uniforms.env = {
		r.env.has_env ? 1 : 0,
		r.env.rotation,
		r.env.intensity,
		f32(SPECULAR_MIPS),
	}
	// How far a refracted ray travels inside a surface before it is looked up
	// behind it. Scaled to the scene so glass bends by a plausible amount at
	// any scale; a real thickness would need the geometry's back face.
	extent := [3]f32{
		r.scene.bounds_max.x - r.scene.bounds_min.x,
		r.scene.bounds_max.y - r.scene.bounds_min.y,
		r.scene.bounds_max.z - r.scene.bounds_min.z,
	}
	diagonal := max(extent.x, max(extent.y, extent.z))
	// The TAA frame turns the soft shadows' tap spiral (pcss_spin), so TAA
	// averages their grain away.
	taa_index := r.taa_on && r.taa_frame > 0 ? f32((r.taa_frame - 1) % TAA_SAMPLES) : 0
	uniforms.forward_params = {diagonal * 0.02, r.settings.ao ? 1 : 0, taa_index, 0}
	for i in 0 ..< r.spots.count {
		uniforms.spot_view_proj[i] = r.spots.view_proj[i]
		uniforms.spot_texel[i / 4][i % 4] = r.spots.texel_scale[i]
	}
	for i in 0 ..< r.omnis.count * 6 {
		uniforms.omni_view_proj[i] = r.omnis.view_proj[i]
	}
	uniforms.omni_range = r.omnis.range
	uniforms.probe_min, uniforms.probe_cell, uniforms.probe_dims = probes_uniforms(r)
	return uniforms
}

// One transmissive draw: a batch and one of its instances, kept apart so they
// can be sorted against each other.
@(private = "file")
Transparent_Draw :: struct {
	// Distance from the eye, for back-to-front ordering.
	depth:    f32,
	batch:    i32,
	instance: u32,
}

// Draws transmissive surfaces over the lit image.
//
// Back to front, one draw per instance: transparency has no depth test to sort
// it, so the order the draws are issued IS the composite order. Depth is
// tested against the opaque scene but not written, so two glass surfaces cannot
// occlude each other -- the far one is simply drawn first.
@(private = "file")
draw_forward :: proc(
	r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera, cascades: Cascades,
) {
	f := camera_frame(cam)
	uniforms := frame_uniforms(r, cam, cascades)

	// Gather the visible transmissive instances.
	scene_mark_visible(&r.scene, uniforms.view_proj)
	clear(&r.transparent_scratch)
	for b, bi in r.scene.batches {
		if !b.transparent {
			continue
		}
		start := int(b.first_instance)
		for inst in start ..< start + int(b.instance_count) {
			if !r.scene.visible[inst] {
				continue
			}
			bounds := r.scene.instance_bounds[inst]
			centre := [3]f32{
				(bounds.lo.x + bounds.hi.x) * 0.5,
				(bounds.lo.y + bounds.hi.y) * 0.5,
				(bounds.lo.z + bounds.hi.z) * 0.5,
			}
			d := [3]f32{centre.x - f.eye.x, centre.y - f.eye.y, centre.z - f.eye.z}
			append(&r.transparent_scratch, Transparent_Draw {
				depth = d.x * d.x + d.y * d.y + d.z * d.z,
				batch = i32(bi),
				instance = u32(inst),
			})
		}
	}
	if len(r.transparent_scratch) == 0 {
		return
	}
	slice.sort_by(r.transparent_scratch[:], proc(a, b: Transparent_Draw) -> bool {
		return a.depth > b.depth // farthest first
	})

	sdl.PushGPUDebugGroup(cmd, "Forward transparency")
	defer sdl.PopGPUDebugGroup(cmd)

	// Each surface refracts a copy of what is already drawn behind it, so the
	// copy is refreshed as glass is laid down, farthest first: an object's
	// back faces over the scene, a copy, then its front faces over that. A
	// pane behind a pane is seen through it, and a glass ball's front sees its
	// own back. (A single copy before any glass showed only the opaque scene
	// through every pane, and a ball's two faces, drawn in one pass without
	// depth writes, came out in whatever order its triangles did.)
	//
	// A copy per face per object is two full-frame blits each; past
	// GLASS_LAYERED_MAX objects, all back faces go down over one copy and all
	// front faces over a second.
	draws := r.transparent_scratch[:]
	if len(draws) <= GLASS_LAYERED_MAX {
		for &draw in draws {
			one := slice.from_ptr(&draw, 1)
			forward_layer(r, cmd, cam, &uniforms, one, GLASS_BACK_FACES)
			forward_layer(r, cmd, cam, &uniforms, one, GLASS_FRONT_FACES)
		}
	} else {
		forward_layer(r, cmd, cam, &uniforms, draws, GLASS_BACK_FACES)
		forward_layer(r, cmd, cam, &uniforms, draws, GLASS_FRONT_FACES)
	}
}

// Which faces a forward layer draws, in forward_params.w: forward_fs discards
// the others.
GLASS_BACK_FACES :: 1
GLASS_FRONT_FACES :: 2
// Glass objects layered one at a time; see draw_forward.
GLASS_LAYERED_MAX :: 16

// Copies what is drawn so far into `refraction`, then draws one face of each
// of `draws` over it.
@(private = "file")
forward_layer :: proc(
	r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera,
	uniforms: ^Lighting_Uniforms, draws: []Transparent_Draw, faces: f32,
) {
	blit := sdl.GPUBlitInfo {
		source = {texture = r.hdr_color, w = u32(r.width), h = u32(r.height)},
		destination = {texture = r.refraction, w = u32(r.width), h = u32(r.height)},
		load_op = .DONT_CARE,
		filter = .NEAREST,
	}
	sdl.BlitGPUTexture(cmd, blit)

	target := sdl.GPUColorTargetInfo {
		texture = r.hdr_color,
		load_op = .LOAD,
		store_op = .STORE,
	}
	depth := sdl.GPUDepthStencilTargetInfo {
		texture = r.depth,
		load_op = .LOAD,
		store_op = .STORE,
		stencil_load_op = .DONT_CARE,
		stencil_store_op = .DONT_CARE,
	}

	pass := sdl.BeginGPURenderPass(cmd, &target, 1, &depth)
	sdl.BindGPUGraphicsPipeline(pass, r.forward_pipeline)

	camera_u := camera_uniforms(cam, r.taa_jitter)
	sdl.PushGPUVertexUniformData(cmd, 0, &camera_u, size_of(camera_u))
	uniforms.forward_params.w = faces
	sdl.PushGPUFragmentUniformData(cmd, 1, uniforms, size_of(uniforms^))

	scene_bind_vertex_buffers(pass, &r.scene)
	probe_irradiance, probe_depth, _, probe_radiance, probe_info := probes_buffers(r)
	buffers := [7]^sdl.GPUBuffer{r.light_buffer, r.clusters.range_buffer, r.clusters.index_buffer, probe_irradiance, probe_depth, probe_radiance, probe_info}
	sdl.BindGPUFragmentStorageBuffers(pass, 0, raw_data(&buffers), len(buffers))

	// Slots 4-13 are the same for every draw; bind them once. No env_source:
	// the forward pass never draws the background, so the shader does not
	// declare it (see forward_fs.slang).
	shared := [10]sdl.GPUTextureSamplerBinding {
		{texture = r.shadow_map, sampler = r.shadow_sampler},
		{texture = env_or(r, r.env.irradiance), sampler = env_sampler(r)},
		{texture = env_or(r, r.env.specular), sampler = env_sampler(r)},
		{texture = r.brdf_lut, sampler = r.target_sampler},
		{texture = r.refraction, sampler = r.target_sampler},
		{texture = layers_or(r, r.spot_layers), sampler = r.shadow_sampler},
		{texture = layers_or(r, r.omni_layers), sampler = r.shadow_sampler},
		{texture = r.ltc.transform, sampler = r.linear_sampler},
		{texture = r.ltc.magnitude, sampler = r.linear_sampler},
		{texture = layers_or(r, r.omni_layers), sampler = r.target_sampler},
	}
	sdl.BindGPUFragmentSamplers(pass, 4, raw_data(&shared), len(shared))
	bind_glass_maps(r, pass)

	bound := i32(-1)
	for draw in draws {
		b := r.scene.batches[draw.batch]
		// Sorted by depth, not material, but neighbours often share a batch --
		// the instances of one scattered glass prop -- and then keep its state.
		if draw.batch != bound {
			samplers := [4]sdl.GPUTextureSamplerBinding {
				{texture = b.albedo, sampler = r.scene.sampler},
				{texture = b.mr, sampler = r.scene.sampler},
				{texture = b.normal, sampler = r.scene.sampler},
				{texture = b.emissive, sampler = r.scene.sampler},
			}
			sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
			mat := b.material
			sdl.PushGPUFragmentUniformData(cmd, 0, &mat, size_of(mat))
			bound = draw.batch
		}
		sdl.DrawGPUIndexedPrimitives(pass, b.index_count, 1, b.first_index, b.base_vertex, draw.instance)
	}

	sdl.EndGPURenderPass(pass)
}

// Renders the scene's depth from the light, once per cascade. Reuses the
// G-buffer's vertex buffer and batch list unchanged: a shadow caster is any
// geometry, and material has no bearing on depth.
@(private = "file")
draw_shadows :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cascades: Cascades) {
	sdl.PushGPUDebugGroup(cmd, "Sun cascades")
	defer sdl.PopGPUDebugGroup(cmd)
	// Any upload may have moved or replaced a caster.
	stamp := [2]u64{r.geometry_uploads, r.instance_uploads}
	if stamp != r.shadow_stamp {
		r.shadow_valid = {}
		r.shadow_stamp = stamp
	}
	for cascade, i in cascades.slices {
		if r.shadow_valid[i] && r.shadow_drawn[i] == cascade.view_proj {
			continue
		}
		r.shadow_drawn[i] = cascade.view_proj
		r.shadow_valid[i] = true
		r.shadow_layer_draws += 1
		depth := sdl.GPUDepthStencilTargetInfo {
			texture     = r.shadow_map,
			clear_depth = 1.0,
			load_op     = .CLEAR,
			store_op    = .STORE,
			layer       = u8(i),
		}

		pass := sdl.BeginGPURenderPass(cmd, nil, 0, &depth)
		sdl.BindGPUGraphicsPipeline(pass, r.shadow_pipeline)

		view_proj := cascade.view_proj
		sdl.PushGPUVertexUniformData(cmd, 0, &view_proj, size_of(view_proj))
		scene_bind_vertex_buffers(pass, &r.scene)
		// Depth does not care which material a triangle has, only whether it
		// is opaque: glass draws its own map.
		scene_mark_visible(&r.scene, view_proj)
		scene_draw_ignoring_material(pass, &r.scene, true)
		sdl.EndGPURenderPass(pass)
		if r.has_glass {
			draw_glass_layer(r, cmd, r.sun_glass, u32(i), view_proj, {0, 0, 0, 0})
		}
	}
}

// A depth array of light views -- spot maps or omni faces -- and what each
// layer currently holds, so an unchanged layer is never redrawn.
MAX_SHADOW_LAYERS :: max(MAX_SPOT_SHADOWS, MAX_OMNI_SHADOWS * 6)

Shadow_Layers :: struct {
	texture: ^sdl.GPUTexture, // created on first use
	// What glass lets through, layer for layer at half the resolution: rgb
	// tint, alpha the nearest glass's distance. Only while the scene has glass.
	glass:   ^sdl.GPUTexture,
	drawn:   [MAX_SHADOW_LAYERS]matrix[4, 4]f32,
	valid:   [MAX_SHADOW_LAYERS]bool,
	stamp:   [2]u64, // geometry_uploads, instance_uploads when drawn
	draws:   u64,    // layers drawn so far, for tests and profiling
}

// Renders each light view into its layer, skipping layers whose view and
// casters are unchanged since they were drawn.
@(private = "file")
draw_shadow_layers :: proc(
	r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, layers: ^Shadow_Layers,
	views: []matrix[4, 4]f32, eyes: [][3]f32, resolution, capacity: u32, name: cstring,
) -> bool {
	if r.has_glass && layers.glass == nil {
		layers.glass = make_glass_maps(r.gpu, resolution, capacity)
		if layers.glass == nil { return false }
		layers.valid = {}
	}
	if layers.texture == nil {
		layers.texture = sdl.CreateGPUTexture(
			r.gpu,
			sdl.GPUTextureCreateInfo {
				type = .D2_ARRAY,
				format = DEPTH_FORMAT,
				usage = {.DEPTH_STENCIL_TARGET, .SAMPLER},
				width = resolution,
				height = resolution,
				layer_count_or_depth = capacity,
				num_levels = 1,
				sample_count = ._1,
			},
		)
		if layers.texture == nil {
			fmt.eprintln("realtime:", name, "map creation failed:", sdl.GetError())
			return false
		}
		layers.valid = {}
	}
	sdl.PushGPUDebugGroup(cmd, name)
	defer sdl.PopGPUDebugGroup(cmd)
	stamp := [2]u64{r.geometry_uploads, r.instance_uploads}
	if stamp != layers.stamp {
		layers.valid = {}
		layers.stamp = stamp
	}
	for view, i in views {
		view_proj := view
		if layers.valid[i] && layers.drawn[i] == view_proj {
			continue
		}
		layers.drawn[i] = view_proj
		layers.valid[i] = true
		layers.draws += 1
		depth := sdl.GPUDepthStencilTargetInfo {
			texture     = layers.texture,
			clear_depth = 1.0,
			load_op     = .CLEAR,
			store_op    = .STORE,
			layer       = u8(i),
		}
		pass := sdl.BeginGPURenderPass(cmd, nil, 0, &depth)
		sdl.BindGPUGraphicsPipeline(pass, r.shadow_pipeline)
		sdl.PushGPUVertexUniformData(cmd, 0, &view_proj, size_of(view_proj))
		scene_bind_vertex_buffers(pass, &r.scene)
		scene_mark_visible(&r.scene, view_proj)
		scene_draw_ignoring_material(pass, &r.scene, true)
		sdl.EndGPURenderPass(pass)
		if r.has_glass {
			e := eyes[i]
			draw_glass_layer(r, cmd, layers.glass, u32(i), view_proj, {e.x, e.y, e.z, 1})
		}
	}
	return true
}

// A light-view array, or the sun's as a stand-in: every declared slot must be
// bound, and the shaders sample these only for lights that own a slot.
@(private = "file")
layers_or :: proc(r: ^Renderer, layers: Shadow_Layers) -> ^sdl.GPUTexture {
	return layers.texture != nil ? layers.texture : r.shadow_map
}

// The sun, if the uploaded scene has one.
@(private = "file")
sun_light_dir :: proc(r: ^Renderer) -> ([3]f32, bool) {
	for l in r.light_scratch {
		if i32(l.params.x) == i32(Light_Kind_GPU.Distant) {
			return {l.direction.x, l.direction.y, l.direction.z}, true
		}
	}
	return {}, false
}

// ── pipelines ────────────────────────────────────────────────────────────────

@(private = "file")
make_gbuffer_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_GBUFFER_VS, "vertexMain", .VERTEX, {uniform_buffers = 1})
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(
		gpu, SHADER_GBUFFER_FS, "fragmentMain", .FRAGMENT,
		{samplers = 4, uniform_buffers = 1},
	)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [4]sdl.GPUColorTargetDescription {
		{format = ALBEDO_FORMAT},
		{format = NORMAL_FORMAT},
		{format = SURFACE_FORMAT},
		{format = EMISSION_FORMAT},
	}

	vertex_input := vertex_input_state()

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			vertex_input_state = vertex_input,
			primitive_type = .TRIANGLELIST,
			// No culling. Lumbre's scenes come from USD and OBJ content with
			// no reliable winding convention, and the path tracer shades both
			// faces — culling here would make the two modes disagree about
			// which surfaces exist, which is worse than drawing backfaces.
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE, front_face = .COUNTER_CLOCKWISE},
			depth_stencil_state = {
				compare_op = DEPTH_COMPARE,
				enable_depth_test = true,
				enable_depth_write = true,
			},
			target_info = {
				num_color_targets = len(targets),
				color_target_descriptions = raw_data(&targets),
				depth_stencil_format = DEPTH_FORMAT,
				has_depth_stencil_target = true,
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: G-buffer pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

// A fullscreen pass with one input texture and one uniform block: the shape
// every prefilter step has.
@(private = "file")
make_prefilter_pipeline :: proc(
	gpu: ^sdl.GPUDevice, blob: Shader_Blob, format: sdl.GPUTextureFormat,
) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_FULLSCREEN_VS, "vertexMain", .VERTEX)
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	// The BRDF LUT takes no input texture; binding one it does not declare is
	// harmless, declaring one it does not have is not, so the counts follow the
	// shader rather than this helper.
	samplers: u32 = blob.msl == SHADER_BRDF_LUT_FS.msl ? 0 : 1
	uniforms: u32 = blob.msl == SHADER_BRDF_LUT_FS.msl ? 0 : 1

	fs := shader_create(gpu, blob, "fragmentMain", .FRAGMENT, {samplers = samplers, uniform_buffers = uniforms})
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription{{format = format}}
	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			target_info = {num_color_targets = 1, color_target_descriptions = raw_data(&targets)},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: prefilter pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

// SDL requires every declared sampler slot to be bound even when the shader
// will not read it, so a scene without an environment substitutes the BRDF LUT.
@(private = "file")
env_or :: proc(r: ^Renderer, tex: ^sdl.GPUTexture) -> ^sdl.GPUTexture {
	return tex != nil ? tex : r.brdf_lut
}

// A label view can be selected before the label targets exist; bind the
// integer fallback as a placeholder rather than leaving the slot empty.
@(private = "file")
label_or :: proc(r: ^Renderer, tex: ^sdl.GPUTexture) -> ^sdl.GPUTexture {
	return tex != nil ? tex : r.labels.fallback
}

@(private = "file")
env_sampler :: proc(r: ^Renderer) -> ^sdl.GPUSampler {
	return r.env.sampler != nil ? r.env.sampler : r.target_sampler
}

@(private = "file")
make_shadow_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_SHADOW_VS, "vertexMain", .VERTEX, {uniform_buffers = 1})
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(gpu, SHADER_SHADOW_FS, "fragmentMain", .FRAGMENT)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	vertex_input := vertex_input_state()

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			vertex_input_state = vertex_input,
			primitive_type = .TRIANGLELIST,
			// Front-face culling is the usual acne remedy, but Lumbre's content
			// has no reliable winding -- the G-buffer pass draws two-sided for
			// the same reason -- so culling here would drop real casters. The
			// slope-scaled bias in the shader does the work instead.
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			depth_stencil_state = {
				compare_op = .LESS,
				enable_depth_test = true,
				enable_depth_write = true,
			},
			target_info = {
				num_color_targets = 0,
				depth_stencil_format = DEPTH_FORMAT,
				has_depth_stencil_target = true,
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: shadow pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

@(private = "file")
make_lighting_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_FULLSCREEN_VS, "vertexMain", .VERTEX)
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(
		gpu, SHADER_LIGHTING_FS, "fragmentMain", .FRAGMENT,
		{samplers = 16, storage_textures = 3, storage_buffers = 7, uniform_buffers = 1},
	)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription{{format = HDR_FORMAT}}

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			target_info = {
				num_color_targets = 1,
				color_target_descriptions = raw_data(&targets),
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: lighting pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

// Transmissive surfaces, drawn with the G-buffer's vertex stage -- same layout,
// same instancing -- and a fragment stage that shades and composites in one go.
@(private = "file")
make_forward_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_GBUFFER_VS, "vertexMain", .VERTEX, {uniform_buffers = 1})
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(
		gpu, SHADER_FORWARD_FS, "fragmentMain", .FRAGMENT,
		{samplers = 14, storage_textures = 3, storage_buffers = 7, uniform_buffers = 2},
	)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription{{format = HDR_FORMAT}}
	vertex_input := vertex_input_state()

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			vertex_input_state = vertex_input,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			depth_stencil_state = {
				compare_op = DEPTH_COMPARE,
				enable_depth_test = true,
				// No depth write: the draws are already sorted back to front,
				// and writing would let the near face of a glass shell hide its
				// own far face.
				enable_depth_write = false,
			},
			target_info = {
				num_color_targets = len(targets),
				color_target_descriptions = raw_data(&targets),
				depth_stencil_format = DEPTH_FORMAT,
				has_depth_stencil_target = true,
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: forward pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

@(private = "file")
make_debug_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_FULLSCREEN_VS, "vertexMain", .VERTEX)
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)

	fs := shader_create(
		gpu, SHADER_DEBUG_FS, "fragmentMain", .FRAGMENT,
		{samplers = 6, storage_textures = 3, uniform_buffers = 1},
	)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription{{format = COLOR_FORMAT}}

	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			target_info = {
				num_color_targets = 1,
				color_target_descriptions = raw_data(&targets),
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: debug pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

// ── targets ──────────────────────────────────────────────────────────────────

@(private = "file")
ensure_targets :: proc(r: ^Renderer, width, height: i32) -> bool {
	if r.color != nil && r.width == width && r.height == height {
		return true
	}
	release_targets(r)

	make_target :: proc(
		gpu: ^sdl.GPUDevice,
		format: sdl.GPUTextureFormat,
		usage: sdl.GPUTextureUsageFlags,
		width, height: i32,
	) -> ^sdl.GPUTexture {
		tex := sdl.CreateGPUTexture(
			gpu,
			sdl.GPUTextureCreateInfo {
				type = .D2,
				format = format,
				usage = usage,
				width = u32(width),
				height = u32(height),
				layer_count_or_depth = 1,
				num_levels = 1,
				sample_count = ._1,
			},
		)
		if tex == nil {
			fmt.eprintln("realtime: render target creation failed:", sdl.GetError())
		}
		return tex
	}

	color_usage := sdl.GPUTextureUsageFlags{.COLOR_TARGET, .SAMPLER}

	r.color = make_target(r.gpu, COLOR_FORMAT, color_usage, width, height)
	r.hdr_color = make_target(r.gpu, HDR_FORMAT, color_usage, width, height)
	r.ldr = make_target(r.gpu, COLOR_FORMAT, color_usage, width, height)
	r.ao_raw = make_target(r.gpu, AO_FORMAT, color_usage, width, height)
	r.ao = make_target(r.gpu, AO_FORMAT, color_usage, width, height)
	r.refraction_levels = mip_levels(width, height)
	r.refraction = make_render_target(r.gpu, HDR_FORMAT, u32(width), u32(height), r.refraction_levels)
	r.fog = make_target(r.gpu, HDR_FORMAT, color_usage, width, height)
	// Glossy rays are traced at half resolution; see ssr_fs.
	r.ssr_trace = make_target(r.gpu, HDR_FORMAT, color_usage, (width + 1) / 2, (height + 1) / 2)
	r.albedo = make_target(r.gpu, ALBEDO_FORMAT, color_usage, width, height)
	r.normal = make_target(r.gpu, NORMAL_FORMAT, color_usage, width, height)
	r.surface = make_target(r.gpu, SURFACE_FORMAT, color_usage, width, height)
	r.emission = make_target(r.gpu, EMISSION_FORMAT, color_usage, width, height)
	r.depth = make_target(r.gpu, DEPTH_FORMAT, {.DEPTH_STENCIL_TARGET, .SAMPLER}, width, height)
	r.taa_history[0] = make_target(r.gpu, HDR_FORMAT, color_usage, width, height)
	r.taa_history[1] = make_target(r.gpu, HDR_FORMAT, color_usage, width, height)
	r.hiz_levels = min(mip_levels(width, height), HIZ_MAX_LEVELS)
	r.hiz = make_render_target(r.gpu, HIZ_FORMAT, u32(width), u32(height), r.hiz_levels)

	if r.taa_history[0] == nil || r.taa_history[1] == nil || r.hiz == nil || r.fog == nil || r.ssr_trace == nil || r.color == nil || r.hdr_color == nil || r.ldr == nil || r.ao_raw == nil || r.ao == nil || r.refraction == nil || r.albedo == nil ||
	   r.normal == nil || r.surface == nil || r.emission == nil || r.depth == nil {
		release_targets(r)
		return false
	}

	r.width = width
	r.height = height
	return true
}

@(private = "file")
release_targets :: proc(r: ^Renderer) {
	for tex in ([]^sdl.GPUTexture{r.taa_history[0], r.taa_history[1], r.color, r.hdr_color, r.ldr, r.ao_raw, r.ao, r.hiz, r.fog, r.ssr_trace, r.refraction, r.albedo, r.normal, r.surface, r.emission, r.depth}) {
		if tex != nil {
			sdl.ReleaseGPUTexture(r.gpu, tex)
		}
	}
	r.color = nil
	r.hdr_color = nil
	r.taa_history = {}
	r.taa_valid = false
	r.ldr = nil
	r.ao_raw = nil
	r.ao = nil
	r.hiz = nil
	r.fog = nil
	r.ssr_trace = nil
	r.refraction = nil
	r.albedo = nil
	r.normal = nil
	r.surface = nil
	r.emission = nil
	r.depth = nil
	r.width = 0
	r.height = 0
}

@(private = "file")
draw_display :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, source: ^sdl.GPUTexture, fxaa: bool) {
	sdl.PushGPUDebugGroup(cmd, "Display encoding")
	defer sdl.PopGPUDebugGroup(cmd)
	// With FXAA the encoded image is an intermediate, carrying luma in alpha
	// for the edge search; without it this pass writes the presented image.
	output := fxaa ? r.ldr : r.color
	target := sdl.GPUColorTargetInfo{texture = output, load_op = .DONT_CARE, store_op = .STORE}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.display_pipeline)
	binding := sdl.GPUTextureSamplerBinding{texture = source, sampler = r.target_sampler}
	sdl.BindGPUFragmentSamplers(pass, 0, &binding, 1)
	params := [4]f32{
		math.pow(f32(2), r.settings.exposure),
		f32(r.settings.view_transform),
		fxaa ? 1 : 0,
		0,
	}
	sdl.PushGPUFragmentUniformData(cmd, 0, &params, size_of(params))
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
}

@(private = "file")
draw_fxaa :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer) {
	sdl.PushGPUDebugGroup(cmd, "FXAA")
	defer sdl.PopGPUDebugGroup(cmd)
	target := sdl.GPUColorTargetInfo{texture = r.color, load_op = .DONT_CARE, store_op = .STORE}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.fxaa_pipeline)
	binding := sdl.GPUTextureSamplerBinding{texture = r.ldr, sampler = r.linear_sampler}
	sdl.BindGPUFragmentSamplers(pass, 0, &binding, 1)
	params := [4]f32{1 / f32(r.width), 1 / f32(r.height), 0, 0}
	sdl.PushGPUFragmentUniformData(cmd, 0, &params, size_of(params))
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
}

// AO shader inputs; mirrors AoUniforms in shaders/ao_fs.slang.
@(private = "file")
Ao_Uniforms :: struct {
	inv_view_proj: matrix[4, 4]f32,
	view_proj:     matrix[4, 4]f32,
	eye:           [4]f32, // xyz, w = radius
	forward:       [4]f32, // xyz, w = world size of a pixel at unit depth
	right:         [4]f32, // xyz, w = image height in pixels
	up:            [4]f32,
	params:        [4]f32, // intensity (an exponent), 1/width, 1/height, unused
}

// Screen-space ambient occlusion from the G-buffer, then a depth-aware blur
// into `ao` for the lighting pass.
@(private = "file")
draw_ao :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera) {
	sdl.PushGPUDebugGroup(cmd, "Ambient occlusion")
	defer sdl.PopGPUDebugGroup(cmd)

	f := camera_frame(cam)
	view_proj := camera_view_proj(f, r.taa_jitter)
	extent := r.scene.bounds_max - r.scene.bounds_min
	radius := max(extent.x, max(extent.y, extent.z)) * AO_RADIUS_FRACTION * max(r.settings.ao_radius, 0.01)
	uniforms := Ao_Uniforms {
		inv_view_proj = linalg.inverse(view_proj),
		view_proj     = view_proj,
		eye           = {f.eye.x, f.eye.y, f.eye.z, radius},
		forward       = {f.forward.x, f.forward.y, f.forward.z, 2 * math.tan(f.vfov * 0.5) / f32(r.height)},
		right         = {f.right.x, f.right.y, f.right.z, f32(r.height)},
		up            = {f.up.x, f.up.y, f.up.z, 0},
		// GTAO's result is visibility itself, so no extra contrast.
		params        = {AO_INTENSITY, 1 / f32(r.width), 1 / f32(r.height), 0},
	}
	{
		target := sdl.GPUColorTargetInfo{texture = r.ao_raw, load_op = .DONT_CARE, store_op = .STORE}
		pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
		sdl.BindGPUGraphicsPipeline(pass, r.ao_pipeline)
		samplers := [2]sdl.GPUTextureSamplerBinding {
			{texture = r.normal, sampler = r.target_sampler},
			{texture = r.depth, sampler = r.target_sampler},
		}
		sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
		sdl.PushGPUFragmentUniformData(cmd, 0, &uniforms, size_of(uniforms))
		sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
		sdl.EndGPURenderPass(pass)
	}
	{
		target := sdl.GPUColorTargetInfo{texture = r.ao, load_op = .DONT_CARE, store_op = .STORE}
		pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
		sdl.BindGPUGraphicsPipeline(pass, r.ao_blur_pipeline)
		samplers := [2]sdl.GPUTextureSamplerBinding {
			{texture = r.ao_raw, sampler = r.target_sampler},
			{texture = r.depth, sampler = r.target_sampler},
		}
		sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
		params := [4]f32{1 / f32(r.width), 1 / f32(r.height), f.focus * NEAR_SCALE, f.focus * FAR_SCALE}
		sdl.PushGPUFragmentUniformData(cmd, 0, &params, size_of(params))
		sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
		sdl.EndGPURenderPass(pass)
	}
}

// Builds the depth pyramid: the depth buffer into level 0, then each level from
// the one below it. A level is read while the next is written, which is two
// subresources of one texture and so allowed.
@(private = "file")
draw_hiz :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer) {
	sdl.PushGPUDebugGroup(cmd, "Depth pyramid")
	defer sdl.PopGPUDebugGroup(cmd)
	for level in 0 ..< r.hiz_levels {
		target := sdl.GPUColorTargetInfo {
			texture   = r.hiz,
			mip_level = level,
			load_op   = .DONT_CARE,
			store_op  = .STORE,
		}
		pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
		sdl.BindGPUGraphicsPipeline(pass, r.hiz_pipeline)
		binding := sdl.GPUTextureSamplerBinding {
			texture = level == 0 ? r.depth : r.hiz,
			sampler = r.hiz_sampler,
		}
		sdl.BindGPUFragmentSamplers(pass, 0, &binding, 1)
		// The source level and its size; level 0 copies the depth buffer.
		source := level == 0 ? 0 : level - 1
		params := [4]f32{
			f32(source),
			level == 0 ? 1 : 0,
			f32(max(r.width >> source, 1)),
			f32(max(r.height >> source, 1)),
		}
		sdl.PushGPUFragmentUniformData(cmd, 0, &params, size_of(params))
		sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
		sdl.EndGPURenderPass(pass)
	}
}

// Reflection shader inputs; mirrors SsrUniforms in shaders/ssr_fs.slang.
@(private = "file")
Ssr_Uniforms :: struct {
	inv_view_proj: matrix[4, 4]f32,
	view_proj:     matrix[4, 4]f32,
	eye:           [4]f32, // xyz, w = how far a ray is traced
	forward:       [4]f32, // xyz, w = world size of a pixel at unit depth
	env:           [4]f32, // has_env, rotation, intensity, specular mip count
	size:          [4]f32, // width, height, 1/width, 1/height
	depth:         [4]f32, // near, far, pyramid levels, thickness
	params:        [4]f32, // AO drawn, reflection alone, blur pyramid levels, TAA frame
	// Baked probes, as in Lighting_Uniforms.
	probe_min:     [4]f32,
	probe_cell:    [4]f32,
	probe_dims:    [4]f32,
	rays:          [4]f32, // rays traced per pixel, gather radius in taps, unused x2
}

// Composites screen-space reflections over the lit opaque image in
// `refraction`, into `hdr_color`. `alone` writes only what was reflected.
@(private = "file")
draw_ssr :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera, alone: bool) {
	sdl.PushGPUDebugGroup(cmd, "Screen-space reflections")
	defer sdl.PopGPUDebugGroup(cmd)

	f := camera_frame(cam)
	view_proj := camera_view_proj(f, r.taa_jitter)
	extent := r.scene.bounds_max - r.scene.bounds_min
	largest := max(extent.x, max(extent.y, extent.z))
	uniforms := Ssr_Uniforms {
		inv_view_proj = linalg.inverse(view_proj),
		view_proj     = view_proj,
		// Nothing in the scene is further from a surface than its diagonal.
		eye           = {f.eye.x, f.eye.y, f.eye.z, linalg.length(extent)},
		forward       = {f.forward.x, f.forward.y, f.forward.z, 2 * math.tan(f.vfov * 0.5) / f32(r.height)},
		env           = {r.env.has_env ? 1 : 0, r.env.rotation, r.env.intensity, f32(SPECULAR_MIPS)},
		size          = {f32(r.width), f32(r.height), 1 / f32(r.width), 1 / f32(r.height)},
		depth         = {f.focus * NEAR_SCALE, f.focus * FAR_SCALE, f32(r.hiz_levels), largest * SSR_THICKNESS_FRACTION},
		// The TAA frame turns the glossy rays' pattern, as it does the soft
		// shadows' (pcss_spin), so TAA averages the grain away.
		params        = {r.settings.ao ? 1 : 0, alone ? 1 : 0, f32(r.refraction_levels), r.taa_on && r.taa_frame > 0 ? f32((r.taa_frame - 1) % TAA_SAMPLES) : 0},
	}
	uniforms.probe_min, uniforms.probe_cell, uniforms.probe_dims = probes_uniforms(r)
	// TAA averages its frames; a still shares each ray among more neighbours.
	uniforms.rays = {1, r.taa_on ? 1 : 2, 0, 0}

	// The trace: one glossy ray per pixel of the half-resolution `ssr_trace`.
	{
		target := sdl.GPUColorTargetInfo{texture = r.ssr_trace, load_op = .DONT_CARE, store_op = .STORE}
		pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
		sdl.BindGPUGraphicsPipeline(pass, r.ssr_pipeline)
		samplers := [4]sdl.GPUTextureSamplerBinding {
			// Linear, so a reflection is not snapped to the pixel grid of what
			// it reflects, and trilinear across the blur pyramid.
			{texture = r.refraction, sampler = r.mip_sampler},
			{texture = r.normal, sampler = r.target_sampler},
			{texture = r.surface, sampler = r.target_sampler},
			{texture = r.hiz, sampler = r.hiz_sampler},
		}
		sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
		sdl.PushGPUFragmentUniformData(cmd, 0, &uniforms, size_of(uniforms))
		sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
		sdl.EndGPURenderPass(pass)
	}

	// The resolve: neighbours share their rays, mirrors trace their own, and
	// the result replaces its share of the environment's reflection in
	// `hdr_color`.
	target := sdl.GPUColorTargetInfo{texture = r.hdr_color, load_op = .DONT_CARE, store_op = .STORE}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.ssr_resolve_pipeline)
	samplers := [9]sdl.GPUTextureSamplerBinding {
		// Trilinear: the resolve traces mirrors' rays itself and reads what
		// they hit as the trace pass does.
		{texture = r.refraction, sampler = r.mip_sampler},
		{texture = r.ssr_trace, sampler = r.target_sampler},
		{texture = r.albedo, sampler = r.target_sampler},
		{texture = r.normal, sampler = r.target_sampler},
		{texture = r.surface, sampler = r.target_sampler},
		{texture = r.hiz, sampler = r.hiz_sampler},
		{texture = env_or(r, r.env.specular), sampler = env_sampler(r)},
		{texture = r.brdf_lut, sampler = r.target_sampler},
		// Bound even with AO off; the shader gates on params.x.
		{texture = r.ao, sampler = r.target_sampler},
	}
	sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
	probe_irradiance, probe_depth, _, probe_radiance, probe_info := probes_buffers(r)
	buffers := [4]^sdl.GPUBuffer{probe_irradiance, probe_depth, probe_radiance, probe_info}
	sdl.BindGPUFragmentStorageBuffers(pass, 0, raw_data(&buffers), len(buffers))
	sdl.PushGPUFragmentUniformData(cmd, 0, &uniforms, size_of(uniforms))
	sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
}

// Fog shader inputs; mirrors FogUniforms in shaders/fog_fs.slang.
@(private = "file")
Fog_Uniforms :: struct {
	media:  [4]f32, // extinction at the floor, floor height, falloff height, anisotropy
	colour: [4]f32, // rgb = scattering albedo, w = how far a ray is marched
	multi:  [4]f32, // x = the distance light travels to leave the fog, y = multiple scattering on
}

// Marches the fog into `fog`, then blurs and blends it over `hdr_color`. After
// the forward pass, so glass is fogged too -- by the fog in front of the
// opaque surface behind it, which is too much, but less wrong than clear
// glass standing in a haze.
@(private = "file")
draw_fog :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, cam: lc.Camera, cascades: Cascades) {
	sdl.PushGPUDebugGroup(cmd, "Fog")
	defer sdl.PopGPUDebugGroup(cmd)

	f := camera_frame(cam)
	extent := r.scene.bounds_max - r.scene.bounds_min
	largest := max(extent.x, max(extent.y, extent.z), 1e-6)
	centre := (r.scene.bounds_max + r.scene.bounds_min) * 0.5
	// Twice the distance to the far side of the scene: past everything in it,
	// and far enough that the sky at the horizon is fogged as a distant
	// object would be.
	reach := 2 * (linalg.length(centre - f.eye) + linalg.length(extent) * 0.5)
	fog := Fog_Uniforms {
		media = {
			r.settings.fog_density / largest,
			r.scene.bounds_min.y,
			max(r.settings.fog_height, 1e-3) * max(extent.y, largest * FOG_FLAT_SCENE_HEIGHT),
			clamp(r.settings.fog_anisotropy, -0.95, 0.95),
		},
		colour = {
			clamp(r.settings.fog_colour.r, 0, 1),
			clamp(r.settings.fog_colour.g, 0, 1),
			clamp(r.settings.fog_colour.b, 0, 1),
			reach,
		},
	}
	// How far light goes before it is out of the fog: up through the layer's
	// falloff, but no further than across the scene.
	fog.multi = {min(fog.media.z, largest), r.settings.fog_multiple ? 1 : 0, 0, 0}
	frame := frame_uniforms(r, cam, cascades)

	{
		target := sdl.GPUColorTargetInfo{texture = r.fog, load_op = .DONT_CARE, store_op = .STORE}
		pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
		sdl.BindGPUGraphicsPipeline(pass, r.fog_pipeline)
		samplers := [5]sdl.GPUTextureSamplerBinding {
			{texture = r.depth, sampler = r.target_sampler},
			{texture = r.shadow_map, sampler = r.shadow_sampler},
			{texture = env_or(r, r.env.irradiance), sampler = env_sampler(r)},
			{texture = layers_or(r, r.spot_layers), sampler = r.shadow_sampler},
			{texture = layers_or(r, r.omni_layers), sampler = r.shadow_sampler},
		}
		sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
		_, _, probe_mean, _, _ := probes_buffers(r)
		buffers := [4]^sdl.GPUBuffer{r.light_buffer, r.clusters.range_buffer, r.clusters.index_buffer, probe_mean}
		sdl.BindGPUFragmentStorageBuffers(pass, 0, raw_data(&buffers), len(buffers))
		sdl.PushGPUFragmentUniformData(cmd, 0, &frame, size_of(frame))
		sdl.PushGPUFragmentUniformData(cmd, 1, &fog, size_of(fog))
		sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
		sdl.EndGPURenderPass(pass)
	}
	{
		target := sdl.GPUColorTargetInfo{texture = r.hdr_color, load_op = .LOAD, store_op = .STORE}
		pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
		sdl.BindGPUGraphicsPipeline(pass, r.fog_apply_pipeline)
		samplers := [2]sdl.GPUTextureSamplerBinding {
			{texture = r.fog, sampler = r.target_sampler},
			{texture = r.depth, sampler = r.target_sampler},
		}
		sdl.BindGPUFragmentSamplers(pass, 0, raw_data(&samplers), len(samplers))
		params := [4]f32{1 / f32(r.width), 1 / f32(r.height), f.focus * NEAR_SCALE, f.focus * FAR_SCALE}
		sdl.PushGPUFragmentUniformData(cmd, 0, &params, size_of(params))
		sdl.DrawGPUPrimitives(pass, 3, 1, 0, 0)
		sdl.EndGPURenderPass(pass)
	}
}

// A fullscreen pass with the given fragment resources and one colour target.
@(private = "file")
make_fullscreen_pipeline :: proc(
	gpu: ^sdl.GPUDevice, blob: Shader_Blob, format: sdl.GPUTextureFormat, bindings: Shader_Bindings,
	blend: sdl.GPUColorTargetBlendState = {},
) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_FULLSCREEN_VS, "vertexMain", .VERTEX)
	if vs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, vs)
	fs := shader_create(gpu, blob, "fragmentMain", .FRAGMENT, bindings)
	if fs == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription{{format = format, blend_state = blend}}
	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			target_info = {num_color_targets = 1, color_target_descriptions = raw_data(&targets)},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: fullscreen pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}

// ── shadows through glass ─────────────────────────────────────────────────────
//
// Glass does not write the opaque shadow maps; it draws glass maps beside them
// (glass_shadow_fs): per texel, the product of the glass tints along it and the
// nearest glass's depth or distance. See glass_lookup in shading.slang.

// RGB tint, alpha a depth or a distance in world units, which wants floats.
GLASS_MAP_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT
// Where no glass is: let everything through, from as far as a half can say.
GLASS_CLEAR :: sdl.FColor{1, 1, 1, 65504}

// Half the opaque map's resolution: about 60 MB for every shadow type at once,
// where full size would be four times that, and only while there is glass.
@(private = "file")
make_glass_maps :: proc(gpu: ^sdl.GPUDevice, resolution, layers: u32) -> ^sdl.GPUTexture {
	tex := sdl.CreateGPUTexture(
		gpu,
		sdl.GPUTextureCreateInfo {
			type = .D2_ARRAY,
			format = GLASS_MAP_FORMAT,
			// Read with Load as storage textures: see glass_lookup.
			usage = {.COLOR_TARGET, .GRAPHICS_STORAGE_READ},
			width = max(resolution / 2, 1),
			height = max(resolution / 2, 1),
			layer_count_or_depth = layers,
			num_levels = 1,
			sample_count = ._1,
		},
	)
	if tex == nil {
		fmt.eprintln("realtime: glass shadow map creation failed:", sdl.GetError())
	}
	return tex
}

// The three glass maps, as the lighting and forward passes' storage textures.
@(private = "file")
bind_glass_maps :: proc(r: ^Renderer, pass: ^sdl.GPURenderPass) {
	maps := [3]^sdl.GPUTexture{glass_or(r, r.sun_glass), glass_or(r, r.spot_layers.glass), glass_or(r, r.omni_layers.glass)}
	sdl.BindGPUFragmentStorageTextures(pass, 0, raw_data(&maps), len(maps))
}

// The glass map to bind: the real one while the scene has glass, else the
// clear 1x1 stand-in.
@(private = "file")
glass_or :: proc(r: ^Renderer, tex: ^sdl.GPUTexture) -> ^sdl.GPUTexture {
	return r.has_glass && tex != nil ? tex : r.glass_none
}

// What one glass surface lets through: its base colour as far as it
// transmits, scaled by how much it does. A closed object has two surfaces
// along most lines, so a clear ball at 0.9 lets 0.81 through.
@(private = "file")
glass_tint :: proc(m: Material_Uniforms) -> [3]f32 {
	t := clamp(m.flags.z, 0, 1)
	base := m.base_color.rgb
	return {(1 + (base.r - 1) * t) * t, (1 + (base.g - 1) * t) * t, (1 + (base.b - 1) * t) * t}
}

// A hash of every glass batch and its tint: 0 with no glass. The shadow maps
// are cached against geometry and the light's view, so a material turning to
// glass, or a pane changing colour, must invalidate them here.
@(private = "file")
glass_key :: proc(r: ^Renderer) -> u64 {
	key: u64 = 0
	for b, i in r.scene.batches {
		if !b.transparent { continue }
		tint := glass_tint(b.material)
		h := xxhash.XXH64(slice.to_bytes([]f32{f32(i), tint.r, tint.g, tint.b}))
		key = key * 31 + h + 1
	}
	return key
}

// Before the shadow passes: whether there is glass, its maps, and the
// stand-in, and a redraw of every shadow map when the glass changed.
@(private = "file")
glass_begin_frame :: proc(r: ^Renderer, cmd: ^sdl.GPUCommandBuffer) -> bool {
	if r.glass_none == nil {
		r.glass_none = make_glass_maps(r.gpu, 2, 1)
		if r.glass_none == nil { return false }
		target := sdl.GPUColorTargetInfo{texture = r.glass_none, clear_color = GLASS_CLEAR, load_op = .CLEAR, store_op = .STORE}
		pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
		sdl.EndGPURenderPass(pass)
	}
	key := glass_key(r)
	r.has_glass = key != 0
	if r.has_glass && r.sun_glass == nil {
		r.sun_glass = make_glass_maps(r.gpu, SHADOW_RESOLUTION, CASCADE_COUNT)
		if r.sun_glass == nil { return false }
		r.glass_drawn_key = ~key // force the redraw below
	}
	if key != r.glass_drawn_key {
		r.shadow_valid = {}
		r.spot_layers.valid = {}
		r.omni_layers.valid = {}
		r.glass_drawn_key = key
	}
	return true
}

// One layer of a glass map, from the light view `view_proj`. `light` is the
// light's position with w = 1 for a spot or omni light, whose maps record
// distance from it, or w = 0 for the sun's, which record depth.
@(private = "file")
draw_glass_layer :: proc(
	r: ^Renderer, cmd: ^sdl.GPUCommandBuffer, tex: ^sdl.GPUTexture, layer: u32,
	view_proj: matrix[4, 4]f32, light: [4]f32,
) {
	target := sdl.GPUColorTargetInfo {
		texture     = tex,
		layer_or_depth_plane = layer,
		clear_color = GLASS_CLEAR,
		load_op     = .CLEAR,
		store_op    = .STORE,
	}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, r.glass_shadow_pipeline)
	vp := view_proj
	sdl.PushGPUVertexUniformData(cmd, 0, &vp, size_of(vp))
	scene_bind_vertex_buffers(pass, &r.scene)
	// The visibility marks from the opaque pass just drawn from this view.
	s := &r.scene
	for b in s.batches {
		if !b.transparent { continue }
		tint := glass_tint(b.material)
		uniforms := [2][4]f32{light, {tint.r, tint.g, tint.b, 0}}
		sdl.PushGPUFragmentUniformData(cmd, 0, &uniforms, size_of(uniforms))
		start := int(b.first_instance)
		end := start + int(b.instance_count)
		for cursor := start; cursor < end; {
			first, count := instance_run(s.visible[:], cursor, end)
			if count == 0 { break }
			cursor = first + count
			sdl.DrawGPUIndexedPrimitives(pass, b.index_count, u32(count), b.first_index, b.base_vertex, u32(first))
		}
	}
	sdl.EndGPURenderPass(pass)
}

// Multiplies into colour and keeps the minimum in alpha; no depth.
@(private = "file")
make_glass_shadow_pipeline :: proc(gpu: ^sdl.GPUDevice) -> (^sdl.GPUGraphicsPipeline, bool) {
	vs := shader_create(gpu, SHADER_GLASS_SHADOW_VS, "vertexMain", .VERTEX, {uniform_buffers = 1})
	if vs == nil { return nil, false }
	defer sdl.ReleaseGPUShader(gpu, vs)
	fs := shader_create(gpu, SHADER_GLASS_SHADOW_FS, "fragmentMain", .FRAGMENT, {uniform_buffers = 1})
	if fs == nil { return nil, false }
	defer sdl.ReleaseGPUShader(gpu, fs)

	targets := [1]sdl.GPUColorTargetDescription {
		{
			format = GLASS_MAP_FORMAT,
			blend_state = {
				enable_blend = true,
				src_color_blendfactor = .ZERO,
				dst_color_blendfactor = .SRC_COLOR,
				color_blend_op = .ADD,
				src_alpha_blendfactor = .ONE,
				dst_alpha_blendfactor = .ONE,
				alpha_blend_op = .MIN,
			},
		},
	}
	pipeline := sdl.CreateGPUGraphicsPipeline(
		gpu,
		sdl.GPUGraphicsPipelineCreateInfo {
			vertex_shader = vs,
			fragment_shader = fs,
			vertex_input_state = vertex_input_state(),
			primitive_type = .TRIANGLELIST,
			rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
			target_info = {
				color_target_descriptions = raw_data(&targets),
				num_color_targets = 1,
			},
		},
	)
	if pipeline == nil {
		fmt.eprintln("realtime: glass shadow pipeline failed:", sdl.GetError())
		return nil, false
	}
	return pipeline, true
}
