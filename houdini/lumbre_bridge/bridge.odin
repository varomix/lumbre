package lumbre_bridge

import "base:runtime"
import "core:c"
import lc "../../core"
import m "core:math/linalg/glsl"
import stbi "vendor:stb/image"

// The Houdini bridge has a deliberately small C ABI. Houdini never sees Odin
// structs or the standalone USD importer; this package imports only the
// USD-free renderer `core`, so the resulting dylib never links
// `lib/darwin/libusd_shim.dylib`. It is the persistent boundary that owns
// renderer state for the Hydra frontend.

LUMBRE_HOUDINI_BRIDGE_ABI_VERSION :: 3

Lumbre_Bridge_Triangle :: struct {
	positions: [9]f32,
	normals:   [9]f32,
	uvs:       [6]f32,
	has_uv:    i32,
	material_index: i32,
}

Lumbre_Bridge_Material :: struct {
	base_color: [3]f32,
	emission: [3]f32,
	metallic, roughness, specular, ior, transmission, emission_strength: f32,
	base_color_texture: [1024]u8,
	metallic_roughness_texture: [1024]u8,
	normal_texture: [1024]u8,
	emission_texture: [1024]u8,
}

Lumbre_Bridge_Light :: struct {
	kind:           i32,
	position:       [3]f32,
	direction:      [3]f32,
	u:              [3]f32,
	v:              [3]f32,
	intensity:      [3]f32,
	radius:         f32,
	height:         f32,
	cos_inner:      f32,
	cos_outer:      f32,
	angular_radius: f32,
}

Lumbre_Bridge_Context :: struct {
	core:      lc.Lumbre_Core,
	// The most recent CPU render. Owned by the bridge; freed on the next
	// render and on destroy. Frontends read it back through the framebuffer
	// API instead of receiving Odin slices.
	buffer:    lc.Render_Buffer,
	has_frame: bool,
}

@(export)
lumbre_bridge_abi_version :: proc "c" () -> u32 {
	return LUMBRE_HOUDINI_BRIDGE_ABI_VERSION
}

@(export)
lumbre_bridge_create :: proc "c" () -> rawptr {
	context = runtime.default_context()
	bridge := new(Lumbre_Bridge_Context)
	bridge.core.settings = lc.Render_Config{
		image_width       = 640,
		image_height      = 360,
		samples_per_pixel = 4,
		max_depth         = 6,
		max_radiance      = 1000,
		use_gpu           = true, // Metal ray tracing is the viewport priority
		roughness_cutoff  = 0.95,
		gi_cache_enabled  = true,
		photon_enabled    = true,
		photon_count      = 200000,
		photon_bounces    = 8,
		// Hydra owns the scene: do not invent a procedural sky when the USD
		// stage has no DomeLight/environment.
		hide_default_sky  = true,
	}
	// A neutral default camera so a first render is framed even before Houdini
	// sends camera data.
	lumbre_bridge_set_camera_internal(bridge,
		lc.Vec3{0, 1, 4}, lc.Vec3{0, 0, 0}, lc.Vec3{0, 1, 0}, 40.0)
	return bridge
}

@(export)
lumbre_bridge_destroy :: proc "c" (handle: rawptr) {
	if handle == nil { return }
	context = runtime.default_context()
	bridge := cast(^Lumbre_Bridge_Context)handle
	if bridge.has_frame {
		lc.destroy_render_buffer(&bridge.buffer)
	}
	lc.destroy_scene(&bridge.core.scene)
	free(bridge)
}

@(export)
lumbre_bridge_replace_triangles :: proc "c" (
	handle: rawptr,
	triangles: [^]Lumbre_Bridge_Triangle,
	triangle_count: i32,
) -> bool {
	if handle == nil || triangles == nil || triangle_count <= 0 { return false }
	context = runtime.default_context()
	converted := make([]lc.Triangle, triangle_count)
	defer delete(converted)
	for i in 0 ..< int(triangle_count) {
		src := triangles[i]
		converted[i] = lc.Triangle{
			v0 = lc.Vec3{f64(src.positions[0]), f64(src.positions[1]), f64(src.positions[2])},
			v1 = lc.Vec3{f64(src.positions[3]), f64(src.positions[4]), f64(src.positions[5])},
			v2 = lc.Vec3{f64(src.positions[6]), f64(src.positions[7]), f64(src.positions[8])},
			n0 = lc.Vec3{f64(src.normals[0]), f64(src.normals[1]), f64(src.normals[2])},
			n1 = lc.Vec3{f64(src.normals[3]), f64(src.normals[4]), f64(src.normals[5])},
			n2 = lc.Vec3{f64(src.normals[6]), f64(src.normals[7]), f64(src.normals[8])},
			uv0 = lc.Vec3{f64(src.uvs[0]), f64(src.uvs[1]), 0},
			uv1 = lc.Vec3{f64(src.uvs[2]), f64(src.uvs[3]), 0},
			uv2 = lc.Vec3{f64(src.uvs[4]), f64(src.uvs[5]), 0},
			has_uv = src.has_uv != 0,
			mat_idx = src.material_index,
		}
	}
	bridge := cast(^Lumbre_Bridge_Context)handle
	lc.lumbre_core_replace_triangles(&bridge.core, converted)
	return true
}

@(export)
lumbre_bridge_replace_materials :: proc "c" (
	handle: rawptr,
	materials: [^]Lumbre_Bridge_Material,
	material_count: i32,
) -> bool {
	if handle == nil || material_count < 0 || (material_count > 0 && materials == nil) { return false }
	context = runtime.default_context()
	bridge := cast(^Lumbre_Bridge_Context)handle
	for &mat in bridge.core.scene.materials {
		lc.destroy_material_textures(&mat)
	}
	delete(bridge.core.scene.materials)
	if material_count == 0 { return true }
	converted := make([]lc.Material, material_count)
	for i in 0 ..< int(material_count) {
		src := materials[i]
		mat := lc.Material{
			kind = .Principled,
			albedo = lc.Color{f64(src.base_color[0]), f64(src.base_color[1]), f64(src.base_color[2])},
			emission = lc.Color{f64(src.emission[0]), f64(src.emission[1]), f64(src.emission[2])},
			metallic = f64(src.metallic), roughness = f64(src.roughness),
			specular = f64(src.specular), ir = f64(src.ior), spec_trans = f64(src.transmission),
			emission_strength = f64(src.emission_strength),
		}
		if path := cstring(&src.base_color_texture[0]); path != "" { mat.albedo_tex, _ = lc.load_texture(string(path), "", true) }
		if path := cstring(&src.metallic_roughness_texture[0]); path != "" { mat.metallic_roughness_tex, _ = lc.load_texture(string(path), "", false) }
		if path := cstring(&src.normal_texture[0]); path != "" { mat.normal_tex, _ = lc.load_texture(string(path), "", false) }
		if path := cstring(&src.emission_texture[0]); path != "" { mat.emissive_tex, _ = lc.load_texture(string(path), "", true) }
		converted[i] = lc.normalize_material(mat)
	}
	bridge.core.scene.materials = converted
	return true
}

@(export)
lumbre_bridge_replace_lights :: proc "c" (
	handle: rawptr,
	lights: [^]Lumbre_Bridge_Light,
	light_count: i32,
) -> bool {
	if handle == nil || light_count < 0 { return false }
	context = runtime.default_context()
	converted := make([]lc.Light, light_count)
	defer delete(converted)
	for i in 0 ..< int(light_count) {
		src := lights[i]
		converted[i] = lc.Light{
			kind = lc.Light_Kind(src.kind),
			position = lc.Vec3{f64(src.position[0]), f64(src.position[1]), f64(src.position[2])},
			direction = lc.Vec3{f64(src.direction[0]), f64(src.direction[1]), f64(src.direction[2])},
			u = lc.Vec3{f64(src.u[0]), f64(src.u[1]), f64(src.u[2])},
			v = lc.Vec3{f64(src.v[0]), f64(src.v[1]), f64(src.v[2])},
			intensity = lc.Color{f64(src.intensity[0]), f64(src.intensity[1]), f64(src.intensity[2])},
			radius = f64(src.radius), height = f64(src.height),
			cos_inner = f64(src.cos_inner), cos_outer = f64(src.cos_outer),
			angular_radius = f64(src.angular_radius),
		}
	}
	bridge := cast(^Lumbre_Bridge_Context)handle
	lc.lumbre_core_replace_lights(&bridge.core, converted)
	return true
}

// Set the render resolution. Cancels no work by itself; the next
// `lumbre_bridge_render` uses the updated size.
@(export)
lumbre_bridge_set_resolution :: proc "c" (handle: rawptr, width, height: i32) -> bool {
	if handle == nil || width <= 0 || height <= 0 { return false }
	bridge := cast(^Lumbre_Bridge_Context)handle
	bridge.core.settings.image_width = width
	bridge.core.settings.image_height = height
	return true
}

// Set samples-per-pixel and maximum ray depth for subsequent renders.
@(export)
lumbre_bridge_set_quality :: proc "c" (handle: rawptr, samples_per_pixel, max_depth: i32) -> bool {
	if handle == nil { return false }
	bridge := cast(^Lumbre_Bridge_Context)handle
	if samples_per_pixel > 0 { bridge.core.settings.samples_per_pixel = samples_per_pixel }
	if max_depth > 0 { bridge.core.settings.max_depth = max_depth }
	return true
}

// Set the camera from a look-from/look-at/up frame plus a vertical field of
// view in degrees. The aspect ratio is derived from the current resolution.
@(export)
lumbre_bridge_set_camera :: proc "c" (
	handle: rawptr,
	// C arrays decay to pointers across the ABI, so these are `^[3]f32`, not
	// arrays passed by value. Passing them by value silently misaligns every
	// argument after the first.
	origin: ^[3]f32,
	look_at: ^[3]f32,
	up: ^[3]f32,
	vfov_degrees: f32,
) -> bool {
	if handle == nil || origin == nil || look_at == nil || up == nil { return false }
	context = runtime.default_context()
	bridge := cast(^Lumbre_Bridge_Context)handle
	lumbre_bridge_set_camera_internal(bridge,
		lc.Vec3{f64(origin[0]), f64(origin[1]), f64(origin[2])},
		lc.Vec3{f64(look_at[0]), f64(look_at[1]), f64(look_at[2])},
		lc.Vec3{f64(up[0]), f64(up[1]), f64(up[2])},
		f64(vfov_degrees))
	return true
}

lumbre_bridge_set_camera_internal :: proc(
	bridge: ^Lumbre_Bridge_Context,
	origin, look_at, up: lc.Vec3,
	vfov_degrees: f64,
) {
	w := bridge.core.settings.image_width
	h := bridge.core.settings.image_height
	aspect := f64(w) / f64(max(h, 1))
	bridge.core.scene.camera = lc.make_camera(
		origin, look_at, up, vfov_degrees, aspect, 0.0, m.length(look_at - origin))
}

// Render synchronously on the CPU into the bridge-owned framebuffer. Returns
// false if nothing could be rendered. The GPU (Metal) path lands in `core`
// later; the CPU path is the portable reference the Houdini viewport uses
// first.
@(export)
lumbre_bridge_render :: proc "c" (handle: rawptr) -> bool {
	if handle == nil { return false }
	context = runtime.default_context()
	bridge := cast(^Lumbre_Bridge_Context)handle

	if bridge.has_frame {
		lc.destroy_render_buffer(&bridge.buffer)
		bridge.has_frame = false
	}
	if bridge.core.settings.use_gpu {
		bridge.buffer = lc.lumbre_core_render_gpu(&bridge.core)
	} else {
		bridge.buffer = lc.lumbre_core_render_cpu(&bridge.core)
	}
	bridge.has_frame = true
	return true
}

// Toggle the GPU (Metal) renderer. On by default; the CPU path is a portable
// fallback for diagnostics.
@(export)
lumbre_bridge_set_use_gpu :: proc "c" (handle: rawptr, use_gpu: b32) -> bool {
	// `b32` matches C's 4-byte `int`; an Odin `bool` (1 byte) would misread the
	// argument across the ABI.
	if handle == nil { return false }
	bridge := cast(^Lumbre_Bridge_Context)handle
	bridge.core.settings.use_gpu = bool(use_gpu)
	return true
}

// Debug: write the last completed frame to a PNG. Lets a frontend confirm the
// renderer works independently of any viewport/present path.
@(export)
lumbre_bridge_write_png :: proc "c" (handle: rawptr, path: cstring) -> bool {
	if handle == nil || path == nil { return false }
	context = runtime.default_context()
	bridge := cast(^Lumbre_Bridge_Context)handle
	if !bridge.has_frame { return false }
	ok := stbi.write_png(path, c.int(bridge.buffer.width), c.int(bridge.buffer.height),
		3, raw_data(bridge.buffer.pixels), c.int(bridge.buffer.width * 3))
	return ok != 0
}

// Report the resolution of the last completed frame.
@(export)
lumbre_bridge_framebuffer_size :: proc "c" (handle: rawptr, out_width, out_height: ^i32) -> bool {
	if handle == nil { return false }
	bridge := cast(^Lumbre_Bridge_Context)handle
	if !bridge.has_frame { return false }
	if out_width != nil { out_width^ = bridge.buffer.width }
	if out_height != nil { out_height^ = bridge.buffer.height }
	return true
}

// Copy the last completed frame into a caller-provided float RGBA buffer of
// `width * height * 4` floats (channels in [0,1], alpha = 1). The image is
// flipped vertically so it matches Hydra's lower-left origin: `core` stores
// the top image row first. Returns false if no frame exists or the caller's
// dimensions do not match the frame.
@(export)
lumbre_bridge_read_rgba_f32 :: proc "c" (
	handle: rawptr,
	dst: [^]f32,
	width, height: i32,
) -> bool {
	if handle == nil || dst == nil { return false }
	bridge := cast(^Lumbre_Bridge_Context)handle
	if !bridge.has_frame { return false }
	if width != bridge.buffer.width || height != bridge.buffer.height { return false }

	src := bridge.buffer.pixels // RGB8, top row first
	inv := f32(1.0) / 255.0
	for y in 0 ..< int(height) {
		src_row := int(height) - 1 - y // flip to lower-left origin
		for x in 0 ..< int(width) {
			si := (src_row * int(width) + x) * 3
			di := (y * int(width) + x) * 4
			dst[di + 0] = f32(src[si + 0]) * inv
			dst[di + 1] = f32(src[si + 1]) * inv
			dst[di + 2] = f32(src[si + 2]) * inv
			dst[di + 3] = 1.0
		}
	}
	return true
}
