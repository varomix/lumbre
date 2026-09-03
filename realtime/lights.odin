package lumbre_realtime

// Analytic lights for the deferred pass.
//
// The path tracer keeps one buffer per light shape because it samples each
// shape's surface properly (core/gpu_scene_cache_darwin.odin's `gpu_convert_lights`
// fans `Scene.lights` out into five arrays). The rasterizer collapses them into
// one struct with a kind tag, because it does not sample area lights at all —
// each is reduced to a representative point carrying its area. One array, one
// loop, and the shape only decides how the radiance is computed.
//
// What is deliberately dropped, and why it is visible:
//
//   - `.Mesh` — emissive triangles. The path tracer treats them as real light
//     sources; here they only appear in the emission target, lighting nothing.
//     A scene lit purely by emissive geometry (the Cornell box) therefore
//     renders its light panel and little else in realtime mode.
//   - `.Dome` — the HDRI environment, which is image-based lighting and lands
//     with the IBL step.

import "core:slice"

import lc "../core"
import sdl "vendor:sdl3"

Light_Kind_GPU :: enum i32 {
	Point    = 0,
	Spot     = 1,
	Distant  = 2,
	Quad     = 3,
	Sphere   = 4,
	Disc     = 5,
	Cylinder = 6,
}

// Mirrors `Light` in shaders/lighting_fs.slang. Vectors only, so MSL and
// SPIR-V agree on the layout.
Light_GPU :: struct {
	position:  [4]f32, // xyz, w = radius
	direction: [4]f32, // xyz aim or axis, w = height
	emission:  [4]f32, // rgb radiance or intensity, w = area
	params:    [4]f32, // kind, cos_inner, cos_outer, unused
}

// Converts `Scene.lights`, skipping the kinds this pass cannot express.
//
// Emission carries different units per kind, exactly as the path tracer's
// buffers do: radiant intensity for point and spot (the shader divides by
// distance squared), radiance for distant and for the area shapes (where the
// shader multiplies by the projected area instead).
lights_convert :: proc(lights: []lc.Light, out: ^[dynamic]Light_GPU) {
	clear(out)

	for l in lights {
		emission := [4]f32{f32(l.intensity.x), f32(l.intensity.y), f32(l.intensity.z), 0}
		position := [4]f32{f32(l.position.x), f32(l.position.y), f32(l.position.z), f32(l.radius)}
		direction := [4]f32{
			f32(l.direction.x), f32(l.direction.y), f32(l.direction.z), f32(l.height),
		}

		switch l.kind {
		case .Point:
			append(out, Light_GPU {
				position = position,
				emission = emission,
				params = {f32(i32(Light_Kind_GPU.Point)), 0, 0, 0},
			})

		case .Spot:
			append(out, Light_GPU {
				position = position,
				direction = direction,
				emission = emission,
				params = {
					f32(i32(Light_Kind_GPU.Spot)), f32(l.cos_inner), f32(l.cos_outer), 0,
				},
			})

		case .Distant:
			append(out, Light_GPU {
				direction = direction,
				emission = emission,
				params = {f32(i32(Light_Kind_GPU.Distant)), 0, 0, 0},
			})

		case .Quad:
			// `position` is a corner and u/v are the edges; the shader wants a
			// representative point, so hand it the centre and the quad's own
			// normal in the direction slot.
			centre := l.position + (l.u + l.v) * 0.5
			normal := normalize3(cross3(vec3f(l.u), vec3f(l.v)))
			e := emission
			e.w = f32(l.area)
			append(out, Light_GPU {
				position = {f32(centre.x), f32(centre.y), f32(centre.z), 0},
				direction = {normal.x, normal.y, normal.z, 0},
				emission = e,
				params = {f32(i32(Light_Kind_GPU.Quad)), 0, 0, 0},
			})

		case .Sphere:
			// A sphere radiates the same from every direction, so there is no
			// cosine at the light; its projected area is that of a disc.
			e := emission
			e.w = f32(3.14159265358979 * l.radius * l.radius)
			append(out, Light_GPU {
				position = position,
				emission = e,
				params = {f32(i32(Light_Kind_GPU.Sphere)), 0, 0, 0},
			})

		case .Disc:
			e := emission
			e.w = f32(3.14159265358979 * l.radius * l.radius)
			append(out, Light_GPU {
				position = position,
				direction = direction,
				emission = e,
				params = {f32(i32(Light_Kind_GPU.Disc)), 0, 0, 0},
			})

		case .Cylinder:
			// Side area only; the caps are not emissive in Lumbre's cylinder.
			// Treated as omnidirectional from the axis midpoint, which is the
			// crudest of these approximations.
			centre := l.position + l.direction * (l.height * 0.5)
			e := emission
			e.w = f32(2.0 * 3.14159265358979 * l.radius * l.height)
			append(out, Light_GPU {
				position = {
					f32(centre.x), f32(centre.y), f32(centre.z), f32(l.radius),
				},
				direction = direction,
				emission = e,
				params = {f32(i32(Light_Kind_GPU.Cylinder)), 0, 0, 0},
			})

		case .Mesh, .Dome:
			// See the file comment: emissive geometry and the environment are
			// not analytic lights and are handled (or not) elsewhere.
			continue
		}
	}
}

// Uploads the converted lights, growing the buffer when the count rises.
// Returns false only on an allocation failure; an empty light list is a
// legitimate result and leaves `count` at zero.
lights_upload :: proc(
	gpu: ^sdl.GPUDevice,
	buffer: ^^sdl.GPUBuffer,
	capacity: ^u32,
	data: []Light_GPU,
) -> bool {
	// SDL rejects a zero-sized buffer, and a scene with no analytic lights is
	// ordinary, so keep one element's worth allocated and draw with count 0.
	needed := u32(max(len(data), 1) * size_of(Light_GPU))

	if buffer^ == nil || capacity^ < needed {
		if buffer^ != nil {
			sdl.ReleaseGPUBuffer(gpu, buffer^)
		}
		buffer^ = sdl.CreateGPUBuffer(
			gpu,
			sdl.GPUBufferCreateInfo{usage = {.GRAPHICS_STORAGE_READ}, size = needed},
		)
		if buffer^ == nil {
			capacity^ = 0
			return false
		}
		capacity^ = needed
	}

	if len(data) == 0 {
		return true
	}
	return upload_bytes(gpu, buffer^, slice.to_bytes(data))
}

@(private = "file")
cross3 :: proc(a, b: [3]f32) -> [3]f32 {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}
}
