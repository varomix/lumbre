package lumbre_realtime

// View and projection matrices for the rasterizer.
//
// There are none anywhere else in Lumbre. `core.Camera` is a pre-baked
// ray-generation basis — an origin plus the three vectors that span the image
// plane — because a path tracer never needs a matrix. A rasterizer needs
// nothing else.
//
// So rather than rebuild matrices from the orbit controls, these are derived
// from the `core.Camera` the path tracer is *actually rendering with*. That
// makes identical framing a property of the derivation instead of two
// independent code paths that have to be kept in agreement, which matters
// because comparing the two modes side by side is how every later step in this
// phase gets verified. `camera_test.odin` checks the round trip numerically.
//
// Reconstruction, given `make_camera`'s definitions:
//
//   centre = lower_left_corner + horizontal/2 + vertical/2 = origin - d*w
//     so   forward = normalize(centre - origin) = -w   and   d = |centre - origin|
//   |horizontal| = d * viewport_width      |vertical| = d * viewport_height
//   tan(vfov/2) = viewport_height / 2      aspect = viewport_width / viewport_height
//
// `u` and `v` are already an orthonormal right/up pair, so the view matrix is
// read straight off them with no re-orthogonalization.

import "core:math/linalg"
import m "core:math/linalg/glsl"

import lc "../core"

// What the shaders receive. Kept to vectors and matrices of 16-byte-aligned
// members so the layout is the same under MSL and SPIR-V without padding
// games.
Camera_Uniforms :: struct {
	view_proj: matrix[4, 4]f32,
	// World-space eye, needed by any view-dependent term (specular, IBL).
	// `w` is unused padding.
	eye:       [4]f32,
}

// The camera's own geometry, recovered from the ray-generation basis.
Camera_Frame :: struct {
	eye:     [3]f32,
	forward: [3]f32,
	right:   [3]f32,
	up:      [3]f32,
	// Vertical field of view in radians, and width/height.
	vfov:    f32,
	aspect:  f32,
	// Distance to the focus plane. The rasterizer ignores depth of field, but
	// this is what the near and far planes are scaled against.
	focus:   f32,
}

camera_frame :: proc(cam: lc.Camera) -> Camera_Frame {
	centre := cam.lower_left_corner + cam.horizontal * 0.5 + cam.vertical * 0.5
	to_centre := centre - cam.origin
	focus := m.length(to_centre)

	viewport_w := m.length(cam.horizontal) / focus
	viewport_h := m.length(cam.vertical) / focus

	return Camera_Frame {
		eye     = vec3f(cam.origin),
		forward = vec3f(to_centre / focus),
		right   = vec3f(cam.u),
		up      = vec3f(cam.v),
		vfov    = f32(2.0 * m.atan(viewport_h * 0.5)),
		aspect  = f32(viewport_w / viewport_h),
		focus   = f32(focus),
	}
}

// Near and far as multiples of the focus distance. A single scene-independent
// pair works here because the orbit camera always focuses on its target, so the
// focus distance already tracks the scene's scale; deriving them from scene
// bounds instead would make the depth buffer change under a camera move.
NEAR_SCALE :: 0.001
FAR_SCALE :: 1000.0

camera_uniforms :: proc(cam: lc.Camera) -> Camera_Uniforms {
	f := camera_frame(cam)
	return Camera_Uniforms {
		view_proj = camera_projection(f) * camera_view(f),
		eye = {f.eye.x, f.eye.y, f.eye.z, 0},
	}
}

// What the deferred lighting pass needs: the inverse transform to recover a
// world position from a depth sample, and the eye for view-dependent terms.
// Must match `LightingUniforms` in shaders/lighting_fs.slang field for field.
Lighting_Uniforms :: struct {
	inv_view_proj:     matrix[4, 4]f32,
	cascade_view_proj: [CASCADE_COUNT]matrix[4, 4]f32,
	eye:               [4]f32,
	forward:           [4]f32,
	cascade_splits:    [4]f32,
	cascade_texel:     [4]f32,
	// light_count, shadows_enabled, depth_bias, shadow texel size.
	params:            [4]f32,
	// has_env, rotation, intensity, specular mip count. Filled by the renderer,
	// which owns the environment.
	env:               [4]f32,
	// Froxel grid dimensions and depth slicing; see clusters.odin.
	cluster_dims:      [4]f32,
	cluster_depth:     [4]f32,
}

// Depth bias applied before the shadow comparison, in light-space depth units,
// scaled by surface slope in the shader. Small because the cascades are fitted
// tightly; a scene at a very different scale may need this revisited.
SHADOW_DEPTH_BIAS :: 0.0015

lighting_uniforms :: proc(
	cam: lc.Camera,
	light_count: u32,
	cascades: Cascades,
) -> Lighting_Uniforms {
	f := camera_frame(cam)
	view_proj := camera_projection(f) * camera_view(f)

	u := Lighting_Uniforms {
		inv_view_proj = linalg.inverse(view_proj),
		eye = {f.eye.x, f.eye.y, f.eye.z, 0},
		forward = {f.forward.x, f.forward.y, f.forward.z, 0},
		params = {
			f32(light_count),
			cascades.enabled ? 1 : 0,
			SHADOW_DEPTH_BIAS,
			1.0 / f32(SHADOW_RESOLUTION),
		},
		cluster_dims = cluster_dims(),
		cluster_depth = cluster_depth_params(f),
	}
	for c, i in cascades.slices {
		u.cascade_view_proj[i] = c.view_proj
		u.cascade_splits[i] = c.split_far
		u.cascade_texel[i] = c.texel_world
	}
	return u
}

// Right-handed look-along view matrix built directly from the basis.
camera_view :: proc(f: Camera_Frame) -> matrix[4, 4]f32 {
	// The camera looks down -z in view space, so the third row is -forward.
	r := f.right
	u := f.up
	b := -f.forward
	e := f.eye

	return matrix[4, 4]f32{
		r.x, r.y, r.z, -dot3(r, e),
		u.x, u.y, u.z, -dot3(u, e),
		b.x, b.y, b.z, -dot3(b, e),
		0,   0,   0,   1,
	}
}

// Perspective projection into SDL_GPU's clip space: y up, and depth in [0, 1]
// rather than OpenGL's [-1, 1]. SDL normalizes the backends to the D3D and
// Metal convention and flips y itself on Vulkan, so one matrix serves all
// three.
camera_projection :: proc(f: Camera_Frame) -> matrix[4, 4]f32 {
	near := f.focus * NEAR_SCALE
	far := f.focus * FAR_SCALE

	g := 1.0 / m.tan(f.vfov * 0.5) // cot(vfov/2)
	a := far / (near - far)
	b := (near * far) / (near - far)

	return matrix[4, 4]f32{
		g / f.aspect, 0, 0,  0,
		0,            g, 0,  0,
		0,            0, a,  b,
		0,            0, -1, 0,
	}
}

@(private = "file")
dot3 :: proc(a, b: [3]f32) -> f32 {
	return a.x * b.x + a.y * b.y + a.z * b.z
}

// Scene geometry is f64 throughout core; the GPU only ever sees f32.
@(private = "file")
vec3f :: proc(v: lc.Vec3) -> [3]f32 {
	return {f32(v.x), f32(v.y), f32(v.z)}
}
