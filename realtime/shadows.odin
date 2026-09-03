package lumbre_realtime

// Cascaded shadow maps for the sun.
//
// One distant light casts shadows — the first one in the scene. Every other
// light is unshadowed, which is the usual realtime bargain: a shadow map per
// point light is another depth render per light per frame, and the sun is the
// one whose shadows carry the image.
//
// The camera's view range is split into four slices, each rendered from the
// light into its own layer of a depth array, so a texel near the eye covers far
// less world than one at the far plane. Without that, a single map stretched
// over the whole range is either blocky up close or too small to cover the
// distance.
//
// Two details keep the result stable rather than merely correct:
//
//   - Each cascade is fitted to the BOUNDING SPHERE of its frustum slice, not
//     to the slice's box. A sphere is invariant under rotation, so turning the
//     camera does not change the fitted size — a box fit grows and shrinks as
//     it spins, and the shadow edges crawl.
//   - The fitted centre is SNAPPED to the shadow map's texel grid. Without it
//     the projection slides by sub-texel amounts every frame and edges shimmer.
//
// The cascade maths is deliberately plain Odin so it can be tested without a
// GPU; `shadows_test.odin` covers it.

import "core:math"
import "core:math/linalg"

import lc "../core"

CASCADE_COUNT :: 4
SHADOW_RESOLUTION :: 2048

// Blend between a uniform and a logarithmic split. 0 is uniform (wastes
// resolution near the eye), 1 is fully logarithmic (starves the far cascade).
// The usual compromise.
CASCADE_SPLIT_LAMBDA :: 0.75

Cascade :: struct {
	view_proj: matrix[4, 4]f32,
	// World-space size of one shadow texel in this cascade. The normal-offset
	// bias scales with it: a far cascade's texel covers much more world, and an
	// offset sized for the near one does nothing out there.
	texel_world: f32,
	// View-space distance where this cascade stops applying. The shader picks a
	// cascade by comparing against these in order.
	split_far: f32,
}

Cascades :: struct {
	slices:    [CASCADE_COUNT]Cascade,
	// Direction the light travels, normalized — matching `core.Light.direction`.
	light_dir: [3]f32,
	enabled:   bool,
}

// The sun: the first distant light in the scene, or nothing.
sun_light :: proc(lights: []lc.Light) -> (dir: [3]f32, found: bool) {
	for l in lights {
		if l.kind == .Distant {
			d := [3]f32{f32(l.direction.x), f32(l.direction.y), f32(l.direction.z)}
			return normalize3(d), true
		}
	}
	return {}, false
}

// Builds the four cascades for this camera and sun.
//
// `bounds_min`/`bounds_max` are the scene's world AABB. They set how far
// shadows reach: the camera's own far plane is a thousand focus distances away
// and fitting cascades to that would spend the entire depth range on empty
// space.
shadow_build_cascades :: proc(
	f: Camera_Frame,
	light_dir: [3]f32,
	bounds_min, bounds_max: [3]f32,
) -> Cascades {
	out: Cascades
	out.light_dir = normalize3(light_dir)
	out.enabled = true

	near := f.focus * NEAR_SCALE
	far := scene_view_far(f, bounds_min, bounds_max)
	if far <= near {
		out.enabled = false
		return out
	}

	prev := near
	for i in 0 ..< CASCADE_COUNT {
		split := cascade_split(near, far, i + 1, CASCADE_COUNT)
		out.slices[i] = fit_cascade(f, out.light_dir, prev, split)
		out.slices[i].split_far = split
		prev = split
	}
	return out
}

// The practical split scheme: a blend of uniform and logarithmic spacing.
@(private = "file")
cascade_split :: proc(near, far: f32, index, count: int) -> f32 {
	ratio := f32(index) / f32(count)
	uniform := near + (far - near) * ratio
	logarithmic := near * math.pow(far / near, ratio)
	return linalg.lerp(uniform, logarithmic, f32(CASCADE_SPLIT_LAMBDA))
}

// Distance along the view axis to the farthest corner of the scene.
//
// Used for two things that both need "how far does this scene actually go":
// the cascades, so they cover the geometry and no more, and the depth debug
// view, which is unreadable when scaled against the camera's far plane.
scene_view_far :: proc(f: Camera_Frame, bounds_min, bounds_max: [3]f32) -> f32 {
	furthest: f32 = 0
	for i in 0 ..< 8 {
		corner := [3]f32 {
			(i & 1) != 0 ? bounds_max.x : bounds_min.x,
			(i & 2) != 0 ? bounds_max.y : bounds_min.y,
			(i & 4) != 0 ? bounds_max.z : bounds_min.z,
		}
		d := corner - f.eye
		// Distance along the view axis, which is what the splits compare to.
		along := d.x * f.forward.x + d.y * f.forward.y + d.z * f.forward.z
		furthest = max(furthest, along)
	}
	return furthest
}

// Fits one cascade to the frustum slice between `near` and `far`.
@(private = "file")
fit_cascade :: proc(f: Camera_Frame, light_dir: [3]f32, near, far: f32) -> Cascade {
	// The slice's eight corners.
	tan_half := math.tan(f.vfov * 0.5)
	corners: [8][3]f32
	idx := 0
	for d in ([2]f32{near, far}) {
		half_h := d * tan_half
		half_w := half_h * f.aspect
		centre := f.eye + f.forward * d
		for sy in ([2]f32{-1, 1}) {
			for sx in ([2]f32{-1, 1}) {
				corners[idx] = centre + f.right * (half_w * sx) + f.up * (half_h * sy)
				idx += 1
			}
		}
	}

	// Bounding sphere of the slice. Rotation-invariant, unlike a box fit.
	centre: [3]f32
	for c in corners {
		centre += c
	}
	centre /= f32(len(corners))

	radius: f32 = 0
	for c in corners {
		d := c - centre
		radius = max(radius, math.sqrt(d.x * d.x + d.y * d.y + d.z * d.z))
	}
	// A hair of slack so texel snapping below can never push geometry outside.
	radius = math.ceil(radius * 16.0) / 16.0

	// Light basis. `light_dir` is the direction light TRAVELS, so the eye sits
	// back along it.
	forward := light_dir
	up_ref := abs(forward.y) > 0.99 ? [3]f32{0, 0, 1} : [3]f32{0, 1, 0}
	right := normalize3(cross3(up_ref, forward))
	up := cross3(forward, right)

	// Snap the centre to the shadow map's texel grid, in light space. Sliding
	// by a fraction of a texel between frames is what makes shadow edges crawl.
	texels_per_unit := f32(SHADOW_RESOLUTION) / (radius * 2.0)
	cx := dot3(right, centre) * texels_per_unit
	cy := dot3(up, centre) * texels_per_unit
	cz := dot3(forward, centre)
	cx = math.floor(cx) / texels_per_unit
	cy = math.floor(cy) / texels_per_unit
	centre = right * cx + up * cy + forward * cz

	// Pull the light's eye back far enough to see everything that could cast
	// into this slice, and extend the far plane by the same amount.
	extent := radius * 4.0
	eye := centre - forward * extent

	view := matrix[4, 4]f32{
		right.x,   right.y,   right.z,   -dot3(right, eye),
		up.x,      up.y,      up.z,      -dot3(up, eye),
		-forward.x, -forward.y, -forward.z, dot3(forward, eye),
		0, 0, 0, 1,
	}

	// Orthographic into [0, 1] depth, matching the perspective convention.
	n: f32 = 0
	fz := extent + radius * 2.0
	proj := matrix[4, 4]f32{
		1.0 / radius, 0, 0, 0,
		0, 1.0 / radius, 0, 0,
		0, 0, -1.0 / (fz - n), -n / (fz - n),
		0, 0, 0, 1,
	}

	return Cascade{view_proj = proj * view, texel_world = (radius * 2.0) / f32(SHADOW_RESOLUTION)}
}

@(private = "file")
cross3 :: proc(a, b: [3]f32) -> [3]f32 {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}
}

@(private = "file")
dot3 :: proc(a, b: [3]f32) -> f32 {
	return a.x * b.x + a.y * b.y + a.z * b.z
}
