package main

import m "core:math/linalg/glsl"
import "core:fmt"

sample_light :: proc(light: Light, from_point: Point3, rng: ^Rng) -> Light_Sample {
	switch light.kind {
	case .Quad:
		return sample_quad_light(light, from_point, rng)
	case .Sphere:
		return sample_sphere_light(light, from_point, rng)
	case .Disc:
		return sample_disc_light(light, from_point, rng)
	case .Cylinder:
		return sample_cylinder_light(light, from_point, rng)
	case .Point:
		return sample_point_light(light, from_point)
	case .Spot:
		return sample_spot_light(light, from_point)
	case .Distant:
		return sample_distant_light(light, from_point, rng)
	case .Mesh, .Dome:
		return Light_Sample{pdf = 0.0}
	}
	return Light_Sample{pdf = 0.0}
}

light_pdf :: proc(light: Light, from_point: Point3, direction: Vec3, distance: f64) -> f64 {
	switch light.kind {
	case .Quad:
		return quad_light_pdf(light, from_point, direction, distance)
	case .Sphere:
		return sphere_light_pdf(light, from_point, direction)
	case .Disc:
		return disc_light_pdf(light, from_point, direction, distance)
	case .Cylinder:
		return cylinder_light_pdf(light, from_point, direction, distance)
	case .Point, .Spot, .Distant, .Mesh, .Dome:
		return 0.0
	}
	return 0.0
}

sample_quad_light :: proc(light: Light, from_point: Point3, rng: ^Rng) -> Light_Sample {
	u := light.u
	v := light.v
	ru := rng_f64(rng)
	rv := rng_f64(rng)
	point := light.position + ru * u + rv * v
	normal := m.normalize(m.cross(u, v))
	area := light.area
	if area <= 0.0 {
		area = m.length(m.cross(u, v))
	}
	to_light := point - from_point
	distance := m.length(to_light)
	direction := to_light / distance

	cos_light := m.abs(m.dot(normal, direction))
	pdf := distance * distance / (cos_light * area)

	return Light_Sample{
		point     = point,
		normal    = normal,
		emission  = light.intensity,
		pdf       = pdf,
		direction = direction,
		distance  = distance,
	}
}

quad_light_pdf :: proc(light: Light, from_point: Point3, direction: Vec3, distance: f64) -> f64 {
	normal := m.normalize(m.cross(light.u, light.v))
	cos_light := m.abs(m.dot(normal, -direction))
	area := light.area
	if area <= 0.0 {
		area = m.length(m.cross(light.u, light.v))
	}
	if cos_light <= 0.0 || area <= 0.0 {
		return 0.0
	}
	return distance * distance / (cos_light * area)
}

sample_sphere_light :: proc(light: Light, from_point: Point3, rng: ^Rng) -> Light_Sample {
	center := light.position
	radius := light.radius
	if radius <= 0.0 {
		radius = 0.5
	}
	to_center := center - from_point
	center_dist := m.length(to_center)
	dir_to_center := to_center / center_dist

	sin_theta_max := radius / center_dist
	cos_theta_max := m.sqrt(max(1.0 - sin_theta_max * sin_theta_max, 0.0))

	eps1 := rng_f64(rng)
	eps2 := rng_f64(rng)
	cos_theta := 1.0 - eps1 * (1.0 - cos_theta_max)
	sin_theta := m.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
	phi := 2.0 * m.PI * eps2

	tangent := make_tangent_vec(dir_to_center)
	bitangent := m.cross(dir_to_center, tangent)
	local_dir := cos_theta * dir_to_center + sin_theta * (m.cos(phi) * tangent + m.sin(phi) * bitangent)

	distance := center_dist * cos_theta - m.sqrt(max(radius * radius - center_dist * center_dist * sin_theta * sin_theta, 0.0))
	point := from_point + distance * local_dir

	surface_normal := m.normalize(point - center)
	surface_to_eye := -local_dir
	if m.dot(surface_normal, surface_to_eye) < 0.0 {
		surface_normal = -surface_normal
	}

	pdf := 1.0 / (2.0 * m.PI * (1.0 - cos_theta_max))

	return Light_Sample{
		point     = point,
		normal    = surface_normal,
		emission  = light.intensity,
		pdf       = pdf,
		direction = local_dir,
		distance  = distance,
	}
}

sphere_light_pdf :: proc(light: Light, from_point: Point3, direction: Vec3) -> f64 {
	center := light.position
	radius := light.radius
	if radius <= 0.0 {
		radius = 0.5
	}
	to_center := center - from_point
	center_dist := m.length(to_center)

	sin_theta_max := radius / center_dist
	cos_theta_max := m.sqrt(max(1.0 - sin_theta_max * sin_theta_max, 0.0))

	return 1.0 / (2.0 * m.PI * (1.0 - cos_theta_max))
}

// ── Disc area light ─────────────────────────────────────────────────────────

sample_disc_light :: proc(light: Light, from_point: Point3, rng: ^Rng) -> Light_Sample {
	center := light.position
	normal := m.normalize(light.direction)
	radius := light.radius if light.radius > 0.0 else 0.5

	// Uniform point on the disc.
	r := radius * m.sqrt(rng_f64(rng))
	phi := 2.0 * m.PI * rng_f64(rng)
	tangent := make_tangent_vec(normal)
	bitangent := m.cross(normal, tangent)
	point := center + r * (m.cos(phi) * tangent + m.sin(phi) * bitangent)

	area := m.PI * radius * radius
	to_light := point - from_point
	distance := m.length(to_light)
	direction := to_light / distance

	if m.dot(normal, direction) > 0.0 {
		normal = -normal
	}
	cos_light := m.abs(m.dot(normal, direction))
	pdf := distance * distance / (cos_light * area) if cos_light > 0.0 else 0.0

	return Light_Sample{
		point     = point,
		normal    = normal,
		emission  = light.intensity,
		pdf       = pdf,
		direction = direction,
		distance  = distance,
	}
}

disc_light_pdf :: proc(light: Light, from_point: Point3, direction: Vec3, distance: f64) -> f64 {
	normal := m.normalize(light.direction)
	radius := light.radius if light.radius > 0.0 else 0.5
	area := m.PI * radius * radius
	cos_light := m.abs(m.dot(normal, -direction))
	if cos_light <= 0.0 || area <= 0.0 {
		return 0.0
	}
	return distance * distance / (cos_light * area)
}

// ── Cylinder area light (lateral surface) ───────────────────────────────────

sample_cylinder_light :: proc(light: Light, from_point: Point3, rng: ^Rng) -> Light_Sample {
	base := light.position
	axis := m.normalize(light.direction)
	radius := light.radius if light.radius > 0.0 else 0.5
	height := light.height if light.height > 0.0 else 1.0

	tangent := make_tangent_vec(axis)
	bitangent := m.cross(axis, tangent)

	t := height * rng_f64(rng)
	phi := 2.0 * m.PI * rng_f64(rng)
	radial := m.cos(phi) * tangent + m.sin(phi) * bitangent
	point := base + t * axis + radius * radial

	area := 2.0 * m.PI * radius * height
	to_light := point - from_point
	distance := m.length(to_light)
	direction := to_light / distance

	normal := radial // outward-facing
	cos_light := m.abs(m.dot(normal, direction))
	pdf := distance * distance / (cos_light * area) if cos_light > 0.0 else 0.0

	return Light_Sample{
		point     = point,
		normal    = normal,
		emission  = light.intensity,
		pdf       = pdf,
		direction = direction,
		distance  = distance,
	}
}

cylinder_light_pdf :: proc(light: Light, from_point: Point3, direction: Vec3, distance: f64) -> f64 {
	radius := light.radius if light.radius > 0.0 else 0.5
	height := light.height if light.height > 0.0 else 1.0
	area := 2.0 * m.PI * radius * height
	// Without the exact hit point we approximate the geometric term; a full
	// cylinder pdf would need the surface normal at the intersection. This is
	// only used for MIS balancing and is close enough for the lateral surface.
	if area <= 0.0 {
		return 0.0
	}
	return distance * distance / (0.5 * area)
}

// ── Delta lights: point, spot, distant ──────────────────────────────────────

sample_point_light :: proc(light: Light, from_point: Point3) -> Light_Sample {
	to_light := light.position - from_point
	distance := m.length(to_light)
	if distance <= 0.0 {
		return Light_Sample{pdf = 0.0}
	}
	direction := to_light / distance
	return Light_Sample{
		point     = light.position,
		normal    = -direction,
		emission  = light.intensity / (distance * distance),
		pdf       = 1.0,
		direction = direction,
		distance  = distance,
		delta     = true,
	}
}

sample_spot_light :: proc(light: Light, from_point: Point3) -> Light_Sample {
	to_light := light.position - from_point
	distance := m.length(to_light)
	if distance <= 0.0 {
		return Light_Sample{pdf = 0.0}
	}
	direction := to_light / distance
	// Cone falloff: angle between the spot axis and the light->surface ray.
	axis := m.normalize(light.direction)
	cos_angle := m.dot(axis, -direction)
	atten := 0.0
	if cos_angle >= light.cos_inner {
		atten = 1.0
	} else if cos_angle > light.cos_outer {
		t := (cos_angle - light.cos_outer) / (light.cos_inner - light.cos_outer)
		atten = t * t * (3.0 - 2.0 * t) // smoothstep
	}
	return Light_Sample{
		point     = light.position,
		normal    = -direction,
		emission  = light.intensity * atten / (distance * distance),
		pdf       = 1.0,
		direction = direction,
		distance  = distance,
		delta     = true,
	}
}

sample_distant_light :: proc(light: Light, from_point: Point3, rng: ^Rng) -> Light_Sample {
	dir := -m.normalize(light.direction)
	// Soft shadow: jitter the direction inside a cone of `angular_radius`.
	if light.angular_radius > 0.0 {
		cos_max := m.cos(light.angular_radius)
		eps1 := rng_f64(rng)
		eps2 := rng_f64(rng)
		cos_theta := 1.0 - eps1 * (1.0 - cos_max)
		sin_theta := m.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
		phi := 2.0 * m.PI * eps2
		tangent := make_tangent_vec(dir)
		bitangent := m.cross(dir, tangent)
		dir = m.normalize(cos_theta * dir + sin_theta * (m.cos(phi) * tangent + m.sin(phi) * bitangent))
	}
	return Light_Sample{
		point     = from_point + dir * 1.0e30,
		normal    = -dir,
		emission  = light.intensity,
		pdf       = 1.0,
		direction = dir,
		distance  = 1.0e30,
		delta     = true,
	}
}

make_tangent_vec :: proc(n: Vec3) -> Vec3 {
	helper := Vec3{0.0, 1.0, 0.0} if m.abs(n.x) > 0.9 else Vec3{1.0, 0.0, 0.0}
	return m.normalize(m.cross(helper, n))
}

make_area_light :: proc(position: Point3, u, v: Vec3, intensity: Color) -> Light {
	area := m.length(m.cross(u, v))
	return Light{
		kind      = .Quad,
		position  = position,
		u         = u,
		v         = v,
		intensity = intensity,
		area      = area,
		two_sided = false,
	}
}

make_sphere_light :: proc(center: Point3, radius: f64, intensity: Color) -> Light {
	return Light{
		kind      = .Sphere,
		position  = center,
		radius    = radius,
		intensity = intensity,
		area      = 4.0 * m.PI * radius * radius,
	}
}

make_disc_light :: proc(center: Point3, normal: Vec3, radius: f64, intensity: Color) -> Light {
	return Light{
		kind      = .Disc,
		position  = center,
		direction = m.normalize(normal),
		radius    = radius,
		intensity = intensity,
		area      = m.PI * radius * radius,
	}
}

make_cylinder_light :: proc(base: Point3, axis: Vec3, radius, height: f64, intensity: Color) -> Light {
	return Light{
		kind      = .Cylinder,
		position  = base,
		direction = m.normalize(axis),
		radius    = radius,
		height    = height,
		intensity = intensity,
		area      = 2.0 * m.PI * radius * height,
	}
}

make_point_light :: proc(position: Point3, intensity: Color) -> Light {
	return Light{kind = .Point, position = position, intensity = intensity}
}

// `inner`/`outer` are cone half-angles in radians (inner <= outer).
make_spot_light :: proc(position: Point3, direction: Vec3, inner, outer: f64, intensity: Color) -> Light {
	return Light{
		kind      = .Spot,
		position  = position,
		direction = m.normalize(direction),
		cos_inner = m.cos(inner),
		cos_outer = m.cos(outer),
		intensity = intensity,
	}
}

// `direction` is the direction the light travels (away from the source).
// `angular_radius` is the source half-angle in radians (0 = hard shadow).
make_distant_light :: proc(direction: Vec3, angular_radius: f64, intensity: Color) -> Light {
	return Light{
		kind           = .Distant,
		direction      = m.normalize(direction),
		angular_radius = angular_radius,
		intensity      = intensity,
	}
}

// Lights that are sampled by next-event estimation (everything except emissive
// meshes and the dome, which are handled on ray escape / via the environment).
is_nee_light :: proc(kind: Light_Kind) -> bool {
	switch kind {
	case .Quad, .Sphere, .Disc, .Cylinder, .Point, .Spot, .Distant:
		return true
	case .Mesh, .Dome:
		return false
	}
	return false
}

total_light_count :: proc(lights: []Light) -> int {
	count := 0
	for l in lights {
		if is_nee_light(l.kind) {
			count += 1
		}
	}
	return count
}

debug_print_lights :: proc(lights: []Light) {
	for l, i in lights {
		switch l.kind {
		case .Quad:
			fmt.println("  Light", i, "Quad pos:", l.position, "u:", l.u, "v:", l.v, "intensity:", l.intensity, "area:", l.area)
		case .Sphere:
			fmt.println("  Light", i, "Sphere center:", l.position, "radius:", l.radius, "intensity:", l.intensity)
		case .Disc:
			fmt.println("  Light", i, "Disc center:", l.position, "normal:", l.direction, "radius:", l.radius, "intensity:", l.intensity)
		case .Cylinder:
			fmt.println("  Light", i, "Cylinder base:", l.position, "axis:", l.direction, "radius:", l.radius, "height:", l.height, "intensity:", l.intensity)
		case .Point:
			fmt.println("  Light", i, "Point pos:", l.position, "intensity:", l.intensity)
		case .Spot:
			fmt.println("  Light", i, "Spot pos:", l.position, "dir:", l.direction, "cos_inner:", l.cos_inner, "cos_outer:", l.cos_outer, "intensity:", l.intensity)
		case .Distant:
			fmt.println("  Light", i, "Distant dir:", l.direction, "angular_radius:", l.angular_radius, "intensity:", l.intensity)
		case .Dome:
			fmt.println("  Light", i, "Dome (HDRI environment)")
		case .Mesh:
			fmt.println("  Light", i, "Mesh area:", l.area, "intensity:", l.intensity)
		}
	}
}
