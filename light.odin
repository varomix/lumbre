package main

import m "core:math/linalg/glsl"
import "core:fmt"

sample_light :: proc(light: Light, from_point: Point3, rng: ^Rng) -> Light_Sample {
	switch light.kind {
	case .Quad:
		return sample_quad_light(light, from_point, rng)
	case .Sphere:
		return sample_sphere_light(light, from_point, rng)
	case .Mesh:
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
	case .Mesh:
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

total_light_count :: proc(lights: []Light) -> int {
	count := 0
	for l in lights {
		if l.kind == .Quad || l.kind == .Sphere {
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
		case .Mesh:
			fmt.println("  Light", i, "Mesh area:", l.area, "intensity:", l.intensity)
		}
	}
}
