#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

// ── GPU data types ───────────────────────────────────────────────────────────

struct GPUMaterial {
	float3 albedo;
	float  _pad0;
	float3 emission;
	float  _pad1;
	int    kind;
	float  fuzz;
	float  ir;
	float  roughness;
	float  metallic;
	float  emission_strength;
	float  _pad2[3];
};

struct GPUSceneData {
	float4 origin;
	float4 lower_left;
	float4 horizontal;
	float4 vertical;
	float4 u;
	float4 v;
	float  lens_radius;
	int    image_width;
	int    image_height;
	int    samples_per_pixel;
	int    max_depth;
	float  max_radiance;
	int    debug_mode;
	int    light_count;
	uint   seed;
	float  _pad[3];
};

struct TriVertex {
	float3 position;
	float  _pad;
};

// ── GPU RNG (PCG-style) ─────────────────────────────────────────────────────

static uint rng_next(thread uint& state) {
	state = state * 747796405u + 2891336453u;
	uint w = state ^ (state >> 22);
	w = w * 1664525u + 1013904223u;
	return w;
}

static float rng_float(thread uint& state) {
	return float(rng_next(state) & 0x00FFFFFFu) / float(0x01000000u);
}

static float rng_float_range(thread uint& state, float lo, float hi) {
	return lo + (hi - lo) * rng_float(state);
}

static float3 rng_in_unit_sphere(thread uint& state) {
	for (;;) {
		float3 p = float3(rng_float_range(state, -1.0, 1.0),
		                  rng_float_range(state, -1.0, 1.0),
		                  rng_float_range(state, -1.0, 1.0));
		if (length_squared(p) < 1.0) return p;
	}
}

static float3 rng_in_unit_disk(thread uint& state) {
	for (;;) {
		float3 p = float3(rng_float_range(state, -1.0, 1.0),
		                  rng_float_range(state, -1.0, 1.0), 0.0);
		if (length_squared(p) < 1.0) return p;
	}
}

static float3 rng_unit_vector(thread uint& state) {
	return normalize(rng_in_unit_sphere(state));
}

// ── Utility functions ───────────────────────────────────────────────────────

static bool near_zero(float3 v) {
	float eps = 1.0e-8;
	return fabs(v.x) < eps && fabs(v.y) < eps && fabs(v.z) < eps;
}

static float schlick_reflectance(float cosine, float ref_idx) {
	float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
	r0 *= r0;
	return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

static float luminance(float3 c) {
	return c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
}

// ── Ray tracing kernel (triangle AS, built-in intersection) ─────────────────

kernel void raytraceKernel(
	uint2                              tid      [[thread_position_in_grid]],
	constant GPUSceneData&             scene    [[buffer(0)]],
	device const GPUMaterial*          materials [[buffer(1)]],
	device float4*                     output   [[buffer(2)]],
	primitive_acceleration_structure   accel    [[buffer(3)]],
	device const TriVertex*            vertices [[buffer(4)]],
	device const uint*                 indices  [[buffer(5)]],
	device const uint*                 light_prims [[buffer(6)]]
) {
	if (tid.x >= uint(scene.image_width) ||
	    tid.y >= uint(scene.image_height)) return;

	uint pixel_idx = tid.y * uint(scene.image_width) + tid.x;
	uint seed = scene.seed + pixel_idx;

	float3 pixel_color = 0.0;

	for (int s = 0; s < scene.samples_per_pixel; s++) {
		float u = (float(tid.x) + rng_float(seed)) / float(scene.image_width - 1);
		float v = (float(tid.y) + rng_float(seed)) / float(scene.image_height - 1);

		float3 ro = scene.origin.xyz;
		float3 rd = scene.lower_left.xyz
			+ u * scene.horizontal.xyz
			+ v * scene.vertical.xyz
			- ro;

		// Depth of Field
		if (scene.lens_radius > 0.0) {
			float3 rd_disk = rng_in_unit_disk(seed) * scene.lens_radius;
			float3 offset = rd_disk.x * scene.u.xyz + rd_disk.y * scene.v.xyz;
			ro += offset;
			rd -= offset;
		}

		ray r;
		r.origin = ro;
		r.direction = rd;
		r.min_distance = 0.001;
		r.max_distance = INFINITY;

		float3 ray_color = 1.0;
		float3 accumulated = 0.0;

		for (int depth = 0; depth < scene.max_depth; depth++) {
			// Metal's built-in triangle intersector
			intersector<> i;
			i.assume_geometry_type(geometry_type::triangle);
			auto result = i.intersect(r, accel);

			if (result.type == intersection_type::none || result.distance >= INFINITY || result.distance <= 0.0) {
				float3 unit_dir = normalize(r.direction);
				float t = 0.5 * (unit_dir.y + 1.0);
				accumulated += ray_color * ((1.0 - t) * float3(1.0) + t * float3(0.5, 0.7, 1.0));
				break;
			}

			uint pid = result.primitive_id;
			float hit_dist = result.distance;
			float3 hit_point = r.origin + hit_dist * r.direction;

			// Interpolate vertex positions for normal calculation
			uint base_idx = pid * 3;
			uint i0 = indices[base_idx];
			uint i1 = indices[base_idx + 1];
			uint i2 = indices[base_idx + 2];

			float3 p0 = vertices[i0].position;
			float3 p1 = vertices[i1].position;
			float3 p2 = vertices[i2].position;

			// Compute face normal from world-space triangle positions
			float3 edge1 = p1 - p0;
			float3 edge2 = p2 - p0;
			float3 face_normal = normalize(cross(edge1, edge2));

			// Determine front face
			bool front_face = dot(r.direction, face_normal) < 0.0;
			float3 shading_normal = front_face ? face_normal : -face_normal;

			GPUMaterial mat = materials[pid];

			// Debug modes: output diagnostic data on first bounce
			if (depth == 0 && scene.debug_mode > 0) {
				if (scene.debug_mode == 1) {
					// Albedo
					accumulated = mat.albedo;
				} else if (scene.debug_mode == 2) {
					// Normal (mapped to [0,1])
					accumulated = shading_normal * 0.5 + 0.5;
				} else if (scene.debug_mode == 3) {
					// Depth (normalized)
					float d = result.distance / 20.0;
					accumulated = float3(d, d, d);
				} else if (scene.debug_mode == 4) {
					// Primitive ID (color-coded)
					float r = float(pid & 0xFF) / 255.0;
					float g = float((pid >> 8) & 0xFF) / 255.0;
					float b = float((pid >> 16) & 0xFF) / 255.0;
					accumulated = float3(r, g, b);
				}
				break;
			}

			if (mat.kind == 4) {
				// Emissive
				accumulated += ray_color * mat.emission * mat.emission_strength;
				break;
			}

			// Direct light sampling (Next-Event Estimation)
			if (mat.kind == 0 || mat.kind == 3) {
				for (uint li = 0; li < uint(scene.light_count); li++) {
					uint lpid = light_prims[li];
					GPUMaterial lmat = materials[lpid];
					if (lmat.kind != 4) continue; // not emissive

					uint lbase = lpid * 3;
					uint li0 = indices[lbase];
					uint li1 = indices[lbase + 1];
					uint li2 = indices[lbase + 2];
					float3 lp0 = vertices[li0].position;
					float3 lp1 = vertices[li1].position;
					float3 lp2 = vertices[li2].position;

					// Random point on light triangle (barycentric)
					float r1 = rng_float(seed);
					float r2 = rng_float(seed);
					float sqrt_r1 = sqrt(r1);
					float b0 = 1.0 - sqrt_r1;
					float b1 = sqrt_r1 * (1.0 - r2);
					float b2 = sqrt_r1 * r2;
					float3 light_pos = b0 * lp0 + b1 * lp1 + b2 * lp2;

					// Direction to light
					float3 to_light = light_pos - hit_point;
					float light_dist = length(to_light);
					float3 light_dir = to_light / light_dist;

					// Surface must face the light
					float cos_surf = dot(shading_normal, light_dir);
					if (cos_surf <= 0.0) continue;

					// Light must face the surface
					float3 ledge1 = lp1 - lp0;
					float3 ledge2 = lp2 - lp0;
					float3 light_normal = normalize(cross(ledge1, ledge2));
					// Ensure light normal faces toward the hit point
					if (dot(light_normal, light_dir) > 0.0) {
						light_normal = -light_normal;
					}
					float light_area = 0.5 * length(cross(ledge1, ledge2));
					float cos_light = dot(-light_dir, light_normal);
					if (cos_light <= 0.0) continue;

					// Shadow ray
					ray shadow_ray;
					shadow_ray.origin = hit_point;
					shadow_ray.direction = light_dir;
					shadow_ray.min_distance = 0.001;
					shadow_ray.max_distance = light_dist - 0.001;

					intersector<> si;
					auto sresult = si.intersect(shadow_ray, accel);
					if (sresult.type == intersection_type::none) {
						// Not occluded — add direct lighting
						float3 le = lmat.emission * lmat.emission_strength;
						float pdf = light_dist * light_dist / (light_area * cos_light);
						float3 direct = le * cos_surf / pdf;
						accumulated += ray_color * mat.albedo * direct;
					}
				}
			}

			// Scatter
			if (mat.kind == 0 || mat.kind == 3) {
				// Lambertian / Principled (diffuse fallback)
				float3 scatter_dir = shading_normal + rng_unit_vector(seed);
				if (near_zero(scatter_dir)) scatter_dir = shading_normal;
				r.origin = hit_point;
				r.direction = scatter_dir;
				r.min_distance = 0.001;
				ray_color *= mat.albedo;
			} else if (mat.kind == 1) {
				// Metal
				float3 reflected = reflect(r.direction, shading_normal);
				float fuzz = min(mat.fuzz, 1.0);
				r.origin = hit_point;
				r.direction = reflected + fuzz * rng_in_unit_sphere(seed);
				r.min_distance = 0.001;
				if (dot(r.direction, shading_normal) <= 0.0) {
					accumulated = 0.0;
					break;
				}
				ray_color *= mat.albedo;
			} else if (mat.kind == 2) {
				// Dielectric
				float ir = mat.ir;
				float refraction_ratio = front_face ? (1.0 / ir) : ir;
				float3 unit_dir = normalize(r.direction);
				float cos_theta = min(dot(-unit_dir, shading_normal), 1.0);
				float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
				bool cannot_refract = refraction_ratio * sin_theta > 1.0;
				float3 dir;
				if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > rng_float(seed)) {
					dir = reflect(unit_dir, shading_normal);
				} else {
					dir = refract(unit_dir, shading_normal, refraction_ratio);
				}
				r.origin = hit_point;
				r.direction = dir;
				r.min_distance = 0.001;
			}

			// Firefly clamping
			if (scene.max_radiance > 0.0) {
				float lum = luminance(ray_color);
				if (lum > scene.max_radiance) {
					ray_color *= (scene.max_radiance / lum);
				}
			}
		}

		pixel_color += accumulated;
	}

	pixel_color /= float(scene.samples_per_pixel);
	float3 final_color = float3(sqrt(pixel_color.x), sqrt(pixel_color.y), sqrt(pixel_color.z));
	output[pixel_idx] = float4(final_color, 1.0);
}
