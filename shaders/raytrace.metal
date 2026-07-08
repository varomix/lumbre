#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant int DIRECT_LIGHT_SAMPLES = 4;
constant float PI = 3.14159265358979323846;
constant float INV_PI = 0.31830988618379067154;

// ── GPU data types ───────────────────────────────────────────────────────────

struct GPUMaterial {
	float4 albedo;
	float4 emission;
	float4 params0; // kind, fuzz, ir, roughness
	float4 params1; // metallic, emission_strength, unused, unused
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
	int    tri_light_count;
	int    primitive_count;
	uint   seed;
	int    quad_light_count;
	int    sphere_light_count;
	float  roughness_cutoff;
};

struct TriVertex {
	float4 position;
	float4 normal;
};

struct GPULightTriangle {
	float4 p0;
	float4 p1;
	float4 p2;
	float4 emission;
};

struct GPUQuadLight {
	float4 position;
	float4 u;
	float4 v;
	float4 emission;
};

struct GPUSphereLight {
	float4 position;
	float4 emission;
	float  radius;
	float  _pad[3];
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

static float3 reinhard_tonemap(float3 c) {
	return c / (float3(1.0) + c);
}

static float power_heuristic(float pdf_a, float pdf_b) {
	float a2 = pdf_a * pdf_a;
	float b2 = pdf_b * pdf_b;
	return a2 / max(a2 + b2, 1.0e-12);
}

static float3 make_tangent(float3 n) {
	float3 helper = fabs(n.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
	return normalize(cross(helper, n));
}

static float3 cosine_sample_hemisphere(float3 n, thread uint& state, thread float& pdf) {
	float r1 = rng_float(state);
	float r2 = rng_float(state);
	float phi = 2.0 * PI * r1;
	float radius = sqrt(r2);
	float x = radius * cos(phi);
	float y = radius * sin(phi);
	float z = sqrt(max(1.0 - r2, 0.0));

	float3 tangent = make_tangent(n);
	float3 bitangent = cross(n, tangent);
	float3 dir = normalize(x * tangent + y * bitangent + z * n);
	pdf = max(dot(n, dir), 0.0) * INV_PI;
	return dir;
}

static float triangle_area(float3 p0, float3 p1, float3 p2) {
	return 0.5 * length(cross(p1 - p0, p2 - p0));
}

static float light_pdf_solid_angle(float dist2, float cos_light, float area, int light_count) {
	if (light_count <= 0 || cos_light <= 0.0 || area <= 0.0) {
		return 0.0;
	}
	return dist2 / (cos_light * area * float(light_count));
}

static float3 sample_quad_light(GPUQuadLight light, float3 from_point, thread uint& seed, thread float3& light_pos, thread float& light_dist, thread float3& light_normal, thread float& pdf) {
	float r1 = rng_float(seed);
	float r2 = rng_float(seed);
	float3 pos = light.position.xyz + r1 * light.u.xyz + r2 * light.v.xyz;
	light_normal = normalize(cross(light.u.xyz, light.v.xyz));
	float3 to_light = pos - from_point;
	light_dist = length(to_light);
	float3 dir = to_light / light_dist;
	float cos_light = max(fabs(dot(light_normal, -dir)), 0.001);
	float area = length(cross(light.u.xyz, light.v.xyz));
	pdf = (light_dist * light_dist) / (cos_light * area);
	return dir;
}

static float3 sample_sphere_light(GPUSphereLight light, float3 from_point, thread uint& seed, thread float3& light_pos, thread float& light_dist, thread float3& light_normal, thread float& pdf) {
	float3 center = light.position.xyz;
	float radius = max(light.radius, 0.001);
	float3 to_center = center - from_point;
	float center_dist = length(to_center);
	float3 dir_to_center = to_center / center_dist;

	float sin_theta_max = radius / center_dist;
	float cos_theta_max = sqrt(max(1.0 - sin_theta_max * sin_theta_max, 0.0));

	float eps1 = rng_float(seed);
	float eps2 = rng_float(seed);
	float cos_theta = 1.0 - eps1 * (1.0 - cos_theta_max);
	float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
	float phi = 2.0 * PI * eps2;

	float3 helper = fabs(dir_to_center.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
	float3 tangent = normalize(cross(helper, dir_to_center));
	float3 bitangent = cross(dir_to_center, tangent);

	float3 local_dir = cos_theta * dir_to_center + sin_theta * (cos(phi) * tangent + sin(phi) * bitangent);

	light_dist = center_dist * cos_theta - sqrt(max(radius * radius - center_dist * center_dist * sin_theta * sin_theta, 0.0));
	light_pos = from_point + light_dist * local_dir;
	light_normal = normalize(light_pos - center);
	float3 surface_to_eye = -local_dir;
	if (dot(light_normal, surface_to_eye) < 0.0) light_normal = -light_normal;

	pdf = 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
	return local_dir;
}

static float3 emissive_radiance(GPUMaterial mat) {
	float3 color = mat.emission.xyz;
	if (luminance(color) <= 0.0) {
		color = mat.albedo.xyz;
	}

	float strength = mat.params1.y;
	if (strength <= 0.0) {
		strength = 20.0;
	}
	return color * strength;
}

// ── Ray tracing kernel (triangle AS, built-in intersection) ─────────────────

kernel void raytraceKernel(
	uint2                              tid            [[thread_position_in_grid]],
	constant GPUSceneData&             scene          [[buffer(0)]],
	device const GPUMaterial*          materials      [[buffer(1)]],
	device float4*                     output         [[buffer(2)]],
	primitive_acceleration_structure   accel          [[buffer(3)]],
	device const TriVertex*            vertices       [[buffer(4)]],
	device const uint*                 indices        [[buffer(5)]],
	device const GPULightTriangle*     tri_lights     [[buffer(6)]],
	device const GPUQuadLight*         quad_lights    [[buffer(7)]],
	device const GPUSphereLight*       sphere_lights  [[buffer(8)]],
	device const int*                  mat_indices    [[buffer(9)]]
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
		float last_bsdf_pdf = 0.0;
		bool last_was_delta = false;

		for (int depth = 0; depth < scene.max_depth; depth++) {
			// Metal's built-in triangle intersector
			intersector<> i;
			i.assume_geometry_type(geometry_type::triangle);
			auto result = i.intersect(r, accel);

			if (result.type == intersection_type::none || result.distance >= INFINITY || result.distance <= 0.0) {
				if (scene.tri_light_count + scene.quad_light_count + scene.sphere_light_count == 0) {
					float3 unit_dir = normalize(r.direction);
					float t = 0.5 * (unit_dir.y + 1.0);
					accumulated += ray_color * ((1.0 - t) * float3(1.0) + t * float3(0.5, 0.7, 1.0));
				}
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

			float3 p0 = vertices[i0].position.xyz;
			float3 p1 = vertices[i1].position.xyz;
			float3 p2 = vertices[i2].position.xyz;

			float3 edge1 = p1 - p0;
			float3 edge2 = p2 - p0;
			float3 geom_normal = normalize(cross(edge1, edge2));

			// Determine front face
			bool front_face = dot(r.direction, geom_normal) < 0.0;
			float3 shading_normal = front_face ? geom_normal : -geom_normal;

			int midx = mat_indices[pid];
			GPUMaterial mat = materials[midx];
			int mat_kind = int(mat.params0.x);

			// Roughness cutoff: treat high-roughness materials as fully diffuse
			if (scene.roughness_cutoff > 0.0 && mat_kind == 3) {
				if (mat.params0.w > scene.roughness_cutoff) {
					mat_kind = 0;
				}
			}

			// Debug modes: output diagnostic data on first bounce
			if (depth == 0 && scene.debug_mode > 0) {
				if (scene.debug_mode == 1) {
					// Albedo
					accumulated = mat.albedo.xyz;
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
				} else if (scene.debug_mode == 5) {
					// Direct light contribution only.
					accumulated = 0.0;
				} else if (scene.debug_mode == 6) {
					// Light count visible to the shader.
					float c = min(float(scene.tri_light_count) / 4.0, 1.0);
					accumulated = float3(c, 0.0, 0.0);
				} else if (scene.debug_mode == 7) {
					// Direct light candidate geometry before shadowing.
					accumulated = 0.0;
				} else if (scene.debug_mode == 8) {
					// Shadow visibility only: green means unoccluded.
					accumulated = 0.0;
				}
				if (scene.debug_mode != 5 && scene.debug_mode != 7 && scene.debug_mode != 8) break;
			}

			if (mat_kind == 4) {
				// Emissive
				float weight = 1.0;
				if (depth > 0 && !last_was_delta && scene.tri_light_count > 0) {
					float light_area = triangle_area(p0, p1, p2);
					float cos_light = max(fabs(dot(geom_normal, -normalize(r.direction))), 0.0);
					float light_pdf = light_pdf_solid_angle(hit_dist * hit_dist, cos_light, light_area, scene.tri_light_count);
					weight = power_heuristic(last_bsdf_pdf, light_pdf);
				}
				accumulated += ray_color * emissive_radiance(mat) * weight;
				break;
			}

			// Direct light sampling (Next-Event Estimation)
			if (mat_kind == 0 || mat_kind == 3) {
				int total_lights = scene.tri_light_count + scene.quad_light_count + scene.sphere_light_count;
				if (total_lights > 0) {
					int light_types_sampled = 0;

					// Triangle lights
					if (scene.tri_light_count > 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							uint li = min(uint(rng_float(seed) * float(scene.tri_light_count)), uint(scene.tri_light_count - 1));
							GPULightTriangle ltri = tri_lights[li];
							float3 lp0 = ltri.p0.xyz;
							float3 lp1 = ltri.p1.xyz;
							float3 lp2 = ltri.p2.xyz;

							float r1 = rng_float(seed);
							float r2 = rng_float(seed);
							float sqrt_r1 = sqrt(r1);
							float3 light_pos = (1.0 - sqrt_r1) * lp0 + (sqrt_r1 * (1.0 - r2)) * lp1 + (sqrt_r1 * r2) * lp2;

							float3 to_light = light_pos - hit_point;
							float light_dist = length(to_light);
							float3 light_dir = to_light / light_dist;
							float3 light_normal = normalize(cross(lp1 - lp0, lp2 - lp0));
							float light_area = triangle_area(lp0, lp1, lp2);
							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							float cos_light = max(fabs(dot(light_normal, -light_dir)), 0.0);

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float dist2 = max(light_dist * light_dist, 1.0e-6);
								float light_pdf = light_pdf_solid_angle(dist2, cos_light, light_area, scene.tri_light_count);
								float bsdf_pdf = cos_surf * INV_PI;
								if (light_pdf > 0.0 && bsdf_pdf > 0.0) {
									float mis_weight = power_heuristic(light_pdf, bsdf_pdf);
									float3 brdf = mat.albedo.xyz * INV_PI;
									float3 direct = ltri.emission.xyz * brdf * cos_surf * mis_weight / light_pdf;
									accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
								}
							}
						}
					}

					// Quad lights
					if (scene.quad_light_count > 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							uint li = min(uint(rng_float(seed) * float(scene.quad_light_count)), uint(scene.quad_light_count - 1));
							GPUQuadLight ql = quad_lights[li];

							float3 light_pos, light_normal;
							float light_dist, light_pdf_val;
							float3 light_dir = sample_quad_light(ql, hit_point, seed, light_pos, light_dist, light_normal, light_pdf_val);
							light_pdf_val /= float(scene.quad_light_count);

							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							if (cos_surf <= 0.0 || light_pdf_val <= 0.0) continue;

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float bsdf_pdf = cos_surf * INV_PI;
								if (bsdf_pdf > 0.0) {
									float mis_weight = power_heuristic(light_pdf_val, bsdf_pdf);
									float3 brdf = mat.albedo.xyz * INV_PI;
									float3 direct = ql.emission.xyz * brdf * cos_surf * mis_weight / light_pdf_val;
									accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
								}
							}
						}
					}

					// Sphere lights
					if (scene.sphere_light_count > 0) {
						light_types_sampled++;
						for (int ls = 0; ls < DIRECT_LIGHT_SAMPLES; ls++) {
							uint li = min(uint(rng_float(seed) * float(scene.sphere_light_count)), uint(scene.sphere_light_count - 1));
							GPUSphereLight sl = sphere_lights[li];

							float3 light_pos, light_normal;
							float light_dist, light_pdf_val;
							float3 light_dir = sample_sphere_light(sl, hit_point, seed, light_pos, light_dist, light_normal, light_pdf_val);
							light_pdf_val /= float(scene.sphere_light_count);

							float cos_surf = max(dot(shading_normal, light_dir), 0.0);
							if (cos_surf <= 0.0 || light_pdf_val <= 0.0) continue;

							ray shadow_ray;
							shadow_ray.origin = hit_point + light_dir * 0.002;
							shadow_ray.direction = light_dir;
							shadow_ray.min_distance = 0.002;
							shadow_ray.max_distance = max(light_dist - 0.004, 0.0);

							intersector<> si;
							si.assume_geometry_type(geometry_type::triangle);
							auto sresult = si.intersect(shadow_ray, accel);
							if (sresult.type == intersection_type::none) {
								float bsdf_pdf = cos_surf * INV_PI;
								if (bsdf_pdf > 0.0) {
									float mis_weight = power_heuristic(light_pdf_val, bsdf_pdf);
									float3 brdf = mat.albedo.xyz * INV_PI;
									float3 direct = sl.emission.xyz * brdf * cos_surf * mis_weight / light_pdf_val;
									accumulated += ray_color * direct / float(DIRECT_LIGHT_SAMPLES);
								}
							}
						}
					}
				}
			}

			if (
				depth == 0 &&
				(scene.debug_mode == 5 ||
				 scene.debug_mode == 7 ||
				 scene.debug_mode == 8)
			) {
				break;
			}

			// Scatter
			if (mat_kind == 0 || mat_kind == 3) {
				// Lambertian / Principled (diffuse fallback)
				float bsdf_pdf = 0.0;
				float3 scatter_dir = cosine_sample_hemisphere(shading_normal, seed, bsdf_pdf);
				if (near_zero(scatter_dir) || bsdf_pdf <= 0.0) scatter_dir = shading_normal;
				r.origin = hit_point;
				r.direction = scatter_dir;
				r.min_distance = 0.001;
				ray_color *= mat.albedo.xyz;
				last_bsdf_pdf = max(bsdf_pdf, 0.0);
				last_was_delta = false;
			} else if (mat_kind == 1) {
				// Metal
				float3 reflected = reflect(r.direction, shading_normal);
				float fuzz = min(mat.params0.y, 1.0);
				r.origin = hit_point;
				r.direction = reflected + fuzz * rng_in_unit_sphere(seed);
				r.min_distance = 0.001;
				if (dot(r.direction, shading_normal) <= 0.0) {
					accumulated = 0.0;
					break;
				}
				ray_color *= mat.albedo.xyz;
				last_bsdf_pdf = 0.0;
				last_was_delta = true;
			} else if (mat_kind == 2) {
				// Dielectric
				float ir = mat.params0.z;
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
				last_bsdf_pdf = 0.0;
				last_was_delta = true;
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
	pixel_color = reinhard_tonemap(pixel_color);
	float3 final_color = float3(sqrt(pixel_color.x), sqrt(pixel_color.y), sqrt(pixel_color.z));
	output[pixel_idx] = float4(final_color, 1.0);
}
