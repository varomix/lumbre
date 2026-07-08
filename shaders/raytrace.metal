#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant int DIRECT_LIGHT_SAMPLES = 4;
constant float PI = 3.14159265358979323846;
constant float INV_PI = 0.31830988618379067154;
constant int GI_CACHE_MAX_POINTS = 262144;
constant int GI_GRID_SIZE = 32768; // power of 2
constant int GI_MAX_PER_CELL = 16;
constant int GI_CACHE_MIN_SAMPLES = 4;
constant float GI_CACHE_MAX_REL_STDDEV = 0.75;

constant int PHOTON_MAX_COUNT = 1048576;
constant int PHOTON_GRID_SIZE = 16384;
constant int PHOTON_MAX_PER_CELL = 32;
constant int PHOTON_MAX_BOUNCES = 8;
constant float PHOTON_SEARCH_RADIUS = 1.0;

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
	int    gi_cache_enabled;
	float  gi_cache_distance;
	float  gi_cache_normal_angle;
	int    gi_cache_num_points;
	int    photon_enabled;
	int    photon_count;
	float  photon_radius;
	int    photon_max_bounces;
};

struct GICachePoint {
	float4 position;   // xyz = world position, w = grid cell x
	float4 normal;     // xyz = shading normal, w = grid cell y
	float4 irradiance; // xyz = cached irradiance, w = grid cell z
};

struct Photon {
	float4 position; // xyz = world position, w = grid cell x
	float4 incident; // xyz = incident direction, w = grid cell y
	float4 power;    // xyz = photon power, w = grid cell z
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

static float3 debug_heat(float v) {
	float x = clamp(v, 0.0, 1.0);
	return clamp(float3(2.0 * x, 2.0 * (1.0 - fabs(x - 0.5) * 2.0), 2.0 * (1.0 - x)) - 0.5, 0.0, 1.0);
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

	// ── Photon hash grid ───────────────────────────────────────────────────────

static int photon_hash_cell(float3 cell) {
	uint h = (uint(as_type<int>(cell.x)) * 73856093u) ^
	         (uint(as_type<int>(cell.y)) * 19349663u) ^
	         (uint(as_type<int>(cell.z)) * 83492791u);
	return int(h & uint(PHOTON_GRID_SIZE - 1));
}

static float3 photon_query(
	device const Photon* photons,
	device const int* grid_cells,
	device const atomic_int* grid_counts,
	float3 pos, float3 normal,
	float radius, float3 albedo
) {
	if (radius <= 0.0) return float3(0.0);

	float3 base_cell = floor(pos / radius);
	float3 accum = 0.0;
	float inv_area = 1.0 / (PI * radius * radius);

	for (int ix = -1; ix <= 1; ix++) {
		for (int iy = -1; iy <= 1; iy++) {
			for (int iz = -1; iz <= 1; iz++) {
				float3 ncell = base_cell + float3(float(ix), float(iy), float(iz));
				int cell = photon_hash_cell(ncell);
				int count = min(atomic_load_explicit(&grid_counts[cell], memory_order_relaxed), PHOTON_MAX_PER_CELL);

				for (int j = 0; j < count; j++) {
					int pi = grid_cells[cell * PHOTON_MAX_PER_CELL + j];
					Photon ph = photons[pi];

					float3 ph_cell = float3(ph.position.w, ph.incident.w, ph.power.w);
					if (any(ph_cell != ncell)) continue;

					float3 delta = ph.position.xyz - pos;
					float dist = length(delta);
					if (dist > radius) continue;

					float normal_sim = max(dot(normal, ph.incident.xyz), 0.0);
					if (normal_sim <= 0.0) continue;

					float w = max(1.0 - dist / radius, 0.0);
					w = w * w * normal_sim;
					float3 brdf = albedo * INV_PI;
					accum += brdf * ph.power.xyz * w * inv_area;
				}
			}
		}
	}
	return accum;
}

	// ── Irradiance Cache (hash grid + deferred write) ────────────────────────────

static int gi_hash_cell(float3 cell) {
	uint h = (uint(as_type<int>(cell.x)) * 73856093u) ^
	         (uint(as_type<int>(cell.y)) * 19349663u) ^
	         (uint(as_type<int>(cell.z)) * 83492791u);
	return int(h & uint(GI_GRID_SIZE - 1));
}

static float3 gi_cache_query(
	device const GICachePoint* cache,
	device const int* grid_cells,
	device const atomic_int* grid_counts,
	int num_cells,
	float3 pos,
	float3 normal,
	float max_dist,
	float normal_angle,
	thread int& sample_count,
	thread float& rel_stddev
) {
	sample_count = 0;
	rel_stddev = 1.0e6;
	if (num_cells <= 0 || max_dist <= 0.0) return float3(0.0);

	float3 base_cell = floor(pos / max_dist);

	float3 result = 0.0;
	float total_weight = 0.0;
	float lum_sum = 0.0;
	float lum2_sum = 0.0;

	for (int ix = -1; ix <= 1; ix++) {
		for (int iy = -1; iy <= 1; iy++) {
			for (int iz = -1; iz <= 1; iz++) {
				float3 ncell = base_cell + float3(float(ix), float(iy), float(iz));
				int cell = gi_hash_cell(ncell);
				int count = min(atomic_load_explicit(&grid_counts[cell], memory_order_relaxed), GI_MAX_PER_CELL);

				for (int j = 0; j < count; j++) {
					int ci = grid_cells[cell * GI_MAX_PER_CELL + j];
					GICachePoint cp = cache[ci];

					float3 cp_cell = float3(cp.position.w, cp.normal.w, cp.irradiance.w);
					if (any(cp_cell != ncell)) continue;

					float3 delta = cp.position.xyz - pos;
					float dist = length(delta);
					if (dist > max_dist) continue;

					float normal_sim = max(dot(normal, cp.normal.xyz), 0.0);
					if (normal_sim < normal_angle) continue;

					float radial = max(1.0 - dist / max_dist, 0.0);
					float w = radial * radial * normal_sim * normal_sim;
					result += cp.irradiance.xyz * w;
					total_weight += w;
					float lum = luminance(cp.irradiance.xyz);
					lum_sum += lum * w;
					lum2_sum += lum * lum * w;
					sample_count++;
				}
			}
		}
	}

	if (total_weight > 0.0) {
		float mean_lum = lum_sum / total_weight;
		float mean_lum2 = lum2_sum / total_weight;
		float variance = max(mean_lum2 - mean_lum * mean_lum, 0.0);
		rel_stddev = sqrt(variance) / max(mean_lum, 1.0e-4);
		return result / total_weight;
	}
	return float3(0.0);
}

static void gi_cache_store(
	device GICachePoint* cache,
	device atomic_int* counter,
	device int* grid_cells,
	device atomic_int* grid_counts,
	float cell_size,
	float3 pos,
	float3 normal,
	float3 irradiance
) {
	int idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
	idx = idx % GI_CACHE_MAX_POINTS;

	float3 cell_coord = floor(pos / cell_size);
	cache[idx].position = float4(pos, cell_coord.x);
	cache[idx].normal = float4(normal, cell_coord.y);
	cache[idx].irradiance = float4(irradiance, cell_coord.z);

	int cell = gi_hash_cell(cell_coord);
	int count = atomic_fetch_add_explicit(&grid_counts[cell], 1, memory_order_relaxed);
	if (count < GI_MAX_PER_CELL) {
		grid_cells[cell * GI_MAX_PER_CELL + count] = idx;
	}
}

static void gi_cache_deferred_write(
	device GICachePoint* cache,
	device atomic_int* counter,
	device int* grid_cells,
	device atomic_int* grid_counts,
	float cell_size,
	thread int& cache_pending,
	thread float3& cache_p_pos,
	thread float3& cache_p_normal,
	thread float3& cache_p_throughput,
	thread float3& cache_p_accum_before,
	float3 accumulated
) {
	if (!cache_pending) return;
	float3 delta = accumulated - cache_p_accum_before;
	float3 throughput = max(cache_p_throughput, 1e-8);
	float3 irradiance = delta * PI / throughput;
	if (luminance(irradiance) > 0.0) {
		gi_cache_store(cache, counter, grid_cells, grid_counts,
			cell_size,
			cache_p_pos, cache_p_normal, irradiance);
	}
	cache_pending = 0;
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
	device const int*                  mat_indices    [[buffer(9)]],
	device GICachePoint*               gi_cache       [[buffer(10)]],
	device atomic_int*                 gi_counter     [[buffer(11)]],
	device int*                        gi_grid_cells  [[buffer(12)]],
	device atomic_int*                 gi_grid_counts [[buffer(13)]],
	device const Photon*               photons        [[buffer(14)]],
	device const atomic_int*           photon_counter [[buffer(15)]],
	device const int*                  photon_grid_cells  [[buffer(16)]],
	device const atomic_int*           photon_grid_counts [[buffer(17)]]
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
		bool debug_found = false;

		// Deferred irradiance cache state
		int cache_pending = 0;
		float3 cache_p_pos;
		float3 cache_p_normal;
		float3 cache_p_throughput;
		float3 cache_p_accum_before; // captures accumulated BEFORE NEE at the cache bounce

		for (int depth = 0; depth < scene.max_depth; depth++) {
			// Metal's built-in triangle intersector
			intersector<> i;
			i.assume_geometry_type(geometry_type::triangle);
			auto result = i.intersect(r, accel);

			if (result.type == intersection_type::none || result.distance >= INFINITY || result.distance <= 0.0) {
				float3 unit_dir = normalize(r.direction);
				float t = 0.5 * (unit_dir.y + 1.0);
				accumulated += ray_color * ((1.0 - t) * float3(1.0) + t * float3(0.5, 0.7, 1.0));
				gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
					scene.gi_cache_distance,
					cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
					accumulated);
				break;
			}

			uint pid = result.primitive_id;
			float hit_dist = result.distance;
			float3 hit_point = r.origin + hit_dist * r.direction;

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
			bool front_face = dot(r.direction, geom_normal) < 0.0;

			// Smooth shading via barycentric interpolation of vertex normals
			float3 edge_cross = cross(edge1, edge2);
			float full_area = length(edge_cross);
			float bu = length(cross(p1 - hit_point, p2 - hit_point)) / max(full_area, 1e-12);
			float bv = length(cross(p2 - hit_point, p0 - hit_point)) / max(full_area, 1e-12);
			float bw = length(cross(p0 - hit_point, p1 - hit_point)) / max(full_area, 1e-12);
			float sum = bu + bv + bw;
			bu /= sum; bv /= sum; bw /= sum;

			float3 vn0 = vertices[i0].normal.xyz;
			float3 vn1 = vertices[i1].normal.xyz;
			float3 vn2 = vertices[i2].normal.xyz;
			float3 shading_normal = normalize(bu * vn0 + bv * vn1 + bw * vn2);
			if (!front_face) shading_normal = -shading_normal;

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
				if (
					scene.debug_mode != 5 &&
					scene.debug_mode != 7 &&
					scene.debug_mode != 8 &&
					scene.debug_mode != 9 &&
					scene.debug_mode != 10 &&
					scene.debug_mode != 11 &&
					scene.debug_mode != 12 &&
					scene.debug_mode != 13
				) break;
			}

			// Snapshot accumulated before NEE for deferred cache write
			float3 accum_at_hit = accumulated;

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
				gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
					scene.gi_cache_distance,
					cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
					accumulated);
				break;
			}

			// Direct light sampling (Next-Event Estimation)
			if ((mat_kind == 0 || mat_kind == 3) && !(scene.debug_mode == 9 && depth == 0)) {
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
				// Irradiance cache: on indirect diffuse bounce, try lookup.
				// Skip cache for rays coming from a specular bounce to avoid
				// noisy cached irradiance showing up in mirror-like reflections.
				if (scene.gi_cache_enabled && depth > 1 && !last_was_delta) {
					int cache_sample_count = 0;
					float cache_rel_stddev = 1.0e6;
					float3 cached = gi_cache_query(
						gi_cache, gi_grid_cells, gi_grid_counts,
						GI_GRID_SIZE,
						hit_point, shading_normal,
						scene.gi_cache_distance,
						scene.gi_cache_normal_angle,
						cache_sample_count,
						cache_rel_stddev
					);
					float cached_lum = luminance(cached);
					bool cache_accepted =
						cached_lum > 0.0 &&
						cache_sample_count >= GI_CACHE_MIN_SAMPLES &&
						cache_rel_stddev <= GI_CACHE_MAX_REL_STDDEV;

					if (scene.debug_mode == 12) {
						accumulated = cache_sample_count > 0 ?
							debug_heat(float(cache_sample_count) / float(GI_CACHE_MIN_SAMPLES * 2)) :
							float3(0.5, 0.0, 0.0);
						debug_found = true;
						ray_color = 0.0;
						cache_pending = 0;
						break;
					}

					if (scene.debug_mode == 13) {
						float confidence = 1.0 - clamp(cache_rel_stddev / GI_CACHE_MAX_REL_STDDEV, 0.0, 1.0);
						accumulated = cache_sample_count > 0 ? debug_heat(confidence) : float3(0.5, 0.0, 0.0);
						debug_found = true;
						ray_color = 0.0;
						cache_pending = 0;
						break;
					}

					if (cache_accepted) {
						if (scene.debug_mode == 10) {
							accumulated = debug_heat(cached_lum * 0.1);
							debug_found = true;
							ray_color = 0.0;
							cache_pending = 0;
							break;
						}
						accumulated += ray_color * cached * mat.albedo.xyz * INV_PI;
						ray_color = 0.0;
						break;
					}
				}

				// Photon mapping: add photon contribution at diffuse bounces
				if (scene.photon_enabled && depth > 1) {
					int pc = min(int(atomic_load_explicit(photon_counter, memory_order_relaxed)), PHOTON_MAX_COUNT);
					if (pc > 0) {
						float3 photon_contrib = photon_query(
							photons, photon_grid_cells, photon_grid_counts,
							hit_point, shading_normal,
							scene.photon_radius, mat.albedo.xyz
						);
						if (luminance(photon_contrib) > 0.0) {
							if (scene.debug_mode == 11) {
								accumulated = debug_heat(luminance(photon_contrib) * 0.5);
								debug_found = true;
								ray_color = 0.0;
								cache_pending = 0;
								break;
							}
							accumulated += ray_color * photon_contrib;
						}
					}
				}

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

				// Mark deferred cache point (accum_when_hit includes this bounce's NEE)
				if (scene.gi_cache_enabled && depth > 1) {
					cache_p_pos = hit_point;
					cache_p_normal = shading_normal;
					cache_p_throughput = ray_color;
					cache_p_accum_before = accum_at_hit;
					cache_pending = 1;
				}
			} else if (mat_kind == 1) {
				// Metal
				float3 reflected = reflect(r.direction, shading_normal);
				float fuzz = min(mat.params0.y, 1.0);
				r.origin = hit_point;
				r.direction = reflected + fuzz * rng_in_unit_sphere(seed);
				r.min_distance = 0.001;
				if (dot(r.direction, shading_normal) <= 0.0) {
					accumulated = 0.0;
					gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
						scene.gi_cache_distance,
						cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
						accumulated);
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
		// Path reached max depth — flush any deferred cache point
		gi_cache_deferred_write(gi_cache, gi_counter, gi_grid_cells, gi_grid_counts,
			scene.gi_cache_distance,
			cache_pending, cache_p_pos, cache_p_normal, cache_p_throughput, cache_p_accum_before,
			accumulated);

		if (
			(scene.debug_mode == 10 ||
			 scene.debug_mode == 11 ||
			 scene.debug_mode == 12 ||
			 scene.debug_mode == 13) &&
			!debug_found
		) {
			accumulated = 0.0;
		}

		pixel_color += accumulated;
	}

	pixel_color /= float(scene.samples_per_pixel);
	pixel_color = reinhard_tonemap(pixel_color);
	float3 final_color = float3(sqrt(pixel_color.x), sqrt(pixel_color.y), sqrt(pixel_color.z));
	output[pixel_idx] = float4(final_color, 1.0);
}

// ── Photon Emission Kernel ──────────────────────────────────────────────────

kernel void photonEmitKernel(
	uint                               tid            [[thread_position_in_grid]],
	constant GPUSceneData&             scene          [[buffer(0)]],
	device const GPUMaterial*          materials      [[buffer(1)]],
	primitive_acceleration_structure   accel          [[buffer(3)]],
	device const TriVertex*            vertices       [[buffer(4)]],
	device const uint*                 indices        [[buffer(5)]],
	device const GPULightTriangle*     tri_lights     [[buffer(6)]],
	device const GPUQuadLight*         quad_lights    [[buffer(7)]],
	device const GPUSphereLight*       sphere_lights  [[buffer(8)]],
	device const int*                  mat_indices    [[buffer(9)]],
	device Photon*                     photons        [[buffer(14)]],
	device atomic_int*                 photon_counter [[buffer(15)]]
) {
	if (tid >= uint(scene.photon_count) || scene.photon_count <= 0) return;

	uint seed = scene.seed + tid * 1973u;

	int tlc = scene.tri_light_count;
	int qlc = scene.quad_light_count;
	int slc = scene.sphere_light_count;
	int total_lights = tlc + qlc + slc;
	if (total_lights == 0) return;

	int li = int(tid) % total_lights;

	float3 origin, lnormal, emission;
	float area = 0.0;

	if (li < tlc) {
		GPULightTriangle lt = tri_lights[li];
		float r1 = rng_float(seed);
		float r2 = rng_float(seed);
		float sqrt_r1 = sqrt(r1);
		origin = (1.0 - sqrt_r1) * lt.p0.xyz + (sqrt_r1 * (1.0 - r2)) * lt.p1.xyz + (sqrt_r1 * r2) * lt.p2.xyz;
		lnormal = normalize(cross(lt.p1.xyz - lt.p0.xyz, lt.p2.xyz - lt.p0.xyz));
		emission = lt.emission.xyz;
		area = triangle_area(lt.p0.xyz, lt.p1.xyz, lt.p2.xyz);
	} else if (li < tlc + qlc) {
		GPUQuadLight ql = quad_lights[li - tlc];
		float r1 = rng_float(seed);
		float r2 = rng_float(seed);
		origin = ql.position.xyz + r1 * ql.u.xyz + r2 * ql.v.xyz;
		lnormal = normalize(cross(ql.u.xyz, ql.v.xyz));
		emission = ql.emission.xyz;
		area = length(cross(ql.u.xyz, ql.v.xyz));
	} else {
		GPUSphereLight sl = sphere_lights[li - tlc - qlc];
		float3 dir = rng_unit_vector(seed);
		origin = sl.position.xyz + dir * sl.radius;
		lnormal = dir;
		emission = sl.emission.xyz;
		area = 4.0 * PI * sl.radius * sl.radius;
	}

	// Cosine-weighted emission direction
	float pdf_dir;
	float3 dir = cosine_sample_hemisphere(lnormal, seed, pdf_dir);

	float3 power = emission * (PI * area * float(total_lights)) / float(scene.photon_count);

	ray r;
	r.origin = origin + dir * 0.002;
	r.direction = dir;
	r.min_distance = 0.001;
	r.max_distance = INFINITY;

	for (int bounce = 0; bounce < scene.photon_max_bounces; bounce++) {
		intersector<> i;
		i.assume_geometry_type(geometry_type::triangle);
		auto result = i.intersect(r, accel);

		if (result.type == intersection_type::none || result.distance >= INFINITY || result.distance <= 0.0) break;

		uint pid = result.primitive_id;
		float hit_dist = result.distance;
		float3 hit_point = r.origin + hit_dist * r.direction;

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
		bool front_face = dot(r.direction, geom_normal) < 0.0;

		float3 edge_cross = cross(edge1, edge2);
		float full_area = length(edge_cross);
		float bu = length(cross(p1 - hit_point, p2 - hit_point)) / max(full_area, 1e-12);
		float bv = length(cross(p2 - hit_point, p0 - hit_point)) / max(full_area, 1e-12);
		float bw = length(cross(p0 - hit_point, p1 - hit_point)) / max(full_area, 1e-12);
		float sum = bu + bv + bw;
		bu /= sum; bv /= sum; bw /= sum;

		float3 vn0 = vertices[i0].normal.xyz;
		float3 vn1 = vertices[i1].normal.xyz;
		float3 vn2 = vertices[i2].normal.xyz;
		float3 shading_normal = normalize(bu * vn0 + bv * vn1 + bw * vn2);
		if (!front_face) shading_normal = -shading_normal;

		int midx = mat_indices[pid];
		GPUMaterial mat = materials[midx];
		int mat_kind = int(mat.params0.x);

		if (scene.roughness_cutoff > 0.0 && mat_kind == 3) {
			if (mat.params0.w > scene.roughness_cutoff) {
				mat_kind = 0;
			}
		}

		if (mat_kind == 0 || mat_kind == 3) {
			// Diffuse — store photon
			int idx = atomic_fetch_add_explicit(photon_counter, 1, memory_order_relaxed);
			if (idx < PHOTON_MAX_COUNT) {
				float3 cell_coord = floor(hit_point / scene.photon_radius);
				photons[idx].position = float4(hit_point, cell_coord.x);
				photons[idx].incident = float4(-r.direction, cell_coord.y);
				photons[idx].power = float4(power, cell_coord.z);
			}
			break;
		} else if (mat_kind == 1) {
			// Metal — reflect
			float3 reflected = reflect(r.direction, shading_normal);
			float fuzz = min(mat.params0.y, 1.0);
			r.origin = hit_point;
			r.direction = reflected + fuzz * rng_in_unit_sphere(seed);
			r.min_distance = 0.001;
			if (dot(r.direction, shading_normal) <= 0.0) break;
			power *= mat.albedo.xyz;
		} else if (mat_kind == 2) {
			// Dielectric — transmit / reflect
			float ir = mat.params0.z;
			float refraction_ratio = front_face ? (1.0 / ir) : ir;
			float3 unit_dir = normalize(r.direction);
			float cos_theta = min(dot(-unit_dir, shading_normal), 1.0);
			float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
			bool cannot_refract = refraction_ratio * sin_theta > 1.0;
			float3 dir_out;
			if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > rng_float(seed)) {
				dir_out = reflect(unit_dir, shading_normal);
			} else {
				dir_out = refract(unit_dir, shading_normal, refraction_ratio);
			}
			r.origin = hit_point;
			r.direction = dir_out;
			r.min_distance = 0.001;
		} else if (mat_kind == 4) {
			break;
		}

		// Russian roulette
		float lum = luminance(power);
		if (lum < 0.001) {
			if (rng_float(seed) > lum / 0.001) break;
			power *= 0.001 / lum;
		}
	}
}

// ── Photon Grid Build Kernel ────────────────────────────────────────────────

static int ph_grid_cell(float3 pos, float radius) {
	float3 cell = floor(pos / radius);
	uint h = (uint(as_type<int>(cell.x)) * 73856093u) ^
	         (uint(as_type<int>(cell.y)) * 19349663u) ^
	         (uint(as_type<int>(cell.z)) * 83492791u);
	return int(h & uint(PHOTON_GRID_SIZE - 1));
}

kernel void photonBuildGridKernel(
	uint                               tid            [[thread_position_in_grid]],
	device const Photon*               photons        [[buffer(0)]],
	device const atomic_int*           photon_counter [[buffer(1)]],
	device int*                        grid_cells     [[buffer(2)]],
	device atomic_int*                 grid_counts    [[buffer(3)]],
	constant GPUSceneData&             scene          [[buffer(4)]]
) {
	int count = min(int(atomic_load_explicit(photon_counter, memory_order_relaxed)), PHOTON_MAX_COUNT);
	if (tid >= uint(count) || scene.photon_radius <= 0.0) return;

	float3 pos = photons[tid].position.xyz;
	int cell = ph_grid_cell(pos, scene.photon_radius);
	int c = atomic_fetch_add_explicit(&grid_counts[cell], 1, memory_order_relaxed);
	if (c < PHOTON_MAX_PER_CELL) {
		grid_cells[cell * PHOTON_MAX_PER_CELL + c] = int(tid);
	}
}
