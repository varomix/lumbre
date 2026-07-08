#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

// ── GPU data types (f32) ────────────────────────────────────────────────────

struct SphereGPU {
	float4 center;
	float  radius;
	int    material_kind;
	float  fuzz;
	float  ir;
	float4 albedo;
};

struct SceneData {
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
	uint   seed;
	float  _pad[2];
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

static float3 rng_vec3_range(thread uint& state, float lo, float hi) {
	return float3(rng_float_range(state, lo, hi),
	              rng_float_range(state, lo, hi),
	              rng_float_range(state, lo, hi));
}

static float3 rng_in_unit_sphere(thread uint& state) {
	for (;;) {
		float3 p = rng_vec3_range(state, -1.0, 1.0);
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

// ── Custom sphere intersection function ─────────────────────────────────────

struct BoundingBoxResult {
	bool   accept  [[accept_intersection]];
	float  distance [[distance]];
};

[[intersection(bounding_box)]]
BoundingBoxResult sphereIntersection(
	float3               origin          [[origin]],
	float3               direction       [[direction]],
	float                minDistance     [[min_distance]],
	float                maxDistance     [[max_distance]],
	device const float4* sphere_data    [[buffer(5)]],
	unsigned int         primitiveIndex  [[primitive_id]]
) {
	float3 center = sphere_data[primitiveIndex * 3 + 0].xyz;
	float  radius = sphere_data[primitiveIndex * 3 + 1].x;

	float3 oc = origin - center;
	float a = dot(direction, direction);
	float half_b = dot(oc, direction);
	float c = dot(oc, oc) - radius * radius;
	float disc = half_b * half_b - a * c;

	if (disc < 0.0) return { false, 0.0 };

	float sqrtd = sqrt(disc);
	float root = (-half_b - sqrtd) / a;
	if (root <= minDistance || maxDistance <= root) {
		root = (-half_b + sqrtd) / a;
		if (root <= minDistance || maxDistance <= root) return { false, 0.0 };
	}

	return { true, root };
}

// ── Ray tracing kernel ──────────────────────────────────────────────────────

kernel void raytraceKernel(
	uint2                              tid   [[thread_position_in_grid]],
	constant SceneData&                scene [[buffer(0)]],
	device const SphereGPU*            spheres [[buffer(1)]],
	device float4*                     output [[buffer(2)]],
	primitive_acceleration_structure   accel  [[buffer(3)]],
	intersection_function_table<>      ift    [[buffer(4)]]
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

		for (int depth = 0; depth < scene.max_depth; depth++) {
			intersector<> i;
			auto result = i.intersect(r, accel, ift);

			if (result.type == intersection_type::none) {
				float3 unit_dir = normalize(r.direction);
				float t = 0.5 * (unit_dir.y + 1.0);
				ray_color *= (1.0 - t) * float3(1.0) + t * float3(0.5, 0.7, 1.0);
				break;
			}

			uint pid = result.primitive_id;
			float hit_dist = result.distance;
			float3 hit_point = r.origin + hit_dist * r.direction;
			float3 normal = normalize(hit_point - spheres[pid].center.xyz);
			float3 outward_normal = normal;
			float3 attenuation = spheres[pid].albedo.xyz;
			int kind = spheres[pid].material_kind;

			// Determine front face
			bool front_face = dot(r.direction, outward_normal) < 0.0;
			float3 facing_normal = front_face ? outward_normal : -outward_normal;

			if (kind == 0) {
				// Lambertian
				float3 scatter_dir = facing_normal + rng_unit_vector(seed);
				if (near_zero(scatter_dir)) scatter_dir = facing_normal;
				r.origin = hit_point;
				r.direction = scatter_dir;
				r.min_distance = 0.001;
				ray_color *= attenuation;
			} else if (kind == 1) {
				// Metal
				float3 reflected = reflect(r.direction, facing_normal);
				float fuzz = min(spheres[pid].fuzz, 1.0);
				r.origin = hit_point;
				r.direction = reflected + fuzz * rng_in_unit_sphere(seed);
				r.min_distance = 0.001;
				if (dot(r.direction, facing_normal) <= 0.0) {
					ray_color = 0.0;
					break;
				}
				ray_color *= attenuation;
			} else if (kind == 2) {
				// Dielectric
				float ir = spheres[pid].ir;
				float refraction_ratio = front_face ? (1.0 / ir) : ir;
				float3 unit_dir = normalize(r.direction);
				float cos_theta = min(dot(-unit_dir, facing_normal), 1.0);
				float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
				bool cannot_refract = refraction_ratio * sin_theta > 1.0;
				float3 dir;
				if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > rng_float(seed)) {
					dir = reflect(unit_dir, facing_normal);
				} else {
					dir = refract(unit_dir, facing_normal, refraction_ratio);
				}
				r.origin = hit_point;
				r.direction = dir;
				r.min_distance = 0.001;
			}
		}

		pixel_color += ray_color;
	}

	pixel_color /= float(scene.samples_per_pixel);
	// Gamma correction (gamma = 2.0)
	float3 final_color = float3(sqrt(pixel_color.x), sqrt(pixel_color.y), sqrt(pixel_color.z));

	output[pixel_idx] = float4(final_color, 1.0);
}
