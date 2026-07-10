#include <metal_stdlib>
using namespace metal;

// ── Edge-avoiding À-Trous wavelet denoiser ──────────────────────────────────
//
// Dammertz et al. 2010, "Edge-Avoiding À-Trous Wavelet Transform for fast
// Global Illumination Filtering". A separable 5-tap B3-spline kernel is
// applied at geometrically increasing step widths (1, 2, 4, ...). Edge
// stopping functions on color, world normal, and view depth preserve
// silhouettes and shading discontinuities while smoothing Monte-Carlo noise.
//
// The host runs this kernel once per iteration, ping-ponging color_in and
// color_out. The normal/depth guides come from dedicated first-hit render
// passes and stay fixed across iterations.

inline float dn_luminance(float3 c) {
	return dot(c, float3(0.2126, 0.7152, 0.0722));
}

struct DenoiseParams {
	int   width;
	int   height;
	int   step_width;   // À-Trous hole size for this iteration (1, 2, 4, ...)
	float sigma_color;  // color edge-stop strength (larger = more blur)
	float sigma_normal; // normal edge-stop strength
	float sigma_depth;  // depth edge-stop strength
};

kernel void atrousDenoiseKernel(
	constant DenoiseParams&     p          [[buffer(0)]],
	device const float4*        color_in   [[buffer(1)]],
	device float4*              color_out  [[buffer(2)]],
	device const float4*        normal_in  [[buffer(3)]],
	device const float4*        depth_in   [[buffer(4)]],
	uint2                       gid        [[thread_position_in_grid]])
{
	int x = int(gid.x);
	int y = int(gid.y);
	if (x >= p.width || y >= p.height) return;

	int idx = y * p.width + x;

	// B3-spline 5-tap kernel weights.
	const float kernel_w[5] = { 1.0/16.0, 1.0/4.0, 3.0/8.0, 1.0/4.0, 1.0/16.0 };

	float3 c_center = color_in[idx].xyz;
	float  l_center = dn_luminance(c_center);
	float3 n_center = normal_in[idx].xyz;
	float  d_center = depth_in[idx].x;

	// Guard against tiny sigmas (avoid divide-by-zero → NaN weights).
	float inv_sc = 1.0 / max(p.sigma_color,  1.0e-4);
	float inv_sn = 1.0 / max(p.sigma_normal, 1.0e-4);
	float inv_sd = 1.0 / max(p.sigma_depth,  1.0e-4);

	float3 sum   = float3(0.0);
	float  sum_w = 0.0;

	for (int dy = -2; dy <= 2; dy++) {
		for (int dx = -2; dx <= 2; dx++) {
			int sx = clamp(x + dx * p.step_width, 0, p.width  - 1);
			int sy = clamp(y + dy * p.step_width, 0, p.height - 1);
			int sidx = sy * p.width + sx;

			float3 c = color_in[sidx].xyz;
			float3 n = normal_in[sidx].xyz;
			float  d = depth_in[sidx].x;

			// Color edge stop (luminance-based). Filtering runs in
			// demodulated illumination space, where per-channel division by a
			// low albedo channel amplifies noise; comparing luminance instead
			// of raw RGB keeps that amplified single-channel noise from
			// over-preserving and blocking the blur.
			float  dl = dn_luminance(c) - l_center;
			float  w_c = exp(-(dl * dl) * inv_sc);

			// Normal edge stop.
			float3 dn = n - n_center;
			float  dist_n = dot(dn, dn);
			float  w_n = exp(-dist_n * inv_sn);

			// Depth edge stop.
			float  dd = d - d_center;
			float  w_d = exp(-(dd * dd) * inv_sd);

			float h = kernel_w[dx + 2] * kernel_w[dy + 2];
			float w = h * w_c * w_n * w_d;

			sum   += c * w;
			sum_w += w;
		}
	}

	float3 result = (sum_w > 0.0) ? (sum / sum_w) : c_center;
	color_out[idx] = float4(result, color_in[idx].w);
}
