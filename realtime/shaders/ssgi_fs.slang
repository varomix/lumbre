// Screen-space indirect diffuse light (SSGI), the gather: what the G-buffer's
// near field gives a surface, in place of what the probes or the environment
// guessed for those same directions.
//
// Probes are a cell apart. Light that varies faster than that -- the dark
// band at the foot of a block standing on a floor, where the block hides the
// ceiling's light from the floor in front of it -- they smear out. This pass
// takes those directions from the screen.
//
// The slices are GTAO's (ao_fs): planes containing the view direction, at
// angles across the screen, with the normal projected into each. Along a
// slice the hemisphere is cut into 32 sectors of equal cosine weight, so a
// sector's bit means the same share of irradiance wherever it lies (the
// visibility bitmask of Therrien, Levesque and Gilet 2023, with the sectors
// cosine-weighted rather than equal in angle). Each depth sample is taken as
// a slab THICKNESS deep, which occludes the sectors between its front and
// its back. Sectors no nearer sample took yet swap what the probes say
// arrives from the sample's direction for the sample's radiance in the lit
// image. Sectors nothing takes keep what the probes or the environment said.
//
// The probes are read along each sample's direction with the weights the
// lighting pass gave them, so a sector whose screen radiance matches what the
// probes saw there changes nothing. Scaling the probes' irradiance by the
// open share instead assumes every direction brought the average: near a
// dark ceiling the open directions are the brighter ones, and that darkened
// every seam of a room lit by bounce. Without probes the environment's
// irradiance about the direction stands in.
//
// Out: rgb = the change to the irradiance the lighting pass used for the
// diffuse ambient, which it had darkened by GTAO:
//
//     irradiance * (1 - ao) + sum over taken sectors of (screen - probes)
//
// so lit + albedo * kd * that is the diffuse ambient this pass computes.
// a = the open share, for inspection. The 4x4 rotation pattern is averaged
// out by ssgi_apply_fs.
#include "shading.slang"

struct SsgiUniforms {
    float4x4 inv_view_proj;
    float4 eye;     // xyz, w = radius in world units
    float4 forward; // xyz = view axis, w = world size of a pixel at unit depth
    float4 right;   // xyz = the camera's right, w = a sample's thickness, world units
    float4 up;      // xyz = the camera's up, w unused
    float4 params;  // AO drawn this frame, 1/width, 1/height, TAA frame
    float4 env;     // has_env, rotation, intensity, scene_color's mip levels
    // Baked probes, as in LightingUniforms; probe_min.w = 1 when in use.
    float4 probe_min;
    float4 probe_cell;
    float4 probe_dims;
};

[[vk::binding(0, 2)]] Sampler2D normal_target;
[[vk::binding(1, 2)]] Sampler2D depth_target;
[[vk::binding(2, 2)]] Sampler2D scene_color; // the lit opaque image, trilinear over its mips
[[vk::binding(3, 2)]] Sampler2D ao_target;
[[vk::binding(4, 2)]] Sampler2D env_irradiance;
[[vk::binding(5, 2)]] Sampler2D surface_target;
// Before the storage buffers: Slang numbers Metal buffers in declaration
// order, and SDL_GPU looks for uniforms first (see lighting_fs).
[[vk::binding(0, 3)]] ConstantBuffer<SsgiUniforms> gi;
[[vk::binding(6, 2)]] StructuredBuffer<float4> probe_irradiance_map;
[[vk::binding(7, 2)]] StructuredBuffer<float2> probe_depth;
[[vk::binding(8, 2)]] StructuredBuffer<float4> probe_radiance_map;
[[vk::binding(9, 2)]] StructuredBuffer<float4> probe_info;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

static const int SLICES = 2;
static const int STEPS = 8; // per side
static const uint SECTORS = 32;
static const float HALF_PI = 1.57079633;

float3 world_at(float2 uv, float depth) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 h = mul(gi.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

// The cosine-weighted measure of the slice from angle 0 (the view direction)
// to `theta`, signed so that it rises monotonically across the slice: the
// integral of |sin t| cos(t - gamma), GTAO's integrand, in closed form.
float slice_measure(float theta, float gamma, float cos_n, float sin_g) {
    float g = (cos_n + 2.0 * theta * sin_g - cos(2.0 * theta - gamma)) * 0.25;
    return theta < 0.0 ? -g : g;
}

// Sectors [start, end) of the slice's 32.
uint sector_mask(uint start, uint end) {
    if (end <= start) return 0u;
    uint count = end - start;
    return count >= SECTORS ? 0xffffffffu : ((1u << count) - 1u) << start;
}

// What the lighting pass's ambient sources say arrives along `d`: the probes'
// radiance maps under the lighting pass's weights, or the environment's
// irradiance about d (its light from that side, blurred), or nothing.
float3 ambient_along(float3 d, ProbeWeights pw, bool use_probes) {
    if (use_probes) {
        float3 sum = float3(0.0);
        [unroll]
        for (int k = 0; k < 8; ++k) {
            if (pw.w[k] <= 0.0) continue;
            sum += pw.w[k] * probe_tap_rgb(probe_radiance_map, probe_tap_res(pw.probe[k], d, PROBE_RADIANCE_RES));
        }
        return sum / pw.sum;
    }
    if (gi.env.x > 0.5) {
        return env_irradiance.SampleLevel(direction_to_equirect(rotate_env(d, gi.env.y)), 0.0).rgb * gi.env.z;
    }
    return float3(0.0);
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float depth = depth_target.SampleLevel(i.uv, 0.0).r;
    if (depth <= 0.0) return float4(0.0, 0.0, 0.0, 1.0); // background

    float3 p = world_at(i.uv, depth);
    float3 n = decode_normal(normal_target.SampleLevel(i.uv, 0.0).rg);
    float3 v = normalize(gi.eye.xyz - p);
    if (dot(n, v) < 0.0) n = -n; // the face the viewer sees, as lighting does

    // The diffuse irradiance the lighting pass used, from the same sources
    // and weights (see ibl_with_probes), and how far GTAO darkened it.
    ProbeWeights pw;
    pw.sum = 0.0;
    float3 irradiance = float3(0.0);
    if (gi.probe_min.w > 0.5) {
        ProbeGrid grid;
        grid.lo = gi.probe_min;
        grid.cell = gi.probe_cell;
        grid.dims = gi.probe_dims;
        pw = probe_weights(p, n, v, grid, probe_depth, probe_info);
        if (pw.sum > 1e-5) {
            [unroll]
            for (int k = 0; k < 8; ++k) {
                if (pw.w[k] > 0.0) irradiance += pw.w[k] * probe_tile_irradiance(probe_irradiance_map, pw.probe[k], n);
            }
            irradiance /= pw.sum;
        }
    }
    bool use_probes = pw.sum > 1e-5;
    if (!use_probes && gi.env.x > 0.5) {
        irradiance = env_irradiance.SampleLevel(direction_to_equirect(rotate_env(n, gi.env.y)), 0.0).rgb * gi.env.z;
    }
    float ao = gi.params.x > 0.5 ? ao_target.SampleLevel(i.uv, 0.0).r : 1.0;
    float3 undo_ao = irradiance * (1.0 - ao);

    float radius = gi.eye.w;
    float thickness = gi.right.w;
    float view_z = max(dot(p - gi.eye.xyz, gi.forward.xyz), 1e-6);
    float radius_px = radius / (view_z * gi.forward.w);
    // Too far to reach a neighbouring pixel: nothing is found, and GTAO's
    // darkening is all that is undone.
    if (radius_px < 1.0) return float4(undo_ao, 1.0);
    float2 texel = gi.params.yz;

    // The 4x4 ordered pattern ao_fs uses, turned per TAA frame so TAA
    // averages more directions than one frame holds.
    uint frame = uint(gi.params.w);
    uint2 px = (uint2(i.pos.xy) + uint2(frame, frame * 3u)) & 3u;
    const float bayer[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float slice_jitter = (bayer[px.y * 4 + px.x] + 0.5) / 16.0;
    float step_jitter = (bayer[(px.x * 4 + px.y) ^ 5u] + 0.5) / 16.0;
    // Where a sector boundary falls against a sample's edge: dithered, so a
    // sliver of a sector counts in proportion on average.
    float sector_jitter = (bayer[(px.y * 4 + px.x) ^ 10u] + 0.5) / 16.0;

    float visible = 0.0;
    float3 change = float3(0.0);
    float total = 0.0;
    [unroll]
    for (int s = 0; s < SLICES; ++s) {
        float phi = (float(s) + slice_jitter) * (3.14159265 / float(SLICES));
        float2 omega = float2(cos(phi), sin(phi));
        float3 dir = gi.right.xyz * omega.x - gi.up.xyz * omega.y;
        float3 ortho = dir - v * dot(dir, v);
        float3 ortho_n = normalize(ortho);
        float3 axis = normalize(cross(ortho, v));
        float3 proj_n = n - axis * dot(n, axis);
        float proj_len = length(proj_n);
        if (proj_len < 1e-4) continue;
        float cos_n = saturate(dot(proj_n, v) / proj_len);
        float gamma = (dot(ortho, proj_n) < 0.0 ? -1.0 : 1.0) * acos(cos_n);
        float sin_g = sin(gamma);

        // The visible half of the slice runs from gamma - pi/2 to gamma + pi/2.
        float lo = gamma - HALF_PI;
        float hi = gamma + HALF_PI;
        float measure_lo = slice_measure(lo, gamma, cos_n, sin_g);
        float measure = slice_measure(hi, gamma, cos_n, sin_g) - measure_lo;
        if (measure <= 1e-6) continue;

        uint taken = 0u;
        float3 slice_change = float3(0.0);
        [unroll]
        for (int side = 0; side < 2; ++side) {
            // Positive angles lie along +omega on screen; see ao_fs, whose
            // horizon0 (the +omega side) becomes its positive h1.
            float sign = side == 0 ? 1.0 : -1.0;
            [unroll]
            for (int k = 0; k < STEPS; ++k) {
                float t = (float(k) + step_jitter) / float(STEPS);
                float offset = max(t * t * radius_px, float(k + 1));
                // The spacing between this step and the next, in pixels: the
                // patch of screen this sample stands for. Its radiance is read
                // that blurred, or a small, bright source between steps is
                // either missed or counted for the whole gap, which streaked
                // the floor around a lit post.
                float spacing = 2.0 * t * radius_px / float(STEPS);
                float level = clamp(log2(max(spacing, 1.0)), 0.0, max(gi.env.w - 1.0, 0.0));
                float2 uv = i.uv + sign * omega * offset * texel;
                if (any(uv < 0.0) || any(uv > 1.0)) break;
                float d = depth_target.SampleLevel(uv, 0.0).r;
                if (d <= 0.0) continue;

                float3 q = world_at(uv, d);
                float3 delta = q - p;
                float dist = length(delta);
                if (dist > radius || dist < 1e-6) continue;
                // On or below p's own plane (3 degrees of slack): the floor
                // around a point on it. It hides nothing, but its edge, right
                // at the horizon, could round into the lowest sector, and a
                // pool of light on the floor then lit its surroundings in
                // blocks.
                if (dot(delta, n) < 0.05 * dist) continue;
                // The sample as a slab: its front, and its back THICKNESS
                // further from the eye.
                float3 back = q + normalize(q - gi.eye.xyz) * thickness - p;
                float front_angle = sign * acos(clamp(dot(delta, v) / dist, -1.0, 1.0));
                float back_angle = sign * acos(clamp(dot(back, v) / max(length(back), 1e-6), -1.0, 1.0));
                float a = clamp(min(front_angle, back_angle), lo, hi);
                float b = clamp(max(front_angle, back_angle), lo, hi);
                float ua = (slice_measure(a, gamma, cos_n, sin_g) - measure_lo) / measure;
                float ub = (slice_measure(b, gamma, cos_n, sin_g) - measure_lo) / measure;
                uint start = uint(clamp(floor(ua * float(SECTORS) + sector_jitter), 0.0, float(SECTORS)));
                uint end = uint(clamp(floor(ub * float(SECTORS) + sector_jitter), 0.0, float(SECTORS)));
                uint mask = sector_mask(start, end);
                uint fresh = mask & ~taken;
                taken |= mask;
                if (fresh == 0u) continue;

                // The sample's outgoing light, as the camera sees it,
                // standing in for what it sends toward p: the same at any
                // angle, so a wall seen edge-on from a corner counts in
                // full. How far it can be trusted for that:
                //  - Only a surface facing p sends it this light. One facing
                //    away hides another side from the camera -- a crate's
                //    back, seen from the wall behind it -- whose light the
                //    screen cannot say, so those sectors keep the probes'.
                //    The ramp only softens precision at the edge.
                //  - Only light a surface scatters every way, not its
                //    reflection, which shows the camera what the camera's
                //    side of the room sends it: a mirror wall seen dark would
                //    darken every corner beside it, and a glossy floor seen
                //    low, mostly Fresnel, brightened the feet of what stands
                //    on it. The diffuse share is ibl_with_probes' kd at the
                //    camera's angle; a metal has none.
                float3 nq = decode_normal(normal_target.SampleLevel(uv, 0.0).rg);
                float3 to_eye = normalize(gi.eye.xyz - q);
                if (dot(nq, to_eye) < 0.0) nq = -nq;
                float facing = saturate(dot(nq, -delta / dist) * 8.0 + 1.0);
                float4 finish = surface_target.SampleLevel(uv, 0.0);
                float f0 = 0.08 * finish.b;
                float fresnel = f0 + (max(1.0 - finish.r, f0) - f0) * pow5(1.0 - saturate(dot(nq, to_eye)));
                float diffuse_share = (1.0 - fresnel) * (1.0 - finish.g);
                float trust = facing * diffuse_share;
                if (trust <= 0.0) continue;
                float3 screen = scene_color.SampleLevel(uv, level).rgb;
                float3 probes = ambient_along(delta / dist, pw, use_probes);
                float share = float(countbits(fresh)) / float(SECTORS);
                slice_change += (screen - probes) * (trust * share);
            }
        }
        float weight = proj_len * measure;
        visible += weight * (1.0 - float(countbits(taken)) / float(SECTORS));
        change += weight * slice_change;
        total += weight;
    }
    if (total <= 1e-6) return float4(undo_ao, 1.0);
    return float4(undo_ao + change / total, visible / total);
}
