// Screen-space reflections: one fullscreen pass over the lit opaque image.
//
// The lighting pass has already given every surface a specular reflection of
// the ENVIRONMENT -- the prefiltered map along the reflection vector. That is
// right for a surface that sees the sky and wrong for one that sees the room.
// This pass traces the same reflection vector through the depth buffer and,
// where it lands on something on screen, swaps the environment's contribution
// for the lit colour found there. Where it lands on nothing, the pixel passes
// through and keeps the environment: that is the whole fallback.
//
// What it cannot do, all inherent to reading the screen: reflect anything
// outside the frame, behind the camera, or hidden behind a nearer surface;
// reflect glass, which is drawn after this pass; or blur. A rough surface needs
// a cone of rays, not one, so the reflection fades out with roughness and the
// prefiltered environment takes over again. There is one bounce: a mirror seen
// in a mirror shows the environment.
//
// The march is hierarchical. hiz_fs builds a pyramid of nearest depths, and a
// ray that is nearer the eye than a cell's nearest surface crosses the whole
// cell in one step, climbing a level each time it does and dropping one when a
// cell might hold a hit. Depth is affine along a screen-space line, so the ray
// is a straight segment in (pixel x, pixel y, depth) and every crossing is
// solved exactly rather than stepped toward.

#include "shading.slang"

struct SsrUniforms {
    float4x4 inv_view_proj;
    float4x4 view_proj;
    float4 eye;     // xyz, w = how far a ray is traced, world units
    float4 forward; // xyz = view axis, w = world size of a pixel at unit depth
    float4 env;     // has_env, rotation, intensity, specular mip count
    float4 size;    // width, height, 1/width, 1/height
    float4 depth;   // near, far, pyramid levels, thickness in world units
    float4 params;  // AO drawn this frame, show the reflection alone, unused, unused
    // Baked probes, as in LightingUniforms; probe_min.w = 1 when in use.
    float4 probe_min;
    float4 probe_cell;
    float4 probe_dims;
};

[[vk::binding(0, 2)]] Sampler2D scene_color; // lit opaque image, linear-filtered
[[vk::binding(1, 2)]] Sampler2D albedo_target;
[[vk::binding(2, 2)]] Sampler2D normal_target;
[[vk::binding(3, 2)]] Sampler2D surface_target;
// Point-sampled at texel centres, one level at a time; see hiz_fs for why not
// Load.
[[vk::binding(4, 2)]] Sampler2D hiz;
[[vk::binding(5, 2)]] Sampler2D env_specular;
[[vk::binding(6, 2)]] Sampler2D env_brdf_lut;
[[vk::binding(7, 2)]] Sampler2D ao_target;
[[vk::binding(0, 3)]] ConstantBuffer<SsrUniforms> ssr;
// What the lighting pass reflected where this pass finds nothing on screen:
// with probes, their radiance rather than the sky's.
[[vk::binding(8, 2)]] StructuredBuffer<float4> probe_irradiance_map;
[[vk::binding(9, 2)]] StructuredBuffer<float2> probe_depth;
[[vk::binding(10, 2)]] StructuredBuffer<float4> probe_radiance_map;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

static const int MAX_STEPS = 96;
// Reflections are sharp, so they are only right on smooth surfaces: full
// strength below the first roughness, gone by the second.
static const float ROUGHNESS_FULL = 0.2;
static const float ROUGHNESS_NONE = 0.45;

// Reversed-Z depth to distance along the view axis.
float linear_depth(float d) {
    float near = ssr.depth.x;
    float far = ssr.depth.y;
    return near * far / max(near + d * (far - near), 1e-12);
}

// A world point as (pixel x, pixel y, depth). Only called in front of the eye.
float3 to_screen(float3 p) {
    float4 clip = mul(ssr.view_proj, float4(p, 1.0));
    float3 ndc = clip.xyz / clip.w;
    return float3((ndc.x * 0.5 + 0.5) * ssr.size.x, (0.5 - ndc.y * 0.5) * ssr.size.y, ndc.z);
}

// How far a pixel's surface depth changes across one pixel, in raw depth. A
// plane's depth is affine across the screen, so it is the step to a neighbour:
// the smaller one on each axis, so a silhouette's jump is not taken for slope.
float depth_step(int2 px, float centre) {
    int2 last = int2(ssr.size.xy) - 1;
    float2 texel = ssr.size.zw;
    float l = hiz.SampleLevel((float2(clamp(px + int2(-1, 0), int2(0), last)) + 0.5) * texel, 0.0).r;
    float r = hiz.SampleLevel((float2(clamp(px + int2(1, 0), int2(0), last)) + 0.5) * texel, 0.0).r;
    float u = hiz.SampleLevel((float2(clamp(px + int2(0, -1), int2(0), last)) + 0.5) * texel, 0.0).r;
    float d = hiz.SampleLevel((float2(clamp(px + int2(0, 1), int2(0), last)) + 0.5) * texel, 0.0).r;
    return min(abs(l - centre), abs(r - centre)) + min(abs(u - centre), abs(d - centre));
}

// Marches the segment `from` -> `to` through the depth pyramid. Returns whether
// it hit, where in pixels, and how far to trust the hit.
bool trace(float3 from, float3 to, out float2 hit_px, out float confidence) {
    hit_px = float2(0.0);
    confidence = 1.0;
    float3 d = to - from;
    float span = max(abs(d.x), abs(d.y));
    // A ray along the line of sight covers no pixels; there is nothing to walk.
    if (span < 1.0) return false;

    // The exit side of a cell in each axis, and the reciprocal slope. A ray
    // running along a row or column never crosses a boundary in the other
    // axis, and its crossing must come out as "never" whichever side of the
    // boundary it starts on -- hence a zero slope and an added infinity, not a
    // large slope that the subtraction can turn negative.
    float2 toward = float2(d.x > 0.0 ? 1.0 : 0.0, d.y > 0.0 ? 1.0 : 0.0);
    bool2 flat = abs(d.xy) <= 1e-6;
    float2 inv = select(flat, float2(0.0), 1.0 / d.xy);
    float2 never = select(flat, float2(1e30), float2(0.0));
    // Far enough past a boundary to be in the next cell, in units of t.
    float nudge = 0.01 / span;

    int top = int(ssr.depth.z) - 1;
    int2 full = int2(ssr.size.xy);
    float thickness = ssr.depth.w;

    // Leave the pixel the ray starts in before testing anything.
    float2 cell0 = floor(from.xy);
    float2 exit0 = ((cell0 + toward) - from.xy) * inv + never;
    float t = max(min(exit0.x, exit0.y), 0.0) + nudge;
    int level = 0;

    for (int iter = 0; iter < MAX_STEPS; ++iter) {
        if (t >= 1.0) return false;
        float2 p = from.xy + d.xy * t;
        if (p.x < 0.0 || p.y < 0.0 || p.x >= ssr.size.x || p.y >= ssr.size.y) return false;

        float cell_size = float(1 << level);
        float2 cell = floor(p / cell_size);
        // An odd dimension's last row or column belongs to the last cell.
        int2 dims = max(full >> level, int2(1));
        float2 cell_uv = (float2(min(int2(cell), dims - 1)) + 0.5) / float2(dims);
        float nearest = hiz.SampleLevel(cell_uv, float(level)).r;

        float2 exit = ((cell + toward) * cell_size - from.xy) * inv + never;
        float t_exit = min(exit.x, exit.y);

        float z_in = from.z + d.z * t;
        float z_out = from.z + d.z * t_exit;
        // Reversed-Z: greater is nearer. Nearer than everything here, the
        // whole way across, means nothing in this cell can be hit.
        if (min(z_in, z_out) > nearest) {
            t = t_exit + nudge;
            level = min(level + 1, top);
            continue;
        }

        if (level > 0) {
            // Something in this cell may be in the way. Go to where the ray
            // reaches that depth and look closer.
            if (z_in > nearest) t += (nearest - z_in) / d.z;
            level -= 1;
            continue;
        }

        // Level 0: one pixel, one surface. A ray that crosses its depth here
        // has hit it. One that arrives already behind has either slipped
        // under the surface between two pixels -- a hit -- or is passing
        // behind an object the depth buffer cannot see around. Nothing tells
        // the two apart but distance, so trust falls off with it rather than
        // stopping at a line that neighbouring pixels land either side of.
        if (z_in > nearest) {
            hit_px = from.xy + d.xy * (t + (nearest - z_in) / d.z);
            return true;
        }
        //
        // Except a ray that runs nearly along a surface -- a mirror's view of
        // a wall at a grazing angle. The depth buffer holds one depth per
        // pixel, at its centre, so such a ray can stay in front of one
        // pixel's centre and arrive behind the next one's without a crossing
        // between them, and how far behind depends on where in the pixel it
        // fell. Grading trust by that turned the reflection into stripes, a
        // few pixels of full strength then a few of fading, column after
        // column. Behind by no more than the surface's own depth change over
        // a pixel is that slip, and a full hit.
        float slip = nearest - z_in;
        if (slip <= depth_step(int2(cell), nearest)) {
            hit_px = p;
            return true;
        }
        float surface_z = linear_depth(nearest);
        float behind = linear_depth(z_in) - surface_z;
        float allowed = thickness + surface_z * 0.01;
        if (behind <= allowed) {
            hit_px = p;
            confidence = 1.0 - smoothstep(0.25, 1.0, behind / allowed);
            return true;
        }
        t = t_exit + nudge;
    }
    return false;
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float3 scene = scene_color.SampleLevel(i.uv, 0.0).rgb;
    bool reflection_only = ssr.params.y > 0.5;
    float3 untouched = reflection_only ? float3(0.0) : scene;

    float depth = hiz.SampleLevel(i.uv, 0.0).r;
    if (depth <= 0.0) return float4(untouched, 1.0); // background

    float4 surface = surface_target.SampleLevel(i.uv, 0.0);
    float roughness = surface.r;
    float strength = 1.0 - smoothstep(ROUGHNESS_FULL, ROUGHNESS_NONE, roughness);
    if (strength <= 0.0) return float4(untouched, 1.0);

    float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    float4 world_h = mul(ssr.inv_view_proj, float4(ndc, depth, 1.0));
    float3 world_pos = world_h.xyz / world_h.w;

    float3 n = decode_normal(normal_target.SampleLevel(i.uv, 0.0).rg);
    float3 wo = normalize(ssr.eye.xyz - world_pos);
    if (dot(n, wo) < 0.0) n = -n; // the face the viewer sees, as lighting does
    float3 r = reflect(-wo, n);

    // A reflection aimed back at the viewer shows what is behind the camera
    // or the hidden side of what is in front of it. Neither is on screen.
    float along = dot(r, ssr.forward.xyz);
    strength *= smoothstep(-0.85, -0.5, along);
    if (strength <= 0.0) return float4(untouched, 1.0);

    // Start a couple of pixels off the surface, so the ray does not land on
    // the surface it is leaving.
    float view_z = dot(world_pos - ssr.eye.xyz, ssr.forward.xyz);
    float3 origin = world_pos + n * (view_z * ssr.forward.w * 2.0);
    float origin_z = dot(origin - ssr.eye.xyz, ssr.forward.xyz);

    // End the segment before it passes behind the near plane, where the
    // projection turns inside out.
    float reach = ssr.eye.w;
    float near = ssr.depth.x * 2.0;
    if (origin_z <= near) return float4(untouched, 1.0);
    if (along < 0.0) reach = min(reach, (origin_z - near) / -along);

    float2 hit_px;
    float confidence;
    if (!trace(to_screen(origin), to_screen(origin + r * reach), hit_px, confidence)) {
        return float4(untouched, 1.0);
    }
    strength *= confidence;

    // What was hit must face the ray. The depth buffer holds only the side of
    // a surface the camera sees, and a ray arriving from the other side would
    // be handed that side's colour.
    float2 hit_uv = hit_px * ssr.size.zw;
    float hit_depth = hiz.SampleLevel(hit_uv, 0.0).r;
    float2 hit_ndc = float2(hit_uv.x * 2.0 - 1.0, 1.0 - hit_uv.y * 2.0);
    float4 hit_h = mul(ssr.inv_view_proj, float4(hit_ndc, hit_depth, 1.0));
    float3 hit_pos = hit_h.xyz / hit_h.w;
    float3 hit_n = decode_normal(normal_target.SampleLevel(hit_uv, 0.0).rg);
    if (dot(hit_n, ssr.eye.xyz - hit_pos) < 0.0) hit_n = -hit_n;
    strength *= smoothstep(0.0, 0.1, -dot(hit_n, r));

    // Fade toward the frame's edge instead of cutting off where the screen
    // stops having an answer.
    float2 edge = min(hit_uv, 1.0 - hit_uv);
    strength *= smoothstep(0.0, 0.08, min(edge.x, edge.y));
    if (strength <= 0.0) return float4(untouched, 1.0);

    float3 reflected = scene_color.SampleLevel(hit_uv, 0.0).rgb;
    if (reflection_only) return float4(reflected * strength, 1.0);

    // The same weight `ibl` gives the prefiltered environment, so what is
    // taken out here is exactly what the lighting pass put in.
    float3 albedo = albedo_target.SampleLevel(i.uv, 0.0).rgb;
    float metallic = surface.g;
    float specular = surface.b;
    float n_dot_v = saturate(dot(n, wo));
    float3 f0 = lerp(float3(0.08 * specular), albedo, metallic);
    float2 ab = env_brdf_lut.SampleLevel(float2(n_dot_v, roughness), 0.0).rg;
    float3 weight = f0 * ab.x + ab.y;

    // What the lighting pass put in for the ambient reflection, recomputed the
    // same way so it comes out exactly: the probes' when they are in use and
    // see this point, else the environment's.
    float4 probe_spec = float4(0.0);
    if (ssr.probe_min.w > 0.5) {
        ProbeGrid grid;
        grid.lo = ssr.probe_min;
        grid.cell = ssr.probe_cell;
        grid.dims = ssr.probe_dims;
        float4 probe_diffuse;
        probe_lighting(world_pos, n, wo, roughness, grid,
            probe_irradiance_map, probe_depth, probe_radiance_map, probe_diffuse, probe_spec);
    }
    float3 from_env = float3(0.0);
    if (probe_spec.w > 0.5) {
        float visibility = ssr.params.x > 0.5 ? ao_target.SampleLevel(i.uv, 0.0).r : 1.0;
        from_env = probe_spec.rgb * weight * visibility;
    } else if (ssr.env.x > 0.5) {
        float2 spec_uv = direction_to_equirect(rotate_env(r, ssr.env.y));
        float mip = roughness * (ssr.env.w - 1.0);
        float visibility = ssr.params.x > 0.5 ? ao_target.SampleLevel(i.uv, 0.0).r : 1.0;
        from_env = env_specular.SampleLevel(spec_uv, mip).rgb * weight * ssr.env.z * visibility;
    }

    return float4(max(scene + strength * (reflected * weight - from_env), 0.0), 1.0);
}
