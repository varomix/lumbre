// Screen-space reflections, the trace: one glossy ray per pixel of a half-
// resolution target. ssr_resolve_fs then shares the rays among neighbours,
// traces mirrors' exact rays at full resolution, and composites.
//
// The lighting pass has already given every surface a specular reflection of
// the ENVIRONMENT -- the prefiltered map along the reflection vector. That is
// right for a surface that sees the sky and wrong for one that sees the room.
// This pass traces reflection rays through the depth buffer and, where they
// land on something on screen, the resolve swaps the environment's
// contribution for the lit colour found there. Where they land on nothing,
// the pixel keeps the environment: that is the whole fallback.
//
// What it cannot do, all inherent to reading the screen: reflect anything
// outside the frame, behind the camera, or hidden behind a nearer surface; or
// reflect glass, which is drawn after this pass. There is one bounce: a mirror
// seen in a mirror shows the environment.
//
// A glossy surface traces several rays, drawn from GGX's distribution of
// visible normals, so they spread as the lobe does -- stretched along the view
// at grazing angles, which is what streaks a reflection down a wet floor. Each
// reads what it hit from a blur pyramid of the lit image (its mips), at the
// level matching its own share of the lobe at that distance. The share of rays
// that hit sets how much of the environment's reflection is replaced, so a
// reflection's edge is as soft as the lobe makes it, and a miss leaves the
// environment exactly as the lighting pass put it.
//
// One ray per half-resolution pixel, not several per pixel: eight per pixel
// cost ten times the single mirror ray's 2.4 ms at 1080p, and full resolution
// alone cost 6 ms in a room of roughness-0.5 surfaces. Instead each ray
// samples a different part of the lobe -- interleaved gradient noise, which
// spreads well over a 3x3 neighbourhood, moved per TAA frame -- and the
// resolve averages each pixel's neighbours, weighted by how alike their
// surfaces are; a still shares them wider. A mirror needs every pixel, so the
// resolve traces its one exact ray itself.
//
// The march is hierarchical. hiz_fs builds a pyramid of nearest depths, and a
// ray that is nearer the eye than a cell's nearest surface crosses the whole
// cell in one step, climbing a level each time it does and dropping one when a
// cell might hold a hit. Depth is affine along a screen-space line, so the ray
// is a straight segment in (pixel x, pixel y, depth) and every crossing is
// solved exactly rather than stepped toward.

#include "ssr_common.slang"

[[vk::binding(0, 2)]] Sampler2D scene_color; // lit opaque image, trilinear over its blur mips
[[vk::binding(1, 2)]] Sampler2D normal_target;
[[vk::binding(2, 2)]] Sampler2D surface_target;
// Point-sampled at texel centres, one level at a time; see hiz_fs for why not
// Load.
[[vk::binding(3, 2)]] Sampler2D hiz;

#include "ssr_march.slang"

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    // rgb = colour found times trust, a = trust. Zero where nothing is traced.
    float depth = hiz.SampleLevel(i.uv, 0.0).r;
    if (depth <= 0.0) return float4(0.0); // background

    float roughness = surface_target.SampleLevel(i.uv, 0.0).r;
    // Mirrors are the resolve's to trace, at full resolution.
    if (roughness < ROUGHNESS_MIRROR || roughness >= ROUGHNESS_NONE) return float4(0.0);

    float3 world_pos = world_at(i.uv, depth);
    float3 n = decode_normal(normal_target.SampleLevel(i.uv, 0.0).rg);
    float3 wo = normalize(ssr.eye.xyz - world_pos);
    if (dot(n, wo) < 0.0) n = -n; // the face the viewer sees, as lighting does
    float n_dot_v = saturate(dot(n, wo));

    // Start a couple of pixels off the surface, so a ray does not land on the
    // surface it is leaving.
    float view_z = dot(world_pos - ssr.eye.xyz, ssr.forward.xyz);
    float3 origin = world_pos + n * (view_z * ssr.forward.w * 2.0);
    float origin_z = dot(origin - ssr.eye.xyz, ssr.forward.xyz);
    if (origin_z <= ssr.depth.x * 2.0) return float4(0.0);

    float3 colour;
    // The lobe's frame: n up, the view in the xz plane.
    float3 t1 = wo - n * dot(wo, n);
    t1 = dot(t1, t1) > 1e-10 ? normalize(t1)
       : normalize(cross(abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0), n));
    float3 t2 = cross(n, t1);
    float3 wo_local = float3(dot(wo, t1), 0.0, n_dot_v);
    // One ray with TAA, which averages the frames; a few for a still.
    int rays = max(int(ssr.rays.x), 1);
    // Each ray is shared among the resolve's taps, a span of 2 radius + 1.
    float span = 2.0 * ssr.rays.y + 1.0;
    float cone_tan = roughness * CONE_TAN_PER_ALPHA / (span * sqrt(float(rays)));
    float2 u0 = float2(
        pcss_spin(i.pos.xy, ssr.params.w),
        pcss_spin(i.pos.yx + float2(17.0, 31.0), ssr.params.w)) / 6.28318531;
    float4 sum = float4(0.0);
    [loop]
    for (int k = 0; k < rays; ++k) {
        // The pixel's offset, stepped along the plastic sequence (R2).
        float2 u = frac(u0 + float(k) * float2(0.7548776662, 0.5698402910));
        float3 h = sample_vndf(wo_local, roughness, u);
        float3 l_local = 2.0 * dot(wo_local, h) * h - wo_local;
        // Below the surface: the lobe's shadowed part, which reflects nothing,
        // as it adds nothing to the prefiltered environment either.
        if (l_local.z <= 0.0) continue;
        float3 dir = t1 * l_local.x + t2 * l_local.y + n * l_local.z;
        float trust = trace_ray(world_pos, origin, origin_z, dir, cone_tan, colour);
        sum += float4(colour * trust, trust);
    }
    return sum / float(rays);
}
