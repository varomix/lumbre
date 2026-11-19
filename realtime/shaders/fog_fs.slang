// Height fog with single scattering: one march per pixel from the eye to the
// surface it sees, or out to a far limit where it sees none.
//
// Density is exponential in height: full at the scene's floor and below it,
// falling by e for every `falloff` above. At each step the fog scatters three
// things toward the eye:
//
//   - the sky, as the environment's average radiance -- the same in every
//     direction, which is all a single lookup can say;
//   - the sun, through the cascaded shadow map. A step in shadow gets none,
//     and that difference from step to step is what a light shaft is;
//   - the local lights reaching this froxel, from the clustered grid the
//     lighting pass uses. A ray stays in one screen tile, so only the depth
//     slice changes along it. Shadowed ones are shadowed here too, hard. An
//     area light gives its radiance times the solid angle it fills from the
//     step (area_solid_angle): a point carrying its area would blow up beside
//     a large panel.
//
// Light is dimmed by the fog on its way in, and what the fog takes out it
// partly gives back by scattering more than once -- the glow that fills real
// fog. Both are local estimates at each step, with H the distance light goes
// to leave the fog (the layer's height, capped at the scene's size) and
// tau = sigma H:
//
//   - the sky and the sun arrive dimmed by exp(-tau), a local light by
//     exp(-sigma d) for its distance d, up to H;
//   - every order after the first is taken as isotropic, each keeping the
//     albedo a times the chance f = 1 - exp(-tau) of scattering again, which
//     sum as a geometric series: a f / (1 - a f) of the first order's light.
//
// Under a uniform sky with a = 1 that comes out exactly at the sky's radiance,
// as it must: (1 - f) times 1 / (1 - f). With multiple scattering off, the
// dimming stays: what is left is single scattering, correctly darker. Fog
// casts no shadow on surfaces.
//
// The output is the light scattered toward the eye in rgb and the fraction of
// the scene behind that survives in alpha. fog_apply_fs blurs and composites
// it. Steps start at an offset from a 4x4 ordered pattern, which that blur
// averages away: few steps per pixel without the banding they would leave.

#include "shading.slang"

struct FogUniforms {
    float4 media;  // extinction at the floor, floor height, falloff height, anisotropy
    float4 colour; // rgb = scattering albedo, w = how far a ray is marched
    float4 multi;  // x = the distance light travels to leave the fog (H), y = multiple scattering on
};

[[vk::binding(0, 2)]] Sampler2D depth_target;
[[vk::binding(1, 2)]] Sampler2DArrayShadow shadow_map;
[[vk::binding(2, 2)]] Sampler2D env_irradiance;

// Uniforms before storage buffers; see lighting_fs for why.
[[vk::binding(0, 3)]] ConstantBuffer<LightingUniforms> frame;
[[vk::binding(1, 3)]] ConstantBuffer<FogUniforms> fog;

[[vk::binding(3, 2)]] Sampler2DArrayShadow spot_shadow_map;
[[vk::binding(4, 2)]] Sampler2DArrayShadow omni_shadow_map;

[[vk::binding(5, 2)]] StructuredBuffer<Light> lights;
[[vk::binding(6, 2)]] StructuredBuffer<ClusterRange> cluster_ranges;
[[vk::binding(7, 2)]] StructuredBuffer<uint> cluster_indices;
// Each baked probe's mean radiance; read only when frame.probe_min.w is set.
[[vk::binding(8, 2)]] StructuredBuffer<float4> probe_mean;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

static const int STEPS = 32;

// Henyey-Greenstein: how much light turning through `cos_theta` keeps going.
// Integrates to one over the sphere; g = 0 is isotropic, toward 1 is a haze
// that glows around the light.
float phase(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * LUMBRE_PI * denom * sqrt(max(denom, 1e-6)));
}

// Whether the sun reaches a point in the air. One tap and a constant bias:
// there is no surface here to offset along or to speckle.
float sun_reaches(float3 p, float view_z) {
    int cascade = pick_cascade(view_z, frame.cascade_splits);
    float4 clip = mul(frame.cascade_view_proj[cascade], float4(p, 1.0));
    float3 proj = clip.xyz / clip.w;
    float2 uv = float2(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || proj.z > 1.0) return 1.0;
    return shadow_map.SampleCmpLevelZero(float3(uv, float(cascade)), proj.z - frame.params.z);
}

// Whether a shadowed spot light reaches a point in the air: one tap, and no
// bias, for the same reason as sun_reaches.
float spot_reaches(float3 p, int layer) {
    float4 clip = mul(frame.spot_view_proj[layer], float4(p, 1.0));
    if (clip.w <= 0.0) return 1.0;
    float3 proj = clip.xyz / clip.w;
    float2 uv = float2(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || proj.z > 1.0) return 1.0;
    return spot_shadow_map.SampleCmpLevelZero(float3(uv, float(layer)), proj.z);
}

// The same for a shadowed point or sphere light, on the face that sees `p`.
float omni_reaches(float3 p, float3 light_pos, int slot) {
    int layer = slot * 6 + omni_face(p - light_pos);
    float4 clip = mul(frame.omni_view_proj[layer], float4(p, 1.0));
    if (clip.w <= 0.0) return 1.0;
    float3 proj = clip.xyz / clip.w;
    float2 uv = float2(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (proj.z > 1.0) return 1.0;
    return omni_shadow_map.SampleCmpLevelZero(float3(uv, float(layer)), proj.z);
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float depth = depth_target.SampleLevel(i.uv, 0.0).r;
    float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    // Depth 0 is the far plane: background, marched out to the limit.
    float4 end_h = mul(frame.inv_view_proj, float4(ndc, max(depth, 0.0), 1.0));
    float3 to_end = end_h.xyz / end_h.w - frame.eye.xyz;
    float reach = length(to_end);
    float3 dir = to_end / max(reach, 1e-9);
    reach = depth > 0.0 ? min(reach, fog.colour.w) : fog.colour.w;

    float sigma = fog.media.x;
    float floor_y = fog.media.y;
    float falloff = max(fog.media.z, 1e-6);
    float g = fog.media.w;

    // The sky as fog sees it: the mean of what arrives from above and below.
    // With baked probes, what arrives at each step instead, sky and bounce
    // alike, occluded where the scene occludes it: fog under a roof is no
    // longer lit by the whole sky.
    bool probes = frame.probe_min.w > 0.5;
    float3 ambient = float3(0.0);
    if (!probes && frame.env.x > 0.5) {
        float3 up = env_irradiance.SampleLevel(direction_to_equirect(float3(0, 1, 0)), 0.0).rgb;
        float3 down = env_irradiance.SampleLevel(direction_to_equirect(float3(0, -1, 0)), 0.0).rgb;
        ambient = 0.5 * (up + down) * frame.env.z;
    }

    int sun = int(frame.eye.w);
    bool shadowed = frame.params.y > 0.5;
    float along_view = dot(dir, frame.forward.xyz);

    uint2 px = uint2(i.pos.xy) & 3u;
    const float bayer[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float jitter = (bayer[px.y * 4 + px.x] + 0.5) / 16.0;

    float dt = reach / float(STEPS);
    float3 scattered = float3(0.0);
    float transmittance = 1.0;
    for (int s = 0; s < STEPS; ++s) {
        float t = (float(s) + jitter) * dt;
        float3 p = frame.eye.xyz + dir * t;
        float density = sigma * exp(-max(p.y - floor_y, 0.0) / falloff);
        float thickness = density * dt;
        if (thickness < 1e-6) continue;

        float view_z = t * along_view;
        // Phase-weighted for the first scattering; `isotropic` is the same
        // light scattered evenly, which the later orders start from. Both
        // dimmed by the fog on the way in.
        float escape = fog.multi.x;
        float through_out = exp(-density * escape);
        float3 light = (probes ? probe_mean_radiance(p, frame, probe_mean) : ambient) * through_out;
        float3 isotropic = light;

        ClusterRange range = cluster_ranges[cluster_of(i.uv, view_z, frame.cluster_dims, frame.cluster_depth)];
        for (uint ci = 0; ci < range.count; ++ci) {
            int li = int(cluster_indices[range.offset + ci]);
            float3 wi;
            float3 radiance;
            if (!light_sample(lights[li], p, wi, radiance)) continue;
            if (int(lights[li].params.x) >= LIGHT_QUAD) {
                radiance = lights[li].emission.rgb * area_solid_angle(lights[li], p);
            }
            // Through the fog between: as far as the light for a local one,
            // out of the layer for the sun.
            float path = li == sun ? escape : min(length(lights[li].position.xyz - p), escape);
            radiance *= exp(-density * path);
            if (li == sun && shadowed) {
                radiance *= sun_reaches(p, view_z);
            }
            int spot = spot_shadow_layer(lights[li]);
            if (spot >= 0) {
                radiance *= spot_reaches(p, spot);
            }
            int omni = omni_shadow_slot(lights[li]);
            if (omni >= 0) {
                radiance *= omni_reaches(p, lights[li].position.xyz, omni);
            }
            // A step that lands on a light would divide by nothing.
            radiance = min(radiance, float3(1e3));
            light += radiance * phase(dot(wi, dir), g);
            isotropic += radiance * (1.0 / (4.0 * LUMBRE_PI));
        }

        // Orders two and up, as above: order n carries albedo^n and n - 1
        // chances to scatter again. The first albedo is applied below.
        if (fog.multi.y > 0.5) {
            float3 af = min(fog.colour.rgb * (1.0 - through_out), float3(0.95));
            light += isotropic * af / (1.0 - af);
        }

        // The slab's own extinction, integrated across it exactly, so a thick
        // step neither gains nor loses energy to the step size.
        float through = exp(-thickness);
        scattered += transmittance * (1.0 - through) * light * fog.colour.rgb;
        transmittance *= through;
    }

    return float4(scattered, transmittance);
}
