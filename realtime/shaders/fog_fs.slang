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
//     slice changes along it. They are unshadowed here as they are on surfaces.
//
// Scattering only: one bounce from light to eye. Light that scatters twice --
// the glow that fills real fog -- is not here, and fog casts no shadow on
// anything.
//
// The output is the light scattered toward the eye in rgb and the fraction of
// the scene behind that survives in alpha. fog_apply_fs blurs and composites
// it. Steps start at an offset from a 4x4 ordered pattern, which that blur
// averages away: few steps per pixel without the banding they would leave.

#include "shading.slang"

struct FogUniforms {
    float4 media;  // extinction at the floor, floor height, falloff height, anisotropy
    float4 colour; // rgb = scattering albedo, w = how far a ray is marched
};

[[vk::binding(0, 2)]] Sampler2D depth_target;
[[vk::binding(1, 2)]] Sampler2DArrayShadow shadow_map;
[[vk::binding(2, 2)]] Sampler2D env_irradiance;

// Uniforms before storage buffers; see lighting_fs for why.
[[vk::binding(0, 3)]] ConstantBuffer<LightingUniforms> frame;
[[vk::binding(1, 3)]] ConstantBuffer<FogUniforms> fog;

[[vk::binding(3, 2)]] Sampler2DArrayShadow spot_shadow_map;

[[vk::binding(4, 2)]] StructuredBuffer<Light> lights;
[[vk::binding(5, 2)]] StructuredBuffer<ClusterRange> cluster_ranges;
[[vk::binding(6, 2)]] StructuredBuffer<uint> cluster_indices;

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
    float3 ambient = float3(0.0);
    if (frame.env.x > 0.5) {
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
        float3 light = ambient;

        ClusterRange range = cluster_ranges[cluster_of(i.uv, view_z, frame.cluster_dims, frame.cluster_depth)];
        for (uint ci = 0; ci < range.count; ++ci) {
            int li = int(cluster_indices[range.offset + ci]);
            float3 wi;
            float3 radiance;
            if (!light_sample(lights[li], p, wi, radiance)) continue;
            if (li == sun && shadowed) {
                radiance *= sun_reaches(p, view_z);
            }
            int spot = spot_shadow_layer(lights[li]);
            if (spot >= 0) {
                radiance *= spot_reaches(p, spot);
            }
            // A step that lands on a light would divide by nothing.
            light += min(radiance, float3(1e3)) * phase(dot(wi, dir), g);
        }

        // The slab's own extinction, integrated across it exactly, so a thick
        // step neither gains nor loses energy to the step size.
        float through = exp(-thickness);
        scattered += transmittance * (1.0 - through) * light * fog.colour.rgb;
        transmittance *= through;
    }

    return float4(scattered, transmittance);
}
