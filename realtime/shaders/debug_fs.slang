// Renders one G-buffer channel to the screen, unlit.
//
// This is the pass that makes the G-buffer inspectable rather than a black box:
// if the normals are wrong, or a material batch picked up the wrong texture,
// this is where it is visible. It stays in the build after deferred lighting
// lands, because "the lighting looks wrong" is nearly always a question about
// what is in these targets.

#include "common.slang"

[[vk::binding(0, 2)]] Sampler2D albedo_target;
[[vk::binding(1, 2)]] Sampler2D normal_target;
[[vk::binding(2, 2)]] Sampler2D surface_target;
[[vk::binding(3, 2)]] Sampler2D emission_target;
[[vk::binding(4, 2)]] Sampler2D depth_target;
// Integer targets, read with Load at integer coordinates: sampling would
// filter ids into values no object has.
[[vk::binding(5, 2)]] Texture2D<uint> instance_target;
[[vk::binding(6, 2)]] Texture2D<uint> semantic_target;

struct DebugUniforms {
    // x = channel (see Debug_View in renderer.odin), y = near, z = far,
    // w = the scene's far extent, used as the display range for depth.
    float4 params;
};

[[vk::binding(0, 3)]] ConstantBuffer<DebugUniforms> debug;

struct FullscreenOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

// A stable, well-spread colour per id, so neighbouring ids do not come out as
// neighbouring shades. Display only — the buffer holds the raw id.
float3 id_colour(uint id) {
    if (id == 0u) return float3(0.0); // unlabelled reads as black
    uint h = id * 2654435761u;
    return float3(
        float((h >> 16) & 0xFFu),
        float((h >> 8) & 0xFFu),
        float(h & 0xFFu)) / 255.0;
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    int channel = int(debug.params.x);
    float3 c = float3(0.0);

    switch (channel) {
    case 1: // albedo
        c = albedo_target.Sample(i.uv).rgb;
        break;
    case 2: // world normal, remapped to [0,1] for display
        c = decode_normal(normal_target.Sample(i.uv).rg) * 0.5 + 0.5;
        break;
    case 3: // roughness
        c = float3(surface_target.Sample(i.uv).r);
        break;
    case 4: // metallic
        c = float3(surface_target.Sample(i.uv).g);
        break;
    case 5: // emission, Reinhard-compressed since it is unbounded
        float3 e = emission_target.Sample(i.uv).rgb;
        c = e / (1.0 + e);
        break;
    case 6: {
        // Linearized depth. The raw buffer is almost entirely near-plane
        // precision, so displaying it directly is a white rectangle.
        float d = depth_target.Sample(i.uv).r;
        float near = debug.params.y;
        float far = debug.params.z;
        float linear_depth = near * far / max(far - d * (far - near), 1e-9);
        // Scaled against how far the scene reaches. The camera's far plane is
        // a thousand focus distances away and crushes everything to black;
        // the focus distance alone saturates any scene deeper than twice it.
        float display_far = max(debug.params.w, 1e-6);
        c = float3(saturate(linear_depth / display_far));
        break;
    }
    case 7: { // instance id
        uint2 px = uint2(i.pos.xy);
        c = id_colour(instance_target.Load(int3(int2(px), 0)));
        break;
    }
    case 8: { // semantic class id
        uint2 px = uint2(i.pos.xy);
        c = id_colour(semantic_target.Load(int3(int2(px), 0)));
        break;
    }
    default:
        c = albedo_target.Sample(i.uv).rgb;
        break;
    }

    // The target is UNORM, not sRGB: encode here so the debug views are shown
    // in the same space as everything else on screen.
    return float4(linear_to_srgb(c), 1.0);
}
