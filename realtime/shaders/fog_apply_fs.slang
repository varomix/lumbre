// Composites the fog over the lit image: the scene dimmed by what the fog let
// through, plus what the fog scattered toward the eye. The blend state does the
// arithmetic -- this pass writes (scattered, transmittance) and the target
// becomes scattered + target * transmittance -- so the lit image is never
// copied to be read.
//
// Depth-aware 4x4 box blur first, matching the 4x4 pattern fog_fs offsets its
// steps by, so every offset contributes once per pixel and the pattern cancels.
// Taps across a depth discontinuity are rejected: fog in front of a near
// object is thinner than in front of the wall behind it, and averaging the two
// draws a halo around the object.

struct ApplyUniforms {
    float4 params; // 1/width, 1/height, near, far
};

[[vk::binding(0, 2)]] Sampler2D fog_raw;
[[vk::binding(1, 2)]] Sampler2D depth_target;
[[vk::binding(0, 3)]] ConstantBuffer<ApplyUniforms> apply;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

// Reversed-Z depth to distance along the view axis. The far plane, where
// nothing was drawn, comes out at `far`.
float linear_depth(float d) {
    float near = apply.params.z;
    float far = apply.params.w;
    return near * far / max(near + d * (far - near), 1e-9);
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float2 texel = apply.params.xy;
    float z0 = linear_depth(depth_target.SampleLevel(i.uv, 0.0).r);

    float4 sum = float4(0.0);
    float weight = 0.0;
    for (int y = -2; y < 2; ++y) {
        for (int x = -2; x < 2; ++x) {
            float2 uv = i.uv + float2(x, y) * texel;
            float z = linear_depth(depth_target.SampleLevel(uv, 0.0).r);
            float w = abs(z - z0) < z0 * 0.05 ? 1.0 : 0.0;
            sum += fog_raw.SampleLevel(uv, 0.0) * w;
            weight += w;
        }
    }
    return weight > 0.0 ? sum / weight : fog_raw.SampleLevel(i.uv, 0.0);
}
