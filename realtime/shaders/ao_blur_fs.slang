// Depth-aware 4x4 box blur of the raw AO. The kernel matches the 4x4 rotation
// pattern ao_fs uses, so every rotation contributes once per output pixel and
// the pattern cancels. Taps across a depth discontinuity are rejected, or the
// darkening at an object's foot would bleed onto the background behind it.

struct BlurUniforms {
    float4 params; // 1/width, 1/height, near, far
};

[[vk::binding(0, 2)]] Sampler2D ao_raw;
[[vk::binding(1, 2)]] Sampler2D depth_target;
[[vk::binding(0, 3)]] ConstantBuffer<BlurUniforms> blur;

struct FullscreenOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

// Reversed-Z depth to distance along the view axis.
float linear_depth(float d) {
    float near = blur.params.z;
    float far = blur.params.w;
    return near * far / max(near + d * (far - near), 1e-9);
}

[shader("fragment")]
float4 fragmentMain(FullscreenOut i) : SV_Target {
    float2 texel = blur.params.xy;
    float d0 = depth_target.SampleLevel(i.uv, 0.0).r;
    if (d0 <= 0.0) return float4(1.0);
    float z0 = linear_depth(d0);

    float sum = 0.0;
    float weight = 0.0;
    for (int y = -2; y < 2; ++y) {
        for (int x = -2; x < 2; ++x) {
            float2 uv = i.uv + float2(x, y) * texel;
            float d = depth_target.SampleLevel(uv, 0.0).r;
            float z = linear_depth(d);
            float w = d > 0.0 && abs(z - z0) < z0 * 0.02 ? 1.0 : 0.0;
            sum += ao_raw.SampleLevel(uv, 0.0).r * w;
            weight += w;
        }
    }
    return float4(weight > 0.0 ? sum / weight : ao_raw.SampleLevel(i.uv, 0.0).r);
}
