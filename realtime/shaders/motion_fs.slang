// Motion pass fragment stage: screen-space velocity in pixels.
//
// Current position minus previous, x to the right and y down, the way image
// rows run: a pixel's content was at (pixel - velocity) one frame earlier.
// Exactly zero for anything that did not move under a camera that did not
// move. Not antialiased or jittered, like the label channels.

struct MotionUniforms {
    float4 size; // width, height, 0, 0
};

[[vk::binding(0, 3)]] ConstantBuffer<MotionUniforms> target;

struct MotionVaryings {
    float4 clip_pos : SV_Position;
    float4 now_clip : TEXCOORD0;
    float4 prev_clip : TEXCOORD1;
    nointerpolation uint still : TEXCOORD2;
};

[shader("fragment")]
float2 fragmentMain(MotionVaryings i) : SV_Target0 {
    if (i.still != 0u) {
        return float2(0.0, 0.0);
    }
    float2 now_ndc = i.now_clip.xy / i.now_clip.w;
    float2 prev_ndc = i.prev_clip.xy / i.prev_clip.w;
    // NDC spans 2 across the target, with y up; pixels run y down.
    float2 delta = (now_ndc - prev_ndc) * 0.5 * target.size.xy;
    return float2(delta.x, -delta.y);
}
