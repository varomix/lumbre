// Motion pass vertex stage: where each surface point is now and was one frame
// earlier, both in clip space.
//
// Current placement comes from the instance stream, as in every other pass;
// the previous one from `prev_worlds`, indexed by the instance id, which only
// this pass reads. Both projections are unjittered: a velocity is a property
// of the scene and the cameras, not of TAA's sub-pixel offset, so like the
// label channels it is data and never sees the jitter.

#include "common.slang"

struct MotionUniforms {
    float4x4 view_proj;      // this frame's camera, unjittered
    float4x4 prev_view_proj; // the previous frame's camera, unjittered
    float4 params;           // x = 1 when prev_view_proj equals view_proj
};

// Uniforms are declared before the buffer so Slang gives them [[buffer(0)]]
// in MSL; SDL binds a stage's uniform buffers first and storage after.
[[vk::binding(0, 1)]] ConstantBuffer<MotionUniforms> frame;
// Four columns per instance id, so the layout cannot depend on a matrix
// packing rule: columns 4*id .. 4*id+3 of the previous world transform.
[[vk::binding(0, 0)]] StructuredBuffer<float4> prev_worlds;

struct VertexInput {
    float3 pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 tangent : TEXCOORD3;
    float4 world0 : TEXCOORD4;
    float4 world1 : TEXCOORD5;
    float4 world2 : TEXCOORD6;
    float4 world3 : TEXCOORD7;
    float3 normal0 : TEXCOORD8;
    float3 normal1 : TEXCOORD9;
    float3 normal2 : TEXCOORD10;
    uint instance : TEXCOORD11;
};

struct MotionVaryings {
    float4 clip_pos : SV_Position;
    // Interpolated in clip space and divided per pixel, which is the
    // perspective-correct position; dividing per vertex would not be.
    float4 now_clip : TEXCOORD0;
    float4 prev_clip : TEXCOORD1;
    // 1 for an unmoved instance under an unmoved camera, whose velocity is
    // exactly zero. Flat: it is a property of the instance.
    nointerpolation uint still : TEXCOORD2;
};

[shader("vertex")]
MotionVaryings vertexMain(VertexInput v) {
    MotionVaryings o;
    float3 world = instance_point(v.world0, v.world1, v.world2, v.world3, v.pos);
    uint base = v.instance * 4;
    float4 p0 = prev_worlds[base], p1 = prev_worlds[base + 1];
    float4 p2 = prev_worlds[base + 2], p3 = prev_worlds[base + 3];
    o.now_clip = mul(frame.view_proj, float4(world, 1.0));
    float3 prev_world = instance_point(p0, p1, p2, p3, v.pos);
    o.prev_clip = mul(frame.prev_view_proj, float4(prev_world, 1.0));
    // An unmoved instance under an unmoved camera is flagged rather than
    // left to the subtraction: the two positions round differently along
    // their separate paths, and a static surface must read exactly zero,
    // not a millionth of a pixel.
    bool still = frame.params.x > 0.5 &&
        all(p0 == v.world0) && all(p1 == v.world1) && all(p2 == v.world2) && all(p3 == v.world3);
    o.still = still ? 1u : 0u;
    o.clip_pos = o.now_clip;
    return o;
}
