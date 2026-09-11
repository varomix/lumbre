// Label pass vertex stage.
//
// Shares the G-buffer's vertex buffer, instance buffer and layout, so a label
// frame draws the same geometry through the same batches. The instance id
// rides the instance buffer: it is a property of the placement, not of the
// mesh's vertices, which every copy of the mesh shares.
//
// Depth is carried as a metric view-space distance rather than left to the
// depth buffer's nonlinear encoding: a label AOV is data, and a consumer
// wanting metres should not have to invert a projection to get them. The path
// tracer's own depth AOV divides distance by a hard-coded 20, which is fine as
// a debug picture and useless as a measurement.

#include "common.slang"

struct LabelUniforms {
    float4x4 view_proj;
    float4 eye;     // xyz
    float4 forward; // camera view axis
};

[[vk::binding(0, 1)]] ConstantBuffer<LabelUniforms> frame;

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

struct LabelVaryings {
    float4 clip_pos : SV_Position;
    float3 normal : TEXCOORD0;
    // Flat, as every integer varying must be: an id is not interpolated
    // across a triangle, and an integer carries it exactly at any size.
    nointerpolation uint instance : TEXCOORD1;
    float view_depth : TEXCOORD2;
};

[shader("vertex")]
LabelVaryings vertexMain(VertexInput v) {
    LabelVaryings o;
    float3 world = instance_point(v.world0, v.world1, v.world2, v.world3, v.pos);
    o.clip_pos = mul(frame.view_proj, float4(world, 1.0));
    o.normal = normalize(instance_vector(v.normal0, v.normal1, v.normal2, v.normal));
    o.instance = v.instance;
    o.view_depth = dot(world - frame.eye.xyz, frame.forward.xyz);
    return o;
}
