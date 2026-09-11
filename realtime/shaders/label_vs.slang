// Label pass vertex stage.
//
// Shares the G-buffer's vertex buffer and layout, so a label frame draws the
// same geometry through the same batches — the instance id rides the vertex
// stream (see scene_upload.odin for why it cannot ride the batch).
//
// Depth is carried as a metric view-space distance rather than left to the
// depth buffer's nonlinear encoding: a label AOV is data, and a consumer
// wanting metres should not have to invert a projection to get them. The path
// tracer's own depth AOV divides distance by a hard-coded 20, which is fine as
// a debug picture and useless as a measurement.

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
    uint instance : TEXCOORD4;
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
    o.clip_pos = mul(frame.view_proj, float4(v.pos, 1.0));
    o.normal = v.normal;
    o.instance = v.instance;
    o.view_depth = dot(v.pos - frame.eye.xyz, frame.forward.xyz);
    return o;
}
