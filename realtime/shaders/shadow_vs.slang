// Depth-only vertex stage for the shadow cascades.
//
// Uses the same vertex layout as the G-buffer pass so the two share one vertex
// buffer and one batch list; everything but position is ignored, and the
// hardware discards the unused attributes.

struct ShadowUniforms {
    float4x4 light_view_proj;
};

[[vk::binding(0, 1)]] ConstantBuffer<ShadowUniforms> shadow;

struct VertexInput {
    float3 pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 tangent : TEXCOORD3;
    uint instance : TEXCOORD4;
};

[shader("vertex")]
float4 vertexMain(VertexInput v) : SV_Position {
    return mul(shadow.light_view_proj, float4(v.pos, 1.0));
}
