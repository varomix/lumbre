// Vertex stage for the glass shadow maps: shadow_vs, also handing the
// fragment its world position for the distance it records.

#include "common.slang"

struct ShadowUniforms {
    float4x4 light_view_proj;
};

[[vk::binding(0, 1)]] ConstantBuffer<ShadowUniforms> shadow;

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

struct VertexOutput {
    float4 pos : SV_Position;
    float3 world : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput v) {
    VertexOutput o;
    o.world = instance_point(v.world0, v.world1, v.world2, v.world3, v.pos);
    o.pos = mul(shadow.light_view_proj, float4(o.world, 1.0));
    return o;
}
