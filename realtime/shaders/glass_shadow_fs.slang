// One glass surface in a glass shadow map. The target is cleared to (1, 1, 1,
// far); colour blends by multiplying, so rgb ends as the product of every glass
// surface's transmittance along the texel, and alpha by keeping the minimum,
// so it ends as the nearest glass's depth: the sun's normalized depth, or the
// distance from a spot or omni light. See glass_lookup in shading.slang.

struct GlassShadowUniforms {
    float4 light; // xyz = the light's position, w = 1 for a distance, 0 for depth
    float4 tint;  // rgb = what one surface lets through
};

[[vk::binding(0, 3)]] ConstantBuffer<GlassShadowUniforms> glass;

struct VertexOutput {
    float4 pos : SV_Position;
    float3 world : TEXCOORD0;
};

[shader("fragment")]
float4 fragmentMain(VertexOutput i) : SV_Target {
    float depth = glass.light.w > 0.5 ? length(i.world - glass.light.xyz) : i.pos.z;
    return float4(glass.tint.rgb, depth);
}
