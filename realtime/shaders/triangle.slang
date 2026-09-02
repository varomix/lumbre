// Proves the whole shader path: Slang source -> slangc -> MSL and SPIR-V ->
// #load'ed blob -> SDL_GPU shader -> graphics pipeline -> offscreen target ->
// ImGui. No vertex buffer, no uniforms, no bindings, so a failure here is a
// failure of the toolchain and nothing else.
//
// Replaced by the G-buffer pass once that lands; kept until then as the
// smallest thing that can go wrong.

struct VSOut {
    float4 pos   : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VSOut vertexMain(uint vid : SV_VertexID) {
    float2 p[3] = { float2(0.0, 0.6), float2(-0.6, -0.6), float2(0.6, -0.6) };
    float3 c[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };

    VSOut o;
    o.pos = float4(p[vid], 0.0, 1.0);
    o.color = c[vid];
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOut i) : SV_Target {
    return float4(i.color, 1.0);
}
