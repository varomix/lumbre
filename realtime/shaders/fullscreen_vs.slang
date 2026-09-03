// One oversized triangle covering the screen, generated from the vertex id.
//
// A triangle rather than two triangles for a quad: no diagonal seam, one fewer
// primitive, and no vertex buffer to bind. Shared by every fullscreen pass.

struct FullscreenOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
FullscreenOut vertexMain(uint vid : SV_VertexID) {
    FullscreenOut o;
    // (0,0) (2,0) (0,2) in UV -> clip coordinates covering [-1,1] twice over.
    o.uv = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}
