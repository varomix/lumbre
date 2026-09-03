// G-buffer vertex stage: world space in, clip space out.
//
// Geometry arrives already in world space -- `scene_upload` bakes the scene
// graph's transforms into the vertices, exactly as the path tracer's flattening
// does -- so there is no model matrix here and no per-object uniform. One
// vertex buffer, one camera, and the draw batching does the rest.

#include "common.slang"

[[vk::binding(0, 1)]] ConstantBuffer<CameraUniforms> camera;

struct VertexInput {
    float3 pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 tangent : TEXCOORD3;
    float instance : TEXCOORD4;
};

[shader("vertex")]
Varyings vertexMain(VertexInput v) {
    Varyings o;
    o.clip_pos = mul(camera.view_proj, float4(v.pos, 1.0));
    o.world_pos = v.pos;
    o.normal = v.normal;
    // Flip V. Lumbre's image loader stores textures bottom-up, so an uploaded
    // texture is vertically mirrored against the GPU's row-0-is-top
    // convention. The path tracer compensates at every sample
    // (shaders/raytrace.metal, "flip V (loader stores bottom-up)"); doing it
    // once per vertex here is the same correction for less work, and keeps the
    // two modes sampling identical texels.
    o.uv = float2(v.uv.x, 1.0 - v.uv.y);
    o.tangent = v.tangent;
    return o;
}
