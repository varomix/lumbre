// G-buffer vertex stage: mesh space in, clip space out.
//
// Geometry arrives in its mesh's own space, once per distinct mesh, and each
// instance supplies the world transform -- `scene_upload` groups every node
// that draws the same mesh into one instanced draw. One camera uniform, and the
// batching does the rest.

#include "common.slang"

[[vk::binding(0, 1)]] ConstantBuffer<CameraUniforms> camera;

struct VertexInput {
    float3 pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 tangent : TEXCOORD3;
    // Per instance.
    float4 world0 : TEXCOORD4;
    float4 world1 : TEXCOORD5;
    float4 world2 : TEXCOORD6;
    float4 world3 : TEXCOORD7;
    float3 normal0 : TEXCOORD8;
    float3 normal1 : TEXCOORD9;
    float3 normal2 : TEXCOORD10;
    uint instance : TEXCOORD11;
};

[shader("vertex")]
Varyings vertexMain(VertexInput v) {
    Varyings o;
    float3 world = instance_point(v.world0, v.world1, v.world2, v.world3, v.pos);
    o.clip_pos = mul(camera.view_proj, float4(world, 1.0));
    o.world_pos = world;
    o.normal = normalize(instance_vector(v.normal0, v.normal1, v.normal2, v.normal));
    // Flip V. Lumbre's image loader stores textures bottom-up, so an uploaded
    // texture is vertically mirrored against the GPU's row-0-is-top
    // convention. The path tracer compensates at every sample
    // (shaders/raytrace.metal, "flip V (loader stores bottom-up)"); doing it
    // once per vertex here is the same correction for less work, and keeps the
    // two modes sampling identical texels.
    o.uv = float2(v.uv.x, 1.0 - v.uv.y);
    // A tangent lies in the surface, so it follows the transform itself rather
    // than the normal matrix.
    o.tangent = float4(normalize(instance_vector(v.world0.xyz, v.world1.xyz, v.world2.xyz, v.tangent.xyz)), v.tangent.w);
    return o;
}
