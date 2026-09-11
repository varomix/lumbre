// Label pass fragment stage: ground-truth channels, no shading.
//
// Nothing here is antialiased, jittered, filtered or tonemapped, and that is
// the point. A segmentation mask whose edges have been blended describes pixels
// that belong to neither object, and a depth that has been through a resolve is
// no longer a measurement. The beauty path may acquire TAA later; this pass
// must never share it.
//
// Semantic class comes from a per-instance buffer rather than a vertex
// attribute: it is a property of the object, and looking it up here keeps the
// vertex format from growing a second id.

struct LabelVaryings {
    float4 clip_pos : SV_Position;
    float3 normal : TEXCOORD0;
    nointerpolation uint instance : TEXCOORD1;
    float view_depth : TEXCOORD2;
};

[[vk::binding(0, 2)]] StructuredBuffer<uint> semantic_of_instance;

struct LabelTargets {
    // Instance id. Integer, not float: an id is not a quantity, and a UINT
    // target cannot be silently blended or rounded by anything downstream.
    uint instance : SV_Target0;
    // Semantic class id; 0 is the unlabelled class.
    uint semantic : SV_Target1;
    // Metric distance along the view axis, in scene units.
    float depth : SV_Target2;
    // World-space normal, raw in [-1, 1] rather than display-encoded to
    // [0, 1] the way the path tracer's normal AOV is. This one is data.
    float4 normal : SV_Target3;
};

[shader("fragment")]
LabelTargets fragmentMain(LabelVaryings i) {
    LabelTargets o;

    uint instance = i.instance;
    o.instance = instance;
    o.semantic = semantic_of_instance[instance];
    o.depth = i.view_depth;
    o.normal = float4(normalize(i.normal), 1.0);

    return o;
}
