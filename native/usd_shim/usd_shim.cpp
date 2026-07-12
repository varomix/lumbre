#include "usd_shim.h"

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layerUtils.h>
#include <pxr/usd/sdf/valueTypeName.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/cylinder_1.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/cylinderLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/shapingAPI.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <new>

PXR_NAMESPACE_USING_DIRECTIVE

// ---------------------------------------------------------------------------
// Opaque handle storage. UsdShimStage owns the flattened UsdStageRefPtr and a
// pool of UsdPrim wrappers (UsdShimPrim) so handles handed to Odin stay valid
// for the stage's lifetime without per-call allocation churn.
// ---------------------------------------------------------------------------

struct UsdShimPrim {
    UsdPrim prim;
    std::string scratch_a; // backing storage for prim_type_name/prim_name
    std::string scratch_b;
};

struct UsdShimStage {
    UsdStageRefPtr stage;
    std::vector<UsdShimPrim*> prim_pool;
    std::string resolve_scratch;

    UsdShimPrim* wrap(const UsdPrim& p) {
        UsdShimPrim* h = new UsdShimPrim();
        h->prim = p;
        prim_pool.push_back(h);
        return h;
    }

    ~UsdShimStage() {
        for (auto* h : prim_pool) delete h;
    }
};

static void write_err(char* err_buf, int err_buf_len, const char* msg) {
    if (!err_buf || err_buf_len <= 0) return;
    std::snprintf(err_buf, static_cast<size_t>(err_buf_len), "%s", msg);
}

// Forward-declared; defined near the bottom alongside the registry it manages.
static void usd_shim_register_owner(UsdStage* raw_stage, UsdShimStage* owner);
static void usd_shim_unregister_owner(UsdStage* raw_stage);

extern "C" UsdShimStageHandle usd_shim_open_flattened(const char* path, char* err_buf, int err_buf_len) {
    if (!path) {
        write_err(err_buf, err_buf_len, "usd_shim_open_flattened: null path");
        return nullptr;
    }
    try {
        UsdStageRefPtr opened = UsdStage::Open(path);
        if (!opened) {
            write_err(err_buf, err_buf_len, "UsdStage::Open failed (bad path or unreadable file)");
            return nullptr;
        }
        UsdStageRefPtr flattened = UsdStage::Open(opened->Flatten());
        if (!flattened) {
            write_err(err_buf, err_buf_len, "UsdStage::Flatten produced no stage");
            return nullptr;
        }
        UsdShimStage* h = new UsdShimStage();
        h->stage = flattened;
        usd_shim_register_owner(flattened.operator->(), h);
        return h;
    } catch (const std::exception& e) {
        write_err(err_buf, err_buf_len, e.what());
        return nullptr;
    } catch (...) {
        write_err(err_buf, err_buf_len, "unknown exception in usd_shim_open_flattened");
        return nullptr;
    }
}

extern "C" void usd_shim_close(UsdShimStageHandle stage) {
    if (!stage) return;
    try {
        if (stage->stage) usd_shim_unregister_owner(stage->stage.operator->());
        delete stage;
    } catch (...) {
        // never let a dtor throw across the boundary
    }
}

extern "C" int usd_shim_get_stage_info(UsdShimStageHandle stage, UsdShimStageInfo* out) {
    if (!out) return 0;
    out->up_axis = 0;              // Y
    out->meters_per_unit = 1.0;
    if (!stage || !stage->stage) return 0;
    try {
        TfToken up = UsdGeomGetStageUpAxis(stage->stage);
        out->up_axis = (up == UsdGeomTokens->z) ? 1 : 0;
        out->meters_per_unit = UsdGeomGetStageMetersPerUnit(stage->stage);
        return 1;
    } catch (...) {
        return 0;
    }
}

extern "C" UsdShimPrimHandle usd_shim_get_pseudo_root(UsdShimStageHandle stage) {
    if (!stage) return nullptr;
    try {
        return stage->wrap(stage->stage->GetPseudoRoot());
    } catch (...) {
        return nullptr;
    }
}

static UsdShimStage* usd_shim_find_owner(const UsdStageWeakPtr& stage);

extern "C" int usd_shim_get_children(UsdShimPrimHandle prim, UsdShimPrimHandle* out, int max) {
    if (!prim || !prim->prim.GetStage()) return 0;
    try {
        // UsdPrim has no back-pointer to our UsdShimStage wrapper, so
        // children are looked up via a small static registry keyed by the
        // stage's raw pointer, populated on open/close (see bottom of file).
        UsdShimStage* owner = usd_shim_find_owner(prim->prim.GetStage());
        if (!owner) return 0;

        int count = 0;
        for (const UsdPrim& child : prim->prim.GetChildren()) {
            if (count < max) {
                out[count] = owner->wrap(child);
            }
            count++;
        }
        return count;
    } catch (...) {
        return 0;
    }
}

extern "C" const char* usd_shim_prim_type_name(UsdShimPrimHandle prim) {
    if (!prim) return "";
    try {
        prim->scratch_a = prim->prim.GetTypeName().GetString();
        return prim->scratch_a.c_str();
    } catch (...) {
        return "";
    }
}

extern "C" const char* usd_shim_prim_name(UsdShimPrimHandle prim) {
    if (!prim) return "";
    try {
        prim->scratch_b = prim->prim.GetName().GetString();
        return prim->scratch_b.c_str();
    } catch (...) {
        return "";
    }
}

extern "C" int usd_shim_get_local_transform(UsdShimPrimHandle prim, double out_mat4x4[16]) {
    for (int i = 0; i < 16; i++) out_mat4x4[i] = (i % 5 == 0) ? 1.0 : 0.0; // identity
    if (!prim) return 0;
    try {
        UsdGeomXformable xformable(prim->prim);
        if (!xformable) return 0;
        GfMatrix4d mat(1.0);
        bool reset_stack = false;
        if (!xformable.GetLocalTransformation(&mat, &reset_stack)) return 0;
        const double* data = mat.GetArray(); // row-major 4x4
        for (int i = 0; i < 16; i++) out_mat4x4[i] = data[i];
        return 1;
    } catch (...) {
        return 0;
    }
}

extern "C" int usd_shim_get_mesh_data(UsdShimPrimHandle prim, UsdShimMeshData* out) {
    if (!prim || !out) return 0;
    std::memset(out, 0, sizeof(UsdShimMeshData));
    try {
        UsdGeomMesh mesh(prim->prim);
        if (!mesh) return 0;

        VtArray<GfVec3f> points;
        if (!mesh.GetPointsAttr().Get(&points) || points.empty()) return 0;

        VtArray<int> face_vertex_indices;
        VtArray<int> face_vertex_counts;
        mesh.GetFaceVertexIndicesAttr().Get(&face_vertex_indices);
        mesh.GetFaceVertexCountsAttr().Get(&face_vertex_counts);
        if (face_vertex_indices.empty() || face_vertex_counts.empty()) return 0;

        out->points = static_cast<float*>(std::malloc(sizeof(float) * 3 * points.size()));
        out->point_count = static_cast<int>(points.size());
        for (size_t i = 0; i < points.size(); i++) {
            out->points[i * 3 + 0] = points[i][0];
            out->points[i * 3 + 1] = points[i][1];
            out->points[i * 3 + 2] = points[i][2];
        }

        out->face_vertex_indices = static_cast<int*>(std::malloc(sizeof(int) * face_vertex_indices.size()));
        out->index_count = static_cast<int>(face_vertex_indices.size());
        std::memcpy(out->face_vertex_indices, face_vertex_indices.cdata(), sizeof(int) * face_vertex_indices.size());

        out->face_vertex_counts = static_cast<int*>(std::malloc(sizeof(int) * face_vertex_counts.size()));
        out->face_count = static_cast<int>(face_vertex_counts.size());
        std::memcpy(out->face_vertex_counts, face_vertex_counts.cdata(), sizeof(int) * face_vertex_counts.size());

        // Normals
        VtArray<GfVec3f> normals;
        if (mesh.GetNormalsAttr().Get(&normals) && !normals.empty()) {
            TfToken interp = mesh.GetNormalsInterpolation();
            out->normal_count = static_cast<int>(normals.size());
            out->normals = static_cast<float*>(std::malloc(sizeof(float) * 3 * normals.size()));
            for (size_t i = 0; i < normals.size(); i++) {
                out->normals[i * 3 + 0] = normals[i][0];
                out->normals[i * 3 + 1] = normals[i][1];
                out->normals[i * 3 + 2] = normals[i][2];
            }
            if (interp == UsdGeomTokens->faceVarying) out->normal_interp = USD_SHIM_INTERP_FACE_VARYING;
            else if (interp == UsdGeomTokens->uniform) out->normal_interp = USD_SHIM_INTERP_UNIFORM;
            else out->normal_interp = USD_SHIM_INTERP_VERTEX;
        }

        // UVs. There is no mandated primvar name: "st" is the convention,
        // but exporters emit "uv", "st0", "UVMap" and others. Rather than
        // guess, take the first texcoord-typed primvar, preferring an
        // exact "st" when several exist.
        UsdGeomPrimvarsAPI primvars_api(prim->prim);
        UsdGeomPrimvar uv_primvar;
        for (const UsdGeomPrimvar& pv : primvars_api.GetPrimvarsWithValues()) {
            const SdfValueTypeName type = pv.GetTypeName();
            if (type != SdfValueTypeNames->TexCoord2fArray &&
                type != SdfValueTypeNames->Float2Array) {
                continue;
            }
            if (pv.GetPrimvarName() == "st") {
                uv_primvar = pv;
                break;
            }
            if (!uv_primvar) uv_primvar = pv;
        }

        if (uv_primvar) {
            VtArray<GfVec2f> uvs;
            // ComputeFlattened resolves indexed primvars into one value per
            // element, so the corner/vertex indexing below is uniform
            // regardless of how the primvar was authored.
            if (uv_primvar.ComputeFlattened(&uvs) && !uvs.empty()) {
                TfToken interp = uv_primvar.GetInterpolation();
                out->uv_count = static_cast<int>(uvs.size());
                out->uvs = static_cast<float*>(std::malloc(sizeof(float) * 2 * uvs.size()));
                for (size_t i = 0; i < uvs.size(); i++) {
                    out->uvs[i * 2 + 0] = uvs[i][0];
                    out->uvs[i * 2 + 1] = uvs[i][1];
                }
                if (interp == UsdGeomTokens->faceVarying) out->uv_interp = USD_SHIM_INTERP_FACE_VARYING;
                else if (interp == UsdGeomTokens->uniform) out->uv_interp = USD_SHIM_INTERP_UNIFORM;
                else out->uv_interp = USD_SHIM_INTERP_VERTEX;
            }
        }

        return 1;
    } catch (const std::exception&) {
        usd_shim_free_mesh_data(out);
        std::memset(out, 0, sizeof(UsdShimMeshData));
        return 0;
    } catch (...) {
        usd_shim_free_mesh_data(out);
        std::memset(out, 0, sizeof(UsdShimMeshData));
        return 0;
    }
}

extern "C" void usd_shim_free_mesh_data(UsdShimMeshData* data) {
    if (!data) return;
    std::free(data->points);
    std::free(data->face_vertex_indices);
    std::free(data->face_vertex_counts);
    std::free(data->normals);
    std::free(data->uvs);
    std::memset(data, 0, sizeof(UsdShimMeshData));
}

// ---------------------------------------------------------------------------
// Parametric gprims. These prims store no points -- a Sphere is a `radius`
// and nothing else -- so there is nothing to marshal but the parameters. The
// caller turns them into triangles.
// ---------------------------------------------------------------------------

namespace {

// UsdGeomCylinder/Cone/Capsule all default their spine to Z.
int read_axis(const UsdAttribute& axis_attr) {
    TfToken axis;
    if (axis_attr && axis_attr.Get(&axis)) {
        if (axis == UsdGeomTokens->x) return USD_SHIM_AXIS_X;
        if (axis == UsdGeomTokens->y) return USD_SHIM_AXIS_Y;
    }
    return USD_SHIM_AXIS_Z;
}

} // namespace

extern "C" int usd_shim_get_gprim_data(UsdShimPrimHandle prim, UsdShimGprimData* out) {
    if (!prim || !out) return 0;
    std::memset(out, 0, sizeof(UsdShimGprimData));
    out->axis = USD_SHIM_AXIS_Z;
    try {
        // Every Get() below falls back to the schema's registered default when
        // the attribute is unauthored, so an empty `def Sphere "s" {}` still
        // comes back as the unit sphere USD says it is.
        if (UsdGeomCube cube{prim->prim}) {
            out->type = USD_SHIM_GPRIM_CUBE;
            out->size = 2.0;
            cube.GetSizeAttr().Get(&out->size);
            return 1;
        }
        if (UsdGeomSphere sphere{prim->prim}) {
            out->type = USD_SHIM_GPRIM_SPHERE;
            out->radius = 1.0;
            sphere.GetRadiusAttr().Get(&out->radius);
            return 1;
        }
        if (UsdGeomCylinder cyl{prim->prim}) {
            out->type = USD_SHIM_GPRIM_CYLINDER;
            double radius = 1.0, height = 2.0;
            cyl.GetRadiusAttr().Get(&radius);
            cyl.GetHeightAttr().Get(&height);
            out->radius = radius;
            out->radius_bottom = radius;
            out->radius_top = radius;
            out->height = height;
            out->axis = read_axis(cyl.GetAxisAttr());
            return 1;
        }
        // Cylinder_1 is the same frustum with the two radii authored apart.
        if (UsdGeomCylinder_1 cyl1{prim->prim}) {
            out->type = USD_SHIM_GPRIM_CYLINDER;
            out->radius_bottom = 1.0;
            out->radius_top = 1.0;
            out->height = 2.0;
            cyl1.GetRadiusBottomAttr().Get(&out->radius_bottom);
            cyl1.GetRadiusTopAttr().Get(&out->radius_top);
            cyl1.GetHeightAttr().Get(&out->height);
            out->radius = out->radius_bottom;
            out->axis = read_axis(cyl1.GetAxisAttr());
            return 1;
        }
        if (UsdGeomCone cone{prim->prim}) {
            out->type = USD_SHIM_GPRIM_CONE;
            double radius = 1.0, height = 2.0;
            cone.GetRadiusAttr().Get(&radius);
            cone.GetHeightAttr().Get(&height);
            out->radius = radius;
            out->radius_bottom = radius;
            out->radius_top = 0.0; // the apex
            out->height = height;
            out->axis = read_axis(cone.GetAxisAttr());
            return 1;
        }
        if (UsdGeomCapsule capsule{prim->prim}) {
            out->type = USD_SHIM_GPRIM_CAPSULE;
            out->radius = 0.5;
            out->height = 1.0;
            capsule.GetRadiusAttr().Get(&out->radius);
            capsule.GetHeightAttr().Get(&out->height);
            out->radius_bottom = out->radius;
            out->radius_top = out->radius;
            out->axis = read_axis(capsule.GetAxisAttr());
            return 1;
        }
        return 0;
    } catch (const std::exception&) {
        std::memset(out, 0, sizeof(UsdShimGprimData));
        return 0;
    } catch (...) {
        std::memset(out, 0, sizeof(UsdShimGprimData));
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Material extraction. Two surface shaders are understood:
//
//   UsdPreviewSurface  -- the universal render context. Its inputs connect
//                         directly to UsdUVTexture nodes.
//   ND_standard_surface_surfaceshader -- MaterialX, under the "mtlx" render
//                         context. What Houdini (and most DCCs with a
//                         MaterialX path) emit. Its inputs rarely connect
//                         straight to an image: separate3/normalmap/multiply/
//                         invert nodes sit in between.
//
// Both are handled by walking from a surface input down through a small set
// of known passthrough nodes until a node with an `inputs:file` asset is
// reached. The walk accumulates (a) which channel of that image feeds the
// input, and (b) an affine `scale * sample + bias` for the arithmetic nodes
// crossed on the way, so a `0.84 - arm.g` roughness survives the trip.
// ---------------------------------------------------------------------------

namespace {

struct TexRef {
    bool found = false;
    std::string path;
    int channel = 0;        // UsdShimChannel; which channel drives a scalar
    bool channel_known = false;
    float scale = 1.0f;     // value = scale * sample + bias
    float bias = 0.0f;
    GfVec3f tint = GfVec3f(1.0f, 1.0f, 1.0f); // color inputs only
};

// The name of the output a connection arrives by tells us which channel of
// the upstream node is being read: UsdUVTexture exposes r/g/b/a/rgb,
// MaterialX separateN exposes outx/outy/outz/outw (vector) or outr/outg/outb
// (color). A whole-value output ("out", "rgb") carries no channel.
bool channel_from_output_name(const TfToken& name, int* out_channel) {
    const std::string& s = name.GetString();
    if (s == "r" || s == "outr" || s == "outx") { *out_channel = USD_SHIM_CHANNEL_R; return true; }
    if (s == "g" || s == "outg" || s == "outy") { *out_channel = USD_SHIM_CHANNEL_G; return true; }
    if (s == "b" || s == "outb" || s == "outz") { *out_channel = USD_SHIM_CHANNEL_B; return true; }
    if (s == "a" || s == "outa" || s == "outw") { *out_channel = USD_SHIM_CHANNEL_A; return true; }
    return false;
}

bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

// Reads an input's authored value as a float, whatever numeric type it
// carries. MaterialX authors `in2` of a multiply as float/color3f/vector3f
// depending on the node variant.
bool get_input_as_float(const UsdShadeInput& input, float* out) {
    if (!input) return false;
    float f;
    if (input.Get(&f)) { *out = f; return true; }
    GfVec3f v3;
    if (input.Get(&v3)) { *out = v3[0]; return true; }
    return false;
}

bool get_input_as_vec3(const UsdShadeInput& input, GfVec3f* out) {
    if (!input) return false;
    GfVec3f v3;
    if (input.Get(&v3)) { *out = v3; return true; }
    float f;
    if (input.Get(&f)) { *out = GfVec3f(f, f, f); return true; }
    return false;
}

// Reads an asset-valued input, following connections. An image node's `file`
// is often not authored on the node at all: Houdini hoists the texture onto
// the Material prim as an interface input (`inputs:base_color_map`) and
// connects `file` up to it, so one network can be reused across variants.
// GetValueProducingAttributes walks that chain to the attribute holding the
// value.
bool get_asset_input(const UsdShadeInput& input, SdfAssetPath* out) {
    if (!input) return false;
    if (!input.HasConnectedSource()) return input.Get(out);
    for (const UsdAttribute& attr : input.GetValueProducingAttributes()) {
        if (attr.Get(out)) return true;
    }
    return false;
}

// An image node is anything exposing an asset-valued `file` input:
// UsdUVTexture and every MaterialX ND_image_* variant qualify. Matching on
// the input rather than the shader id keeps this working for the renderer-
// specific image nodes DCCs sometimes substitute.
bool read_image_node(const UsdShadeShader& node, const TfToken& arrived_by, TexRef* ref) {
    UsdShadeInput file_input = node.GetInput(TfToken("file"));
    if (!file_input) return false;
    SdfAssetPath asset_path;
    if (!get_asset_input(file_input, &asset_path)) return false;

    // Prefer the resolved path: for a texture packaged inside a .usdz it
    // is the only form that identifies the archive entry
    // (".../model.usdz[0/tex.jpg]"), and it is what usd_shim_read_asset
    // expects. Fall back to the authored path when the resolver could not
    // resolve it.
    std::string p = asset_path.GetResolvedPath();
    if (p.empty()) p = asset_path.GetAssetPath();
    if (p.empty()) return false;

    if (!ref->channel_known) {
        int ch = 0;
        if (channel_from_output_name(arrived_by, &ch)) {
            ref->channel = ch;
            ref->channel_known = true;
        }
    }

    // UsdUVTexture's own scale/bias remap the sampled value. Fold the
    // channel's row into the accumulated transform.
    GfVec4f uv_scale(1.0f, 1.0f, 1.0f, 1.0f);
    GfVec4f uv_bias(0.0f, 0.0f, 0.0f, 0.0f);
    UsdShadeInput scale_input = node.GetInput(TfToken("scale"));
    UsdShadeInput bias_input = node.GetInput(TfToken("bias"));
    bool has_scale = scale_input && scale_input.Get(&uv_scale);
    bool has_bias = bias_input && bias_input.Get(&uv_bias);
    if (has_scale || has_bias) {
        const int ch = ref->channel;
        ref->bias += ref->scale * uv_bias[ch];
        ref->scale *= uv_scale[ch];
        ref->tint = GfVec3f(ref->tint[0] * uv_scale[0],
                            ref->tint[1] * uv_scale[1],
                            ref->tint[2] * uv_scale[2]);
    }

    ref->path = p;
    ref->found = true;
    return true;
}

// Follows `input` upstream to the image node that ultimately feeds it,
// crossing the passthrough/arithmetic nodes MaterialX inserts. Returns
// false when the input is unconnected or the chain ends somewhere we
// don't understand.
bool resolve_texture(const UsdShadeInput& input, TexRef* ref) {
    if (!input) return false;

    UsdShadeConnectableAPI source;
    TfToken arrived_by;
    UsdShadeAttributeType source_type;
    if (!input.GetConnectedSource(&source, &arrived_by, &source_type)) return false;

    UsdPrim prim = source.GetPrim();

    // Bounded so a cyclic or unexpectedly deep network can't hang the load.
    for (int depth = 0; depth < 8; depth++) {
        UsdShadeShader node(prim);
        if (!node) return false;

        if (read_image_node(node, arrived_by, ref)) return true;

        TfToken id;
        node.GetShaderId(&id);
        const std::string& sid = id.GetString();

        // The input to follow, plus any transform this node applies to the
        // value flowing through it.
        TfToken next_input_name;

        if (starts_with(sid, "ND_separate")) {
            // Splits a vector/color into channels. Which channel we take is
            // named by the output we arrived through.
            int ch = 0;
            if (channel_from_output_name(arrived_by, &ch)) {
                ref->channel = ch;
                ref->channel_known = true;
            }
            next_input_name = TfToken("in");
        } else if (starts_with(sid, "ND_normalmap")) {
            // Decodes a tangent-space normal map. Lumbre's shader does that
            // decode itself, so pass the raw image through unchanged.
            next_input_name = TfToken("in");
        } else if (starts_with(sid, "ND_invert")) {
            // out = amount - in. Composed onto the accumulated transform,
            // which maps this node's output to the surface input value:
            // A(amount - v) = -scale*v + (scale*amount + bias).
            // `tint` has no bias term to absorb the amount, so an inverted
            // colour input keeps its tint and loses the inversion. Only
            // scalars (roughness from a gloss map) invert in practice.
            float amount = 1.0f;
            get_input_as_float(node.GetInput(TfToken("amount")), &amount);
            ref->bias += ref->scale * amount;
            ref->scale = -ref->scale;
            next_input_name = TfToken("in");
        } else if (starts_with(sid, "ND_multiply") || starts_with(sid, "ND_divide")) {
            // One side carries the upstream image, the other a constant.
            // Fold the constant in and keep walking the connected side.
            const bool is_divide = starts_with(sid, "ND_divide");
            UsdShadeInput in1 = node.GetInput(TfToken("in1"));
            UsdShadeInput in2 = node.GetInput(TfToken("in2"));
            UsdShadeInput connected = in1;
            UsdShadeInput constant = in2;
            if (in1 && !in1.HasConnectedSource() && in2 && in2.HasConnectedSource()) {
                connected = in2;
                constant = in1;
            }
            if (!connected) return false;

            float k = 1.0f;
            GfVec3f kv(1.0f, 1.0f, 1.0f);
            get_input_as_float(constant, &k);
            get_input_as_vec3(constant, &kv);
            if (is_divide) {
                k = (k != 0.0f) ? 1.0f / k : 1.0f;
                for (int i = 0; i < 3; i++) kv[i] = (kv[i] != 0.0f) ? 1.0f / kv[i] : 1.0f;
            }
            // A(k*v) = (scale*k)*v + bias, so only the scale moves.
            ref->scale *= k;
            ref->tint = GfVec3f(ref->tint[0] * kv[0], ref->tint[1] * kv[1], ref->tint[2] * kv[2]);
            next_input_name = connected.GetBaseName();
        } else if (starts_with(sid, "ND_convert") || starts_with(sid, "ND_swizzle")) {
            next_input_name = TfToken("in");
        } else {
            return false; // a node we don't model; leave the input constant
        }

        UsdShadeInput next = node.GetInput(next_input_name);
        if (!next) return false;
        UsdShadeConnectableAPI next_source;
        UsdShadeAttributeType next_type;
        if (!next.GetConnectedSource(&next_source, &arrived_by, &next_type)) return false;
        prim = next_source.GetPrim();
    }
    return false;
}

// Reads a color-valued surface input. When a texture drives it the constant
// is left as a tint (1,1,1 unless an arithmetic node supplied one), because
// Lumbre multiplies the factor by the sampled texel.
void read_color_input(const UsdShadeShader& shader, const TfToken& name,
                      float out3[3], int* has_tex, char* tex_buf, int tex_buf_len) {
    *has_tex = 0;
    UsdShadeInput input = shader.GetInput(name);
    if (!input) return;

    TexRef ref;
    if (resolve_texture(input, &ref)) {
        std::snprintf(tex_buf, static_cast<size_t>(tex_buf_len), "%s", ref.path.c_str());
        out3[0] = ref.tint[0];
        out3[1] = ref.tint[1];
        out3[2] = ref.tint[2];
        *has_tex = 1;
        return;
    }
    if (input.HasConnectedSource()) return; // connected, but not to anything we read

    GfVec3f value;
    if (input.Get(&value)) {
        out3[0] = value[0];
        out3[1] = value[1];
        out3[2] = value[2];
    }
}

// Reads a scalar-valued surface input. `out_channel` receives the texture
// channel the value is read from; the caller bakes `scale`/`bias` into the
// packed map it builds, so the constant is left at 1.0 when textured.
void read_scalar_input(const UsdShadeShader& shader, const TfToken& name,
                       float* out, int* has_tex, char* tex_buf, int tex_buf_len,
                       int* out_channel, float* out_scale, float* out_bias) {
    *has_tex = 0;
    UsdShadeInput input = shader.GetInput(name);
    if (!input) return;

    TexRef ref;
    if (resolve_texture(input, &ref)) {
        std::snprintf(tex_buf, static_cast<size_t>(tex_buf_len), "%s", ref.path.c_str());
        *out_channel = ref.channel;
        *out_scale = ref.scale;
        *out_bias = ref.bias;
        *out = 1.0f;
        *has_tex = 1;
        return;
    }
    if (input.HasConnectedSource()) return;

    get_input_as_float(input, out);
}

// Input names differ between the two surface shaders; everything downstream
// of the terminal is identical.
struct SurfaceInputs {
    TfToken base_color;
    TfToken roughness;
    TfToken metallic;
    TfToken opacity;
    TfToken normal;
    TfToken emissive_color;
    TfToken emissive_scale; // MaterialX only; multiplies emission_color
};

SurfaceInputs surface_inputs_for(const UsdShadeShader& surface) {
    TfToken id;
    surface.GetShaderId(&id);
    if (starts_with(id.GetString(), "ND_standard_surface")) {
        return SurfaceInputs{
            TfToken("base_color"), TfToken("specular_roughness"), TfToken("metalness"),
            TfToken("opacity"), TfToken("normal"), TfToken("emission_color"), TfToken("emission"),
        };
    }
    return SurfaceInputs{
        TfToken("diffuseColor"), TfToken("roughness"), TfToken("metallic"),
        TfToken("opacity"), TfToken("normal"), TfToken("emissiveColor"), TfToken(),
    };
}

// Finds the surface terminal, preferring UsdPreviewSurface when a material
// authors both (assets exported for several renderers commonly do). The
// preview surface is the simpler, better-specified network of the two.
UsdShadeShader compute_surface(const UsdShadeMaterial& material) {
    UsdShadeShader surface = material.ComputeSurfaceSource();
    if (surface) return surface;
    return material.ComputeSurfaceSource(TfToken("mtlx"));
}

} // namespace

// Shared by usd_shim_get_bound_material and the per-subset material read.
// `prim` may be a Mesh or a materialBind GeomSubset -- ComputeBoundMaterial
// works on either.
static int read_bound_material(const UsdPrim& prim, UsdShimMaterialData* out) {
    std::memset(out, 0, sizeof(UsdShimMaterialData));

    UsdShadeMaterialBindingAPI binding_api(prim);
    UsdShadeMaterial material = binding_api.ComputeBoundMaterial();
    if (!material) return 0;

    UsdShadeShader surface = compute_surface(material);
    if (!surface) return 0;

    std::snprintf(out->material_path, sizeof(out->material_path), "%s",
                  material.GetPrim().GetPath().GetText());

    const SurfaceInputs names = surface_inputs_for(surface);

    out->roughness = 0.5f;
    out->metallic = 0.0f;
    out->opacity = 1.0f;
    out->roughness_tex_scale = 1.0f;
    out->metallic_tex_scale = 1.0f;
    // Neutral fallback for a base color that turns out to be
    // texture-connected: read_color_input overwrites this with the tint
    // (usually white) when it finds a texture, so a texture that later
    // fails to load leaves an unlit-looking grey rather than pure black.
    out->base_color[0] = out->base_color[1] = out->base_color[2] = 0.8f;

    read_color_input(surface, names.base_color, out->base_color,
                     &out->has_base_color_tex,
                     out->base_color_tex, sizeof(out->base_color_tex));
    read_scalar_input(surface, names.roughness, &out->roughness,
                      &out->has_roughness_tex,
                      out->roughness_tex, sizeof(out->roughness_tex),
                      &out->roughness_tex_channel,
                      &out->roughness_tex_scale, &out->roughness_tex_bias);
    read_scalar_input(surface, names.metallic, &out->metallic,
                      &out->has_metallic_tex,
                      out->metallic_tex, sizeof(out->metallic_tex),
                      &out->metallic_tex_channel,
                      &out->metallic_tex_scale, &out->metallic_tex_bias);

    // opacity has no texture slot in UsdShimMaterialData; read the scalar
    // value only (textured opacity is out of scope).
    {
        UsdShadeInput opacity_input = surface.GetInput(names.opacity);
        if (opacity_input && !opacity_input.HasConnectedSource()) {
            get_input_as_float(opacity_input, &out->opacity);
        }
    }

    // Normal map: only a connection is meaningful, a constant `normal`
    // value would just restate the geometric normal.
    {
        int has_tex = 0;
        float ignored[3] = {0, 0, 0};
        read_color_input(surface, names.normal, ignored, &has_tex,
                         out->normal_tex, sizeof(out->normal_tex));
        out->has_normal_tex = has_tex;
    }

    read_color_input(surface, names.emissive_color, out->emissive_color,
                     &out->has_emissive_tex,
                     out->emissive_tex, sizeof(out->emissive_tex));

    // MaterialX splits emission into a colour and a scalar strength; fold
    // the strength in so the caller sees one emissive colour either way.
    if (!names.emissive_scale.IsEmpty()) {
        UsdShadeInput scale_input = surface.GetInput(names.emissive_scale);
        float emission_scale = 0.0f;
        if (scale_input && get_input_as_float(scale_input, &emission_scale)) {
            for (int i = 0; i < 3; i++) out->emissive_color[i] *= emission_scale;
        } else if (!out->has_emissive_tex) {
            // `emission` unauthored means no emission at all, whatever
            // emission_color says.
            out->emissive_color[0] = out->emissive_color[1] = out->emissive_color[2] = 0.0f;
        }
    }

    return 1;
}

extern "C" int usd_shim_get_bound_material(UsdShimPrimHandle prim, UsdShimMaterialData* out) {
    if (!prim || !out) return 0;
    try {
        return read_bound_material(prim->prim, out);
    } catch (...) {
        std::memset(out, 0, sizeof(UsdShimMaterialData));
        return 0;
    }
}

// Collects the mesh's "materialBind"-family face subsets. A DCC that
// assigns several materials to one mesh (per-region texture sets) encodes
// them this way, leaving the Mesh prim itself with no binding at all.
static std::vector<UsdGeomSubset> material_bind_subsets(const UsdPrim& prim) {
    UsdGeomImageable imageable(prim);
    if (!imageable) return {};
    return UsdGeomSubset::GetGeomSubsets(imageable, UsdGeomTokens->face,
                                          UsdShadeTokens->materialBind);
}

extern "C" int usd_shim_get_subset_count(UsdShimPrimHandle mesh) {
    if (!mesh) return 0;
    try {
        return static_cast<int>(material_bind_subsets(mesh->prim).size());
    } catch (...) {
        return 0;
    }
}

extern "C" int usd_shim_get_subsets(UsdShimPrimHandle mesh, UsdShimSubsetData* out, int max) {
    if (!mesh || !out || max <= 0) return 0;
    try {
        const std::vector<UsdGeomSubset> subsets = material_bind_subsets(mesh->prim);
        int written = 0;
        for (const UsdGeomSubset& subset : subsets) {
            if (written >= max) break;
            UsdShimSubsetData* dst = &out[written];
            std::memset(dst, 0, sizeof(UsdShimSubsetData));

            VtArray<int> indices;
            if (!subset.GetIndicesAttr().Get(&indices) || indices.empty()) continue;

            dst->face_indices = static_cast<int*>(std::malloc(sizeof(int) * indices.size()));
            if (!dst->face_indices) continue;
            std::memcpy(dst->face_indices, indices.cdata(), sizeof(int) * indices.size());
            dst->face_index_count = static_cast<int>(indices.size());

            dst->has_material = read_bound_material(subset.GetPrim(), &dst->material);
            written++;
        }
        return written;
    } catch (...) {
        return 0;
    }
}

extern "C" void usd_shim_free_subsets(UsdShimSubsetData* subsets, int count) {
    if (!subsets) return;
    for (int i = 0; i < count; i++) {
        std::free(subsets[i].face_indices);
        subsets[i].face_indices = nullptr;
        subsets[i].face_index_count = 0;
    }
}

extern "C" unsigned char* usd_shim_read_asset(const char* resolved_path, size_t* out_size) {
    if (out_size) *out_size = 0;
    if (!resolved_path || !out_size) return nullptr;
    try {
        // OpenAsset goes through the resolver rather than the filesystem,
        // so it transparently reads entries out of a .usdz archive.
        std::shared_ptr<ArAsset> asset = ArGetResolver().OpenAsset(ArResolvedPath(resolved_path));
        if (!asset) return nullptr;

        const size_t size = asset->GetSize();
        if (size == 0) return nullptr;

        std::shared_ptr<const char> buffer = asset->GetBuffer();
        if (!buffer) return nullptr;

        // Copy out of USD's buffer so ownership crossing the boundary is
        // a plain malloc block, freed by usd_shim_free_asset.
        unsigned char* out = static_cast<unsigned char*>(std::malloc(size));
        if (!out) return nullptr;
        std::memcpy(out, buffer.get(), size);

        *out_size = size;
        return out;
    } catch (...) {
        if (out_size) *out_size = 0;
        return nullptr;
    }
}

extern "C" void usd_shim_free_asset(unsigned char* data) {
    std::free(data);
}

extern "C" const char* usd_shim_resolve_asset_path(UsdShimStageHandle stage, const char* asset_path) {
    if (!stage || !asset_path) return "";
    try {
        SdfLayerHandle layer = stage->stage->GetRootLayer();
        std::string anchored = SdfComputeAssetPathRelativeToLayer(layer, asset_path);
        stage->resolve_scratch = anchored;
        return stage->resolve_scratch.c_str();
    } catch (...) {
        return asset_path;
    }
}

// ---------------------------------------------------------------------------
// Camera. All attributes are `float`/`float2` per the schema (not `double`),
// confirmed against usdGeom's generated schema -- Get()ing into the wrong
// numeric type silently fails and leaves the caller's default in place.
// ---------------------------------------------------------------------------

extern "C" int usd_shim_get_camera_data(UsdShimPrimHandle prim, UsdShimCameraData* out) {
    if (!prim || !out) return 0;
    std::memset(out, 0, sizeof(UsdShimCameraData));
    try {
        UsdGeomCamera camera(prim->prim);
        if (!camera) return 0;

        float focal_length = 50.0f;
        camera.GetFocalLengthAttr().Get(&focal_length);
        out->focal_length_mm = focal_length;

        float h_aperture = 20.955f, v_aperture = 15.2908f;
        camera.GetHorizontalApertureAttr().Get(&h_aperture);
        camera.GetVerticalApertureAttr().Get(&v_aperture);
        out->horizontal_aperture_mm = h_aperture;
        out->vertical_aperture_mm = v_aperture;

        GfVec2f clip_range(1.0f, 1000000.0f);
        camera.GetClippingRangeAttr().Get(&clip_range);
        out->clipping_range[0] = clip_range[0];
        out->clipping_range[1] = clip_range[1];

        float focus_distance = 0.0f;
        camera.GetFocusDistanceAttr().Get(&focus_distance);
        out->focus_distance = focus_distance;

        float f_stop = 0.0f;
        camera.GetFStopAttr().Get(&f_stop);
        out->f_stop = f_stop;

        return 1;
    } catch (const std::exception&) {
        std::memset(out, 0, sizeof(UsdShimCameraData));
        return 0;
    } catch (...) {
        std::memset(out, 0, sizeof(UsdShimCameraData));
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Lights (UsdLux). Every concrete light schema (SphereLight, RectLight, ...)
// inherits GetIntensityAttr/GetExposureAttr/GetColorAttr/GetNormalizeAttr
// directly from its LightAPI-forwarding base (BoundableLightBase or
// NonboundableLightBase), so they're called straight on the concrete schema
// object below rather than through a separate UsdLuxLightAPI(prim).
// ---------------------------------------------------------------------------

namespace {

template <typename LightSchema>
void read_common_light(const UsdPrim& prim, const LightSchema& light, UsdShimLightData* out) {
    float intensity = 1.0f;
    light.GetIntensityAttr().Get(&intensity);
    out->intensity = intensity;

    float exposure = 0.0f;
    light.GetExposureAttr().Get(&exposure);
    out->exposure = exposure;

    GfVec3f color(1.0f, 1.0f, 1.0f);
    light.GetColorAttr().Get(&color);
    out->color[0] = color[0];
    out->color[1] = color[1];
    out->color[2] = color[2];

    bool normalize = false;
    light.GetNormalizeAttr().Get(&normalize);
    out->normalize = normalize ? 1 : 0;

    // ShapingAPI (a spot cone) is only meaningful applied to a point-like
    // light; SphereLight is the only one Lumbre maps to Point/Spot.
    if (prim.HasAPI<UsdLuxShapingAPI>()) {
        UsdLuxShapingAPI shaping(prim);
        float cone_angle = 90.0f, cone_softness = 0.0f;
        shaping.GetShapingConeAngleAttr().Get(&cone_angle);
        shaping.GetShapingConeSoftnessAttr().Get(&cone_softness);
        out->has_shaping = 1;
        out->shaping_cone_angle = cone_angle;
        out->shaping_cone_softness = cone_softness;
    }
}

} // namespace

extern "C" int usd_shim_get_light_data(UsdShimPrimHandle prim, UsdShimLightData* out) {
    if (!prim || !out) return 0;
    std::memset(out, 0, sizeof(UsdShimLightData));
    out->color[0] = out->color[1] = out->color[2] = 1.0f;
    try {
        if (UsdLuxSphereLight light{prim->prim}) {
            out->kind = USD_SHIM_LIGHT_SPHERE;
            read_common_light(prim->prim, light, out);
            float radius = 0.5f;
            light.GetRadiusAttr().Get(&radius);
            out->radius = radius;
            bool treat_as_point = false;
            light.GetTreatAsPointAttr().Get(&treat_as_point);
            out->treat_as_point = (treat_as_point || radius <= 0.0f) ? 1 : 0;
            return 1;
        }
        if (UsdLuxRectLight light{prim->prim}) {
            out->kind = USD_SHIM_LIGHT_RECT;
            read_common_light(prim->prim, light, out);
            float width = 1.0f, height = 1.0f;
            light.GetWidthAttr().Get(&width);
            light.GetHeightAttr().Get(&height);
            out->width = width;
            out->height = height;
            return 1;
        }
        if (UsdLuxDiskLight light{prim->prim}) {
            out->kind = USD_SHIM_LIGHT_DISK;
            read_common_light(prim->prim, light, out);
            float radius = 0.5f;
            light.GetRadiusAttr().Get(&radius);
            out->radius = radius;
            return 1;
        }
        if (UsdLuxCylinderLight light{prim->prim}) {
            out->kind = USD_SHIM_LIGHT_CYLINDER;
            read_common_light(prim->prim, light, out);
            float radius = 0.5f, length = 1.0f;
            light.GetRadiusAttr().Get(&radius);
            light.GetLengthAttr().Get(&length);
            out->radius = radius;
            out->length = length;
            return 1;
        }
        if (UsdLuxDistantLight light{prim->prim}) {
            out->kind = USD_SHIM_LIGHT_DISTANT;
            read_common_light(prim->prim, light, out);
            float angle = 0.53f; // degrees; USD default ~= the sun's angular diameter
            light.GetAngleAttr().Get(&angle);
            out->angle = angle;
            return 1;
        }
        if (UsdLuxDomeLight light{prim->prim}) {
            out->kind = USD_SHIM_LIGHT_DOME;
            read_common_light(prim->prim, light, out);
            SdfAssetPath asset_path;
            if (light.GetTextureFileAttr().Get(&asset_path)) {
                std::string p = asset_path.GetResolvedPath();
                if (p.empty()) p = asset_path.GetAssetPath();
                std::snprintf(out->texture_file, sizeof(out->texture_file), "%s", p.c_str());
            }
            return 1;
        }
        return 0;
    } catch (const std::exception&) {
        std::memset(out, 0, sizeof(UsdShimLightData));
        return 0;
    } catch (...) {
        std::memset(out, 0, sizeof(UsdShimLightData));
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Owner lookup used by usd_shim_get_children (see comment there). A UsdPrim
// carries a UsdStageWeakPtr but not a pointer back to our wrapper, so we keep
// a small static registry populated on open/close.
// ---------------------------------------------------------------------------

#include <unordered_map>
#include <mutex>

namespace {
std::mutex g_registry_mutex;
std::unordered_map<UsdStage*, UsdShimStage*> g_registry;
}

static void usd_shim_register_owner(UsdStage* raw_stage, UsdShimStage* owner) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_registry[raw_stage] = owner;
}

static void usd_shim_unregister_owner(UsdStage* raw_stage) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_registry.erase(raw_stage);
}

static UsdShimStage* usd_shim_find_owner(const UsdStageWeakPtr& stage) {
    if (!stage) return nullptr;
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_registry.find(stage.operator->());
    return it == g_registry.end() ? nullptr : it->second;
}
