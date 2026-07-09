#include "usd_shim.h"

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layerUtils.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/vt/array.h>

#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
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

        // UVs: look for a "st" or "uv" primvar, the de-facto standard names.
        UsdGeomPrimvarsAPI primvars_api(prim->prim);
        UsdGeomPrimvar uv_primvar = primvars_api.GetPrimvar(TfToken("st"));
        if (!uv_primvar.HasValue()) uv_primvar = primvars_api.GetPrimvar(TfToken("uv"));
        if (uv_primvar.HasValue()) {
            VtArray<GfVec2f> uvs;
            if (uv_primvar.Get(&uvs) && !uvs.empty()) {
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
// Material extraction: UsdPreviewSurface only for v1. A shader input is
// either a bound scalar/color value, or connected to a UsdUVTexture node
// whose "file" input gives the texture asset path.
// ---------------------------------------------------------------------------

static bool read_shader_color_input(const UsdShadeShader& shader, const TfToken& name,
                                     float out3[3], int* has_tex, char* tex_buf, int tex_buf_len) {
    *has_tex = 0;
    UsdShadeInput input = shader.GetInput(name);
    if (!input) return false;

    UsdShadeConnectableAPI source;
    TfToken source_name;
    UsdShadeAttributeType source_type;
    if (input.GetConnectedSource(&source, &source_name, &source_type)) {
        UsdShadeShader tex_shader(source.GetPrim());
        if (tex_shader) {
            UsdShadeInput file_input = tex_shader.GetInput(TfToken("file"));
            SdfAssetPath asset_path;
            if (file_input && file_input.Get(&asset_path)) {
                std::string p = asset_path.GetAssetPath();
                std::snprintf(tex_buf, static_cast<size_t>(tex_buf_len), "%s", p.c_str());
                *has_tex = 1;
                return true;
            }
        }
        return false;
    }

    GfVec3f value;
    if (input.Get(&value)) {
        out3[0] = value[0];
        out3[1] = value[1];
        out3[2] = value[2];
        return true;
    }
    return false;
}

static bool read_shader_scalar_input(const UsdShadeShader& shader, const TfToken& name,
                                      float* out, int* has_tex, char* tex_buf, int tex_buf_len) {
    *has_tex = 0;
    UsdShadeInput input = shader.GetInput(name);
    if (!input) return false;

    UsdShadeConnectableAPI source;
    TfToken source_name;
    UsdShadeAttributeType source_type;
    if (input.GetConnectedSource(&source, &source_name, &source_type)) {
        UsdShadeShader tex_shader(source.GetPrim());
        if (tex_shader) {
            UsdShadeInput file_input = tex_shader.GetInput(TfToken("file"));
            SdfAssetPath asset_path;
            if (file_input && file_input.Get(&asset_path)) {
                std::string p = asset_path.GetAssetPath();
                std::snprintf(tex_buf, static_cast<size_t>(tex_buf_len), "%s", p.c_str());
                *has_tex = 1;
                return true;
            }
        }
        return false;
    }

    float value;
    if (input.Get(&value)) {
        *out = value;
        return true;
    }
    return false;
}

extern "C" int usd_shim_get_bound_material(UsdShimPrimHandle prim, UsdShimMaterialData* out) {
    if (!prim || !out) return 0;
    std::memset(out, 0, sizeof(UsdShimMaterialData));
    try {
        UsdShadeMaterialBindingAPI binding_api(prim->prim);
        UsdShadeMaterial material = binding_api.ComputeBoundMaterial();
        if (!material) return 0;

        UsdShadeShader surface = material.ComputeSurfaceSource();
        if (!surface) return 0;

        out->roughness = 0.5f;
        out->metallic = 0.0f;
        out->opacity = 1.0f;
        // Neutral fallbacks for color inputs that turn out to be
        // texture-connected: read_shader_color_input leaves out3
        // untouched when it finds a texture (only the *_tex path is
        // filled in), so without this the caller sees pure black
        // whenever the texture itself fails to load (e.g. a .usdz
        // packed-asset path that our texture loader can't unpack yet).
        out->base_color[0] = out->base_color[1] = out->base_color[2] = 0.8f;

        read_shader_color_input(surface, TfToken("diffuseColor"), out->base_color,
                                 &out->has_base_color_tex,
                                 out->base_color_tex, sizeof(out->base_color_tex));
        read_shader_scalar_input(surface, TfToken("roughness"), &out->roughness,
                                  &out->has_roughness_tex,
                                  out->roughness_tex, sizeof(out->roughness_tex));
        read_shader_scalar_input(surface, TfToken("metallic"), &out->metallic,
                                  &out->has_metallic_tex,
                                  out->metallic_tex, sizeof(out->metallic_tex));
        // opacity has no texture slot in UsdShimMaterialData; read the
        // scalar value only (textured opacity is out of scope for v1).
        {
            UsdShadeInput opacity_input = surface.GetInput(TfToken("opacity"));
            float opacity_value;
            if (opacity_input && opacity_input.Get(&opacity_value)) {
                out->opacity = opacity_value;
            }
        }
        // Normal map: value-only inputs don't apply, only look for connection.
        {
            int dummy_has_tex = 0;
            float dummy3[3] = {0, 0, 0};
            read_shader_color_input(surface, TfToken("normal"), dummy3,
                                     &dummy_has_tex, out->normal_tex, sizeof(out->normal_tex));
            out->has_normal_tex = dummy_has_tex;
        }
        read_shader_color_input(surface, TfToken("emissiveColor"), out->emissive_color,
                                 &out->has_emissive_tex,
                                 out->emissive_tex, sizeof(out->emissive_tex));

        return 1;
    } catch (...) {
        std::memset(out, 0, sizeof(UsdShimMaterialData));
        return 0;
    }
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
