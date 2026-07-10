// extern "C" boundary between Lumbre (Odin) and OpenUSD (C++).
// No C++ types cross this header; every function catches all exceptions
// internally and reports failure via return code, never via a thrown
// exception (throwing across a `foreign import` boundary is undefined
// behavior in Odin).
#ifndef LUMBRE_USD_SHIM_H
#define LUMBRE_USD_SHIM_H

#include <stddef.h> // size_t

#ifdef __cplusplus
extern "C" {
#endif

typedef struct UsdShimStage UsdShimStage;
typedef struct UsdShimPrim UsdShimPrim;
typedef UsdShimStage* UsdShimStageHandle;
typedef UsdShimPrim* UsdShimPrimHandle;

// Interpolation mode for normals/UVs, mirrors UsdGeom's TfToken values.
enum UsdShimInterp {
    USD_SHIM_INTERP_NONE = 0,
    USD_SHIM_INTERP_VERTEX = 1,
    USD_SHIM_INTERP_FACE_VARYING = 2,
    USD_SHIM_INTERP_UNIFORM = 3,
};

// Opens `path`, flattening all composition (references/layers/variants)
// into a single layer via UsdStage::Flatten(). Returns NULL on failure and
// writes a human-readable message into err_buf (if non-NULL).
UsdShimStageHandle usd_shim_open_flattened(const char* path, char* err_buf, int err_buf_len);
void usd_shim_close(UsdShimStageHandle stage);

// Prim handles are owned by the stage; valid until usd_shim_close. Caller
// must NOT free them individually.
UsdShimPrimHandle usd_shim_get_pseudo_root(UsdShimStageHandle stage);

// Writes up to `max` child prim handles into `out`, returns the actual
// child count (may be > max; caller should re-call with a bigger buffer
// if so, though in practice DCC-exported hierarchies are shallow).
int usd_shim_get_children(UsdShimPrimHandle prim, UsdShimPrimHandle* out, int max);

// Returned strings point at shim-internal storage valid until the next
// call on the same prim handle. Copy immediately if needed longer.
const char* usd_shim_prim_type_name(UsdShimPrimHandle prim);
const char* usd_shim_prim_name(UsdShimPrimHandle prim);

// Local (non-inherited) transform as a row-major 4x4 double matrix.
// Returns 1 if the prim is Xformable and a transform was written, 0
// otherwise (out_mat4x4 is left as identity in that case).
int usd_shim_get_local_transform(UsdShimPrimHandle prim, double out_mat4x4[16]);

typedef struct {
    float* points;               // 3 floats per point (x,y,z)
    int point_count;
    int* face_vertex_indices;    // index_count entries, into points[]
    int index_count;
    int* face_vertex_counts;     // face_count entries; N-gon support (>=3)
    int face_count;
    float* normals;              // 3 floats per entry
    int normal_count;
    int normal_interp;           // UsdShimInterp
    float* uvs;                  // 2 floats per entry
    int uv_count;
    int uv_interp;               // UsdShimInterp
} UsdShimMeshData;

// Populates `out` from a UsdGeomMesh prim. Returns 1 on success, 0 if
// `prim` is not a Mesh or has no authored point data. All array fields
// are heap-allocated by the shim; caller must call
// usd_shim_free_mesh_data exactly once when done.
int usd_shim_get_mesh_data(UsdShimPrimHandle prim, UsdShimMeshData* out);
void usd_shim_free_mesh_data(UsdShimMeshData* data);

typedef struct {
    float base_color[3];
    int has_base_color_tex;
    char base_color_tex[1024];
    float roughness;
    int has_roughness_tex;
    char roughness_tex[1024];
    float metallic;
    int has_metallic_tex;
    char metallic_tex[1024];
    float opacity;
    float emissive_color[3];
    int has_emissive_tex;
    char emissive_tex[1024];
    int has_normal_tex;
    char normal_tex[1024];
} UsdShimMaterialData;

// Resolves the UsdShadeMaterial bound to `prim` (via
// UsdShadeMaterialBindingAPI) and reads its UsdPreviewSurface inputs into
// `out`. Returns 1 if a material was found and read, 0 if none is bound
// (out is left zeroed; caller should fall back to a default material).
int usd_shim_get_bound_material(UsdShimPrimHandle prim, UsdShimMaterialData* out);

// Resolves `asset_path` (as authored, e.g. an asset-path-valued shader
// input) against the stage's asset resolver context, relative to the
// layer it was authored in. Returned string points at shim-internal
// storage valid until the next call to this function.
const char* usd_shim_resolve_asset_path(UsdShimStageHandle stage, const char* asset_path);

// Reads an asset's bytes through the USD asset resolver. Unlike opening
// the path with fopen/stb_image, this handles assets packaged inside a
// .usdz archive, whose resolved paths look like
//   /abs/path/model.usdz[0/texture.jpg]
// and are not openable as ordinary files.
//
// `resolved_path` should be a path as produced by the material texture
// fields or usd_shim_resolve_asset_path. Returns NULL on failure. On
// success, *out_size holds the byte count and the caller owns the buffer
// and must release it with usd_shim_free_asset.
unsigned char* usd_shim_read_asset(const char* resolved_path, size_t* out_size);
void usd_shim_free_asset(unsigned char* data);

#ifdef __cplusplus
}
#endif

#endif // LUMBRE_USD_SHIM_H
