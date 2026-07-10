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

// Which channel of a texture drives a scalar input. Scalar inputs fed by
// a packed map (an ARM/ORM texture) read one channel of it; a dedicated
// greyscale map is read as USD_SHIM_CHANNEL_R.
typedef enum {
    USD_SHIM_CHANNEL_R = 0,
    USD_SHIM_CHANNEL_G = 1,
    USD_SHIM_CHANNEL_B = 2,
    USD_SHIM_CHANNEL_A = 3,
} UsdShimChannel;

typedef struct {
    // Stage path of the UsdShadeMaterial these values were read from. The
    // same material is bound by many meshes and subsets; the path is what
    // lets a caller read (and load textures for) each one only once.
    char material_path[512];
    float base_color[3];
    int has_base_color_tex;
    char base_color_tex[1024];
    float roughness;
    int has_roughness_tex;
    char roughness_tex[1024];
    int roughness_tex_channel;   // UsdShimChannel
    float roughness_tex_scale;   // roughness = sample[channel] * scale + bias
    float roughness_tex_bias;
    float metallic;
    int has_metallic_tex;
    char metallic_tex[1024];
    int metallic_tex_channel;    // UsdShimChannel
    float metallic_tex_scale;    // metallic = sample[channel] * scale + bias
    float metallic_tex_bias;
    float opacity;
    float emissive_color[3];
    int has_emissive_tex;
    char emissive_tex[1024];
    int has_normal_tex;
    char normal_tex[1024];
} UsdShimMaterialData;

// Resolves the UsdShadeMaterial bound to `prim` (via
// UsdShadeMaterialBindingAPI) and reads its surface shader's inputs into
// `out`. Both UsdPreviewSurface (universal render context) and MaterialX
// ND_standard_surface_surfaceshader (the "mtlx" render context, what
// Houdini and other DCCs emit) are understood. Returns 1 if a material
// was found and read, 0 if none is bound (out is left zeroed; caller
// should fall back to a default material).
//
// An input that resolves to a texture leaves its constant factor at 1.0,
// since Lumbre multiplies the two: base_color/roughness/metallic/emissive
// are only meaningful as tints when the matching has_*_tex is set.
int usd_shim_get_bound_material(UsdShimPrimHandle prim, UsdShimMaterialData* out);

// A "materialBind"-family face subset: a set of face indices on the mesh
// with its own bound material. This is how a single mesh carries several
// materials (one texture set per region), so a mesh with subsets usually
// has no material bound at the mesh level at all.
typedef struct {
    int* face_indices;      // indices into the mesh's faceVertexCounts
    int face_index_count;
    int has_material;
    UsdShimMaterialData material;
} UsdShimSubsetData;

// Number of materialBind face subsets on `mesh`. 0 for the common case of
// a single whole-mesh material.
int usd_shim_get_subset_count(UsdShimPrimHandle mesh);

// Fills up to `max` subsets. Returns the number written. The caller owns
// the face_indices arrays and must release them with
// usd_shim_free_subsets.
int usd_shim_get_subsets(UsdShimPrimHandle mesh, UsdShimSubsetData* out, int max);
void usd_shim_free_subsets(UsdShimSubsetData* subsets, int count);

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
