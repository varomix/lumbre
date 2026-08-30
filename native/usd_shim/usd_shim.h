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

// Stage-level metadata used to reconcile the stage's coordinate conventions
// with Lumbre's (Y-up). `up_axis` is 0 for Y (the default) or 1 for Z;
// `meters_per_unit` defaults to 0.01 (centimeters) per the USD spec when
// unauthored. Returns 1 on success, 0 (leaving Y-up / 1.0 defaults) if the
// stage handle is null.
typedef struct {
    int up_axis;            // 0 = Y, 1 = Z
    double meters_per_unit; // scene units expressed in meters
} UsdShimStageInfo;
int usd_shim_get_stage_info(UsdShimStageHandle stage, UsdShimStageInfo* out);

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

// ── Stage inspection (GUI) ──────────────────────────────────────────────────
//
// These exist for the viewer's USD tree / text / properties panels. They are
// read-only: nothing here authors to the stage.

// Full scene path, e.g. "/World/Geo/Mesh". Backed by shim-internal storage
// valid until the next call on the same prim handle.
const char* usd_shim_prim_path(UsdShimPrimHandle prim);

// Serialise to .usda text. Unlike the accessors above, the caller OWNS the
// returned buffer and must release it with usd_shim_free_string; these results
// can be megabytes, so they are not kept in per-handle scratch. Returns NULL
// on failure.
// `max_array_elems` drops the value of any array attribute longer than it,
// which keeps a Mesh from exporting its full point and index arrays — one
// guitar mesh is 33 MB otherwise. Pass 0 for the complete text.
char* usd_shim_export_stage_to_string(UsdShimStageHandle stage, int max_array_elems);
// Exports `prim` and its descendants only.
char* usd_shim_export_prim_to_string(UsdShimStageHandle stage, UsdShimPrimHandle prim, int max_array_elems);
void usd_shim_free_string(char* s);

// One authored property (attribute or relationship) for the properties panel.
// All three strings point at shim-internal storage valid until the next
// usd_shim_get_properties call on the same prim handle.
typedef struct {
    const char* name;
    const char* type_name; // "" for relationships
    const char* value;     // formatted default value; "" if unauthored
    int is_relationship;   // 1 for a relationship, 0 for an attribute
} UsdShimProperty;

int usd_shim_get_property_count(UsdShimPrimHandle prim);
// Writes up to `max` entries, returns the number written.
int usd_shim_get_properties(UsdShimPrimHandle prim, UsdShimProperty* out, int max);

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
    // primvars:displayColor, the DCC's viewport tint. Many assets (the Pixar
    // Kitchen_set among them) ship it as their only color and bind no
    // UsdShadeMaterial at all, so the caller uses it as a fallback albedo.
    // Reduced to one constant color (the mean of the authored values) since
    // it feeds a single flat material.
    int has_display_color;       // 1 if primvars:displayColor was authored
    float display_color[3];      // representative (mean) linear color
} UsdShimMeshData;

// Populates `out` from a UsdGeomMesh prim. Returns 1 on success, 0 if
// `prim` is not a Mesh or has no authored point data. All array fields
// are heap-allocated by the shim; caller must call
// usd_shim_free_mesh_data exactly once when done.
//
// `subdiv_level` > 0 refines meshes whose subdivisionScheme is not "none"
// (catmullClark is the schema default) to that many uniform Catmark/Loop/
// bilinear levels, returning the refined surface with smooth normals and
// face-varying UVs. 0 disables refinement and returns the authored cage.
int usd_shim_get_mesh_data(UsdShimPrimHandle prim, UsdShimMeshData* out, int subdiv_level);
void usd_shim_free_mesh_data(UsdShimMeshData* data);

// Parametric gprims (UsdGeomCube/Sphere/Cylinder/Cone/Capsule) carry no
// point data at all -- only the handful of numbers below. The shim reports
// those numbers; the caller tessellates, keeping the shim to marshalling
// exactly as with the mesh triangulation.
typedef enum {
    USD_SHIM_GPRIM_NONE = 0,
    USD_SHIM_GPRIM_CUBE = 1,
    USD_SHIM_GPRIM_SPHERE = 2,
    USD_SHIM_GPRIM_CYLINDER = 3,
    USD_SHIM_GPRIM_CONE = 4,
    USD_SHIM_GPRIM_CAPSULE = 5,
} UsdShimGprimType;

typedef enum {
    USD_SHIM_AXIS_X = 0,
    USD_SHIM_AXIS_Y = 1,
    USD_SHIM_AXIS_Z = 2,
} UsdShimAxis;

typedef struct {
    int type;               // UsdShimGprimType
    int axis;               // UsdShimAxis; the spine. Unused by Cube/Sphere.
    double size;            // Cube: edge length, centered on the origin.
    double radius;          // Sphere, Capsule.
    // Cylinder and Cone are one shape: a frustum spanning +/- height/2 along
    // `axis`. A Cone sets radius_top to 0 (its apex); a plain Cylinder sets
    // both radii to its single `radius`; UsdGeomCylinder_1 authors them
    // separately. Capsule uses `radius` for both its spine and its caps.
    double radius_bottom;
    double radius_top;
    // Cylinder/Cone: the full extent along `axis`. Capsule: the length of the
    // cylindrical section only, excluding the two hemispherical caps, per
    // UsdGeomCapsule -- so the shape spans height + 2*radius.
    double height;
} UsdShimGprimData;

// Returns 1 and fills `out` if `prim` is a parametric gprim this shim knows
// how to describe, 0 otherwise (including for Mesh prims, which go through
// usd_shim_get_mesh_data instead). Nothing is allocated.
int usd_shim_get_gprim_data(UsdShimPrimHandle prim, UsdShimGprimData* out);

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
    float ior;               // default 1.5 (unauthored UsdPreviewSurface/mtlx default)
    float transmission;      // 0 = opaque, 1 = fully transmissive (glass)
    float transmission_color[3]; // tints the transmitted ray; default (1,1,1)
    float coat;               // clearcoat amount, 0-1
    float coat_roughness;
    float specular;           // MaterialX standard_surface lobe weight
    float specular_color[3];  // MaterialX standard_surface lobe tint
    float emissive_color[3];
    int has_emissive_tex;
    char emissive_tex[1024];
    int has_normal_tex;
    char normal_tex[1024];
    // MaterialX standard_surface subsurface inputs.  These are scalar/color
    // values for now; texture-driven SSS is not represented by the renderer.
    float subsurface;
    float subsurface_color[3];
    float subsurface_radius[3];
    float subsurface_scale;
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

// Decodes an image asset to 8-bit RGBA (4 channels, alpha forced opaque)
// via OpenUSD's Hio, which reads any format its plugins support -- notably
// OpenEXR, which stb_image cannot. Resolves package (.usdz) paths through
// Ar, so `asset_path` may be the "/abs/model.usdz[maps/tex.exr]" form.
// `flip_v` flips vertically to match stb_image's load convention. Pixel
// values are read raw (no color-space transform; EXR data is linear) and
// clamped to [0,1]. Returns 1 on success, writing width/height and a
// heap buffer of width*height*4 bytes the caller must release with
// usd_shim_free_image. Returns 0 on any failure (unsupported format,
// unreadable asset), leaving *out_pixels null.
int usd_shim_load_image(const char* asset_path, int flip_v,
                        int* out_w, int* out_h, unsigned char** out_pixels);
void usd_shim_free_image(unsigned char* pixels);

// ---------------------------------------------------------------------------
// Camera. UsdGeomCamera carries only the lens parameters below -- its
// position/orientation come from the same usd_shim_get_local_transform every
// other Xformable prim uses.
// ---------------------------------------------------------------------------

typedef struct {
    float focal_length_mm;
    float horizontal_aperture_mm;
    float vertical_aperture_mm;
    float clipping_range[2];
    float focus_distance;   // 0 = unauthored
    float f_stop;           // 0 = unauthored (no depth of field)
} UsdShimCameraData;

// Returns 1 and fills `out` if `prim` is a UsdGeomCamera, 0 otherwise.
int usd_shim_get_camera_data(UsdShimPrimHandle prim, UsdShimCameraData* out);

// ---------------------------------------------------------------------------
// Lights (UsdLux). One function covers every light schema Lumbre maps onto
// an existing Light_Kind; `kind` tells the caller which shape-specific
// fields below are meaningful. Position/orientation, as with Camera, come
// from usd_shim_get_local_transform.
// ---------------------------------------------------------------------------

typedef enum {
    USD_SHIM_LIGHT_NONE = 0,
    USD_SHIM_LIGHT_SPHERE = 1,
    USD_SHIM_LIGHT_RECT = 2,
    USD_SHIM_LIGHT_DISK = 3,
    USD_SHIM_LIGHT_CYLINDER = 4,
    USD_SHIM_LIGHT_DISTANT = 5,
    USD_SHIM_LIGHT_DOME = 6,
} UsdShimLightKind;

typedef struct {
    int kind; // UsdShimLightKind

    // Common to every UsdLuxLightAPI-backed schema. Radiance = intensity *
    // 2^exposure * color; `normalize` divides by the light's world-space
    // area so brightness stays constant as the shape is resized.
    float intensity;
    float exposure;
    float color[3];
    int normalize;

    // Shape-specific. Only the fields relevant to `kind` are meaningful.
    float width, height;  // RectLight
    float radius;         // Sphere/Disk/CylinderLight
    float length;         // CylinderLight
    float angle;          // DistantLight, degrees (angular diameter)
    int treat_as_point;   // SphereLight: explicit treatAsPoint or radius<=0

    // ShapingAPI (spot cone), meaningful only on a SphereLight.
    int has_shaping;
    float shaping_cone_angle;
    float shaping_cone_softness;

    // DomeLight only: resolved (or, failing that, authored) path to the
    // equirect texture. Empty if unauthored (a constant-color dome).
    char texture_file[1024];
} UsdShimLightData;

// Returns 1 and fills `out` if `prim` is a UsdLux light schema this shim
// understands (out->kind != USD_SHIM_LIGHT_NONE), 0 otherwise.
int usd_shim_get_light_data(UsdShimPrimHandle prim, UsdShimLightData* out);

#ifdef __cplusplus
}
#endif

#endif // LUMBRE_USD_SHIM_H
