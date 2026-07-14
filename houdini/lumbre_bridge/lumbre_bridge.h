#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increment only for intentionally incompatible ABI changes.
#define LUMBRE_HOUDINI_BRIDGE_ABI_VERSION 5u

uint32_t lumbre_bridge_abi_version(void);

typedef struct LumbreBridgeTriangle {
    float positions[9];
    float normals[9];
    float uvs[6];
    int32_t has_uv;
    int32_t material_index;
} LumbreBridgeTriangle;

// A Hydra UsdPreviewSurface or MaterialX standard_surface reduced to the
// renderer's common material, mirroring the CLI's Usd_Shim_Material_Data so the
// bridge builds the same core Imported_Material and shares
// imported_material_to_principled. The delegate fills these from either shader's
// input set (see _LumbreMaterialFromNetwork). Texture paths must be resolved
// absolute paths when possible.
typedef struct LumbreBridgeMaterial {
    float base_color[3];
    float emission[3];
    float emission_strength;
    float metallic;
    float roughness;
    float specular;
    float specular_color[3];
    float ior;
    float opacity;
    float transmission;
    float transmission_color[3];
    float coat;
    float coat_roughness;
    float subsurface;
    float subsurface_color[3];
    float subsurface_radius[3];
    float subsurface_scale;
    char base_color_texture[1024];
    char metallic_roughness_texture[1024];
    char normal_texture[1024];
    char emission_texture[1024];
} LumbreBridgeMaterial;

// Authored UsdLux parameters forwarded verbatim. The bridge applies the SAME
// intensity/exposure, area-normalization, and shaping-cone math the CLI USD
// importer uses, so viewport lights match the CLI. Do NOT pre-bake intensity or
// shape geometry here: pass the light's authored values and its world transform.
//
// `kind` follows core's Usd_Light_Kind: none=0, sphere=1, rect=2, disk=3,
// cylinder=4, distant=5, dome=6. A dome with `texture_file` set becomes the
// scene environment (HDRI), not an analytic light.
//
// `world` is the light's 16-float world transform copied straight across:
// GfMatrix4d is row-major/row-vector and Odin's mat4 is column-major/
// column-vector, so the two conventions cancel and the values transfer 1:1
// (the same trick the CLI's usd_shim path uses). Store it row-major
// (`world[row*4 + col] = xform[row][col]`).
typedef struct LumbreBridgeLight {
    float world[16];
    int32_t kind;
    float intensity;
    float exposure;
    float color[3];
    int32_t normalize;
    float width;
    float height;
    float radius;
    float length;
    float angle;
    int32_t treat_as_point;
    int32_t has_shaping;
    float shaping_cone_angle;
    float shaping_cone_softness;
    char texture_file[1024];
} LumbreBridgeLight;

typedef void *LumbreBridgeContext;

// Lifecycle.
LumbreBridgeContext lumbre_bridge_create(void);
void lumbre_bridge_destroy(LumbreBridgeContext context);

// Scene resources. Positions/normals are triples of triples (one per triangle
// vertex), in world space.
int lumbre_bridge_replace_triangles(
    LumbreBridgeContext context,
    const LumbreBridgeTriangle *triangles,
    int32_t triangle_count);
int lumbre_bridge_replace_materials(
    LumbreBridgeContext context,
    const LumbreBridgeMaterial *materials,
    int32_t material_count);
int lumbre_bridge_set_material_texture(LumbreBridgeContext context, int32_t material_index,
    int32_t slot, const uint8_t *rgba, int32_t width, int32_t height, int srgb);
int lumbre_bridge_replace_lights(
    LumbreBridgeContext context,
    const LumbreBridgeLight *lights,
    int32_t light_count);

// Render settings.
int lumbre_bridge_set_resolution(LumbreBridgeContext context, int32_t width, int32_t height);
int lumbre_bridge_set_quality(LumbreBridgeContext context, int32_t samples_per_pixel, int32_t max_depth);
// Toggle the Metal GPU renderer (on by default). Pass 0 for the CPU fallback.
int lumbre_bridge_set_use_gpu(LumbreBridgeContext context, int use_gpu);
int lumbre_bridge_set_camera(
    LumbreBridgeContext context,
    const float origin[3],
    const float look_at[3],
    const float up[3],
    float vfov_degrees);

// Render synchronously (CPU) into the bridge-owned framebuffer.
int lumbre_bridge_render(LumbreBridgeContext context);

// Framebuffer read-back.
int lumbre_bridge_framebuffer_size(LumbreBridgeContext context, int32_t *out_width, int32_t *out_height);
// Debug: write the last completed frame to a PNG at `path`.
int lumbre_bridge_write_png(LumbreBridgeContext context, const char *path);
// Fill `dst` with width*height*4 floats (RGBA in [0,1], alpha = 1), flipped to
// Hydra's lower-left origin. Dimensions must match the last frame.
int lumbre_bridge_read_rgba_f32(
    LumbreBridgeContext context,
    float *dst,
    int32_t width,
    int32_t height);

#ifdef __cplusplus
}
#endif
