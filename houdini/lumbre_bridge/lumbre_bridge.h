#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increment only for intentionally incompatible ABI changes.
#define LUMBRE_HOUDINI_BRIDGE_ABI_VERSION 2u

uint32_t lumbre_bridge_abi_version(void);

typedef struct LumbreBridgeTriangle {
    float positions[9];
    float normals[9];
} LumbreBridgeTriangle;

// Analytic lights expressed in world space. `kind` follows the Lumbre core
// Light_Kind enum: quad=0, sphere=1, disc=3, cylinder=4, point=5,
// spot=6, distant=7.  `intensity` is already color * USD intensity/exposure.
typedef struct LumbreBridgeLight {
    int32_t kind;
    float position[3];
    float direction[3];
    float u[3];
    float v[3];
    float intensity[3];
    float radius;
    float height;
    float cos_inner;
    float cos_outer;
    float angular_radius;
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
