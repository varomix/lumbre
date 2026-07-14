// Functional smoke test for the Lumbre Houdini bridge dylib. It exercises the
// same C ABI the Hydra plugin uses: create, upload a triangle, set camera and
// resolution, render, and read the framebuffer back. It asserts the render
// produced a non-empty image so a broken bridge fails the build, not Houdini.
//
// Build/run: houdini/scripts/build_bridge.sh runs this after building the dylib.
#include "lumbre_bridge.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    if (lumbre_bridge_abi_version() != LUMBRE_HOUDINI_BRIDGE_ABI_VERSION) {
        fprintf(stderr, "smoke: ABI version mismatch\n");
        return 1;
    }

    LumbreBridgeContext ctx = lumbre_bridge_create();
    if (!ctx) {
        fprintf(stderr, "smoke: create failed\n");
        return 1;
    }

    const int W = 96, H = 64;
    if (!lumbre_bridge_set_resolution(ctx, W, H)) {
        fprintf(stderr, "smoke: set_resolution failed\n");
        return 1;
    }
    lumbre_bridge_set_quality(ctx, 8, 4);

    // A single large triangle facing the camera, centered at the origin.
    LumbreBridgeTriangle tri = {
        .positions = {-1.0f, -1.0f, 0.0f,  1.0f, -1.0f, 0.0f,  0.0f, 1.0f, 0.0f},
        .normals   = { 0.0f,  0.0f, 1.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, 1.0f},
    };
    if (!lumbre_bridge_replace_triangles(ctx, &tri, 1)) {
        fprintf(stderr, "smoke: replace_triangles failed\n");
        return 1;
    }
    // Verify the analytic-light ABI too. This point light illuminates the
    // front-facing test triangle independently of the environment.
    const LumbreBridgeLight light = {
        .kind = 5, // point
        .position = {0.0f, 1.5f, 2.0f},
        .intensity = {20.0f, 20.0f, 20.0f},
    };
    if (!lumbre_bridge_replace_lights(ctx, &light, 1)) {
        fprintf(stderr, "smoke: replace_lights failed\n");
        return 1;
    }

    const float origin[3]  = {0.0f, 0.0f, 4.0f};
    const float look_at[3] = {0.0f, 0.0f, 0.0f};
    const float up[3]      = {0.0f, 1.0f, 0.0f};
    if (!lumbre_bridge_set_camera(ctx, origin, look_at, up, 45.0f)) {
        fprintf(stderr, "smoke: set_camera failed\n");
        return 1;
    }

    if (!lumbre_bridge_render(ctx)) {
        fprintf(stderr, "smoke: render failed\n");
        return 1;
    }

    int fw = 0, fh = 0;
    if (!lumbre_bridge_framebuffer_size(ctx, &fw, &fh) || fw != W || fh != H) {
        fprintf(stderr, "smoke: framebuffer size mismatch (%d x %d)\n", fw, fh);
        return 1;
    }

    float *pixels = (float *)malloc((size_t)W * H * 4 * sizeof(float));
    if (!lumbre_bridge_read_rgba_f32(ctx, pixels, W, H)) {
        fprintf(stderr, "smoke: read_rgba_f32 failed\n");
        free(pixels);
        return 1;
    }

    // The triangle covers the frame center, so at least some pixels must carry
    // non-background radiance. We just check for any variation, i.e. the image
    // is not a single flat value.
    int nonzero = 0;
    float first = pixels[0];
    int varies = 0;
    for (int i = 0; i < W * H * 4; i += 4) {
        if (pixels[i] > 0.0f || pixels[i + 1] > 0.0f || pixels[i + 2] > 0.0f) {
            nonzero++;
        }
        if (pixels[i] != first) {
            varies = 1;
        }
    }
    free(pixels);
    lumbre_bridge_destroy(ctx);

    if (nonzero == 0 || !varies) {
        fprintf(stderr, "smoke: render produced an empty/flat image (nonzero=%d varies=%d)\n",
                nonzero, varies);
        return 1;
    }

    printf("smoke: OK (%d x %d, %d non-background pixels)\n", W, H, nonzero);
    return 0;
}
