# Realtime stabilization and throughput

This milestone builds on realtime Phases 01–03. It fixes dataset identity and
shader portability, avoids repeated GPU preparation, and pipelines headless
capture before the larger Phase 04 effects.

## Dataset identity

A labelled output such as `out/ds.png` owns `out/ds.classes.json`. This ordered
JSON array starts with the empty, unlabelled class at index 0. New classes append;
removing or reordering prims never reassigns existing IDs. The manifest is loaded
again on resume and replaced atomically only when it changes.

Declare a fixed vocabulary before distributed generation:

```python
lumbre.render(stage, "out/ds.png", frame=7, labels=True,
              classes=["chair", "table", "lamp"])
```

These names receive IDs 1, 2, 3. A conflicting manifest, duplicate name, or scene
class outside that vocabulary fails the render. A registry has one writer; parallel
workers use copies of a predeclared registry and disjoint frame outputs. Different
output stems are separate datasets unless given the same explicit vocabulary.

Camera selection accepts a full prim path, or a short name if it is unambiguous.
When short names collide, output filenames include a stable suffix derived from
the camera prim path. Selecting one camera still uses the caller's output stem.

COCO image IDs derive from the final image filename, including frame and camera.
Annotation IDs derive from image ID plus mask instance ID; `instance_id` records
the original value used in the EXR. IDs are deterministic 53-bit hashes, safe for
JSON numeric consumers. Dataset mergers should validate uniqueness, including
file identities, rather than concatenate blindly. Unlabelled geometry remains
in instance/depth/normal outputs but is omitted from COCO detections.

## Resource updates

Material refresh reads materials directly and never flattens scene geometry.
Scene upload flattens once. Content revisions distinguish geometry, texture,
light, semantic-table and environment changes, surviving a USD re-import whose
allocations have different addresses.

Camera and material-only changes reuse the vertex buffer. Identical texture
content reuses GPU textures, including across geometry changes. Unreferenced
textures are released after a successful scene update, keeping the cache bounded
by the current scene. Environment rotation and intensity update lookup parameters;
only changed source pixels trigger convolution. The BRDF LUT remains per renderer.

USD traversal/import is still synchronous per `lumbre.render`/`show`. Geometry
revision calculation scans source triangles; this does not implement USD notices,
GPU instancing, or transform buffers. Those are separate follow-up investments.

## Rendering and capture

- Material batches retain object bounds for camera and cascade culling. Adjacent
  surviving ranges merge back into a single draw where possible. Geometry order
  stays unchanged, preserving the existing depth-tie policy.
- The first distant light owns the cascade maps. Other distant lights do not
  accidentally sample its shadows.
- Deferred lighting writes linear RGBA16F. A separate display pass applies the
  existing clamp/sRGB encoding, so HDR is available for future effects without
  changing the display transform. The extra half-float rounding may change beauty
  by a quantization step; integer labels are independent.
- A two-slot capture queue submits beauty and labels before waiting, reuses
  transfer buffers, and overlaps the next camera's GPU work with current CPU
  encoding. Calls still return only after their files are complete. A scene stays
  alive until all its queued captures drain, so annotations cannot observe edits
  for a later frame. Partial failure drains the queue and reports failure.
- Label debug views use integer storage textures and a separate 1x1 fallback.
  Unused stencil state is explicit when depth targets cycle.

## Verification and measurement

```sh
odin test realtime
odin test output
odin test core
python3 scripts/check_shader_bindings.py
bash scripts/test_realtime_backend.sh metal
# Run on a machine where SDL has a working Vulkan driver:
bash scripts/test_realtime_backend.sh vulkan
```

The backend runner explicitly selects the driver, enables GPU validation, runs
GPU HDR/cache/debug-view checks, and exercises headless scripting. It fails if
the requested backend is unavailable rather than silently choosing another one.
The SPIR-V checker validates contiguous descriptors, resource-type ordering and
the lighting pipeline's exact binding layout. D3D12 still needs DXIL artifacts.

```sh
LUMBRE_RASTER_BENCH=1 ./lumbre --script python/examples/random_dataset.py -- /tmp/lumbre-bench/ds 20 7
LUMBRE_RASTER_PROFILE=1 ./lumbre -s scene.usda --raster -w 1920 -h 1080
```

The log reports post-import batch time, exported frames/second, cumulative
geometry uploads/environment builds, and allocated transfer-buffer bytes. These
are CPU wall time and allocation counters, not GPU timestamp measurements. GPU
passes have named debug groups for timing in a backend profiler.

`LUMBRE_RASTER_PROFILE=1`, or "GPU pass times" in the viewport, reports a time
per pass. SDL_GPU has no timestamp queries, so the profiler submits each pass
as its own command buffer and waits on a fence (`realtime/profile.odin`). The
split shows which pass dominates; it serialises the frame, so the total exceeds
an unprofiled frame, and each pass carries a fixed overhead of roughly 0.3 ms
on Apple silicon that the empty-submit calibration does not fully remove.
Never compare profiled totals against `LUMBRE_RASTER_BENCH` throughput.

Acceptance checks for changes to this path:

1. Camera/material-only edits do not increment geometry uploads; environment
   rotation/intensity do not increment environment builds.
2. Equal image data in new CPU allocations reuses GPU textures; replacing content
   releases unused cached textures instead of accumulating them.
3. Queued and individual camera label readbacks match byte-for-byte; repeated
   captures and fixed-seed datasets remain deterministic.
4. Class IDs survive object removal and process/session reload. Image and annotation
   IDs distinguish cameras and frames, and every COCO category reference resolves.
5. Benchmark cold startup separately from steady-state navigation and exported
   throughput at fixed resolution/scene complexity. Collect GPU pass timings and
   process peak memory with a platform profiler before claiming a frame-rate gain.

## Temporal antialiasing

The shaded view jitters the camera through 16 Halton (2, 3) points and blends
each frame into an HDR history (`realtime/shaders/taa_fs.slang`), before
exposure and the view transform. History is reprojected from depth and the two
cameras -- the scene is static between edits, and the G-buffer has no free
target for motion vectors -- then clipped to the frame's 3x3 neighbourhood in
YCoCg. TAA replaces FXAA while on.

- The viewport enables it by default and keeps drawing `TAA_SETTLE_FRAMES`
  after the last change, then idles as before.
- Headless runs keep FXAA by default, so dataset throughput and existing outputs
  are unchanged (verified byte-identical against the previous build). `--taa 1`
  renders each still as a converged sequence of `TAA_SETTLE_FRAMES` frames from
  a cleared history, so stills stay deterministic.
- Labels are drawn by their own unjittered pass and are byte-identical with TAA
  on or off (`test_gpu_taa_leaves_labels_alone`).
- Objects moved by an edit are not reprojected; the neighbourhood clip removes
  their ghosts at the cost of a few frames of aliasing on the moved edges.

## Spot light shadows

The first eight spot lights in scene order each get a 1024² perspective depth
layer (`realtime/shadows.odin`), fitted to the outer cone (capped at an 80°
half-angle) and reaching to the far side of the scene bounds. Deferred
lighting, glass and fog sample them; fog shows the cone and its shadows as
shafts. A layer depends on its light and the casters, not the camera, so it is
redrawn only when either changes. The bias is a normal offset scaled to the
texel size at the receiver's distance from the light, with 3x3 PCF.

Scenes without spot lights render byte-identically to before. Without bounce
light in realtime mode, spot shadows read darker than the path tracer's, which
fills them from the lit floor around them.

Point-light shadows (six faces each), baked GI and incremental USD import
remain on the broader roadmap.
