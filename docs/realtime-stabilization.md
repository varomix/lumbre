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

## Render tests

`docs/render-tests/index.html` shows each milestone's visual checks: a scene
rendered with the feature off and on, with the build before the change, or
with the path tracer as reference, in an A/B wipe viewer with pixel zoom and
the GPU pass times of each render. `scripts/render_tests.py` renders them and
writes `data.js`; add a step to its `STEPS` list with each change and commit
the images with it. "Before" variants name a commit, built once into
`.render_tests/`.

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

## Point and sphere light shadows

The first four point or sphere lights in scene order each get six 512²
perspective faces, stored as consecutive layers of a 2D depth array rather
than a cube map. The shader picks the face by the dominant axis of the
direction from the light, so the faces reuse the spot path's pipeline, sampler
type, caching and normal-offset bias. A face spans slightly more than 90° so
PCF near its edge stays on it. A sphere light's near plane is its radius, so
it never shadows itself. Shadows stay hard: the light's radius does not soften
them as it does in the path tracer.

Scene order rather than importance keeps a light's slot stable as the camera
moves, so navigating never reassigns or redraws a map. At 1080p, drawing six
faces costs about 1.2 ms (once, then cached) and the lookups add about 0.3 ms
to lighting. Scenes without point or sphere lights render byte-identically.

## Area lights

Quad, disc, sphere and cylinder lights shade surfaces with linearly transformed
cosines (`realtime/ltc`, `realtime/area_lights.odin`). Diffuse is the light
polygon's exact form factor; specular comes from a 64×64 table fitted by
`tools/ltc_fit` to Lumbre's own GGX, not the paper's published table, which
assumes a different Smith term and roughness mapping. A disc is a 12-gon, a
sphere the 12-gon facing the point that subtends the same solid angle, and a
cylinder the rectangle along its axis facing the point. Polygons are not
clipped. The unclipped vector form factor is corrected for the horizon as a
spherical cap would be, from a second table in the same texture. Lights are
two-sided, as in the path tracer.

Measured: the fit is within 5% of the lobe's total against the exact BRDF
(worst case smooth and grazing, mostly under 1%). The horizon correction is
within 0.002 of exact clipping. Rect, disk and sphere lights alone over a matte
floor are within 1–2% of the path tracer. Four area lights add about 1 ms to
lighting at 1080p. Fog still treats area lights as points, and they cast no
shadows, except sphere lights through their omni slot.

Cylinder lights agree across renderers. The path tracer, CPU and GPU, samples
only the half of a cylinder facing the point. It used to count the far side
through the near one, which doubled the light. The USD importer places a
`CylinderLight` by its centre, as UsdLux defines it, rather than its end.
Realtime, both path tracers and the closed form for a thin tube agree within a
few percent.

Shadows for quad, disc and cylinder lights and incremental USD import remain
on the broader roadmap. Baked GI landed since; see below.

## Baked GI

The diffuse ambient term can come from irradiance probes baked by the path
tracer instead of from the environment alone (Phase 04 steps 12–14). The
environment knows nothing of the scene: every surface facing up sees the whole
sky, ceiling or not, and nothing receives light bounced off anything else,
including emissive meshes such as the Cornell box's ceiling panel. Probes carry
both, at the grid's resolution.

**Bake.** `lumbre --bake-probes scene.usda [--grid 8] [--probe-rays 2048]`, or
`lumbre.bake_probes(stage)` in a script. Probes sit at the cell centres of an
axis-aligned grid over the scene's bounds, `--grid` along the longest axis and
the other axes in proportion. A flat axis counts as at least a tenth of the
largest, as for fog. Each probe traces a fixed Fibonacci set of directions
with the CPU path tracer's `ray_color`, using `--depth`, `--max-radiance` and the
bias switches. Each probe draws from its own seeded generator and threads only
pick which probe runs next, so two bakes are byte-identical. The output is the
sidecar `scene.probes`, or `-o` when that ends in `.probes`. It is gitignored
as a derived file.

Per probe, two 8×8 octahedral maps and one 16×16, each with a one-texel
border:

- **Irradiance.** For each texel direction n, the cosine-convolved incoming
  radiance divided by π. That is what the environment's irradiance map holds,
  so it substitutes directly.
- **Depth moments.** The mean and mean square of the first-hit distance under a
  cos⁵⁰ lobe, normalized by 1.5 cell diagonals.
- **Radiance** (16×16). The incoming radiance itself: what a reflection sees
  along a direction. Each texel averages the rays under a cos^k lobe, with k
  chosen so that about 128 rays carry it: cos¹⁵ at 4096 rays, up to cos⁶⁴
  (about one texel wide) from 33k rays. A ray's radiance is one path. In a
  scene lit by an emissive mesh it swings by orders of magnitude depending on
  whether a bounce found the light. A fixed cos⁶⁴ (about 32 rays a texel at
  4096) made the Cornell box's mirror wall reflect blotches; fewer rays now
  give a blurrier map instead of a noisier one.

Before the trace, each probe is **relocated** within its cell, as DDGI does
at runtime:

- A probe whose relocation rays (256, first hits only) mostly meet back faces
  is inside a solid. It moves out through the nearest back face, plus
  clearance.
- A probe nearer a surface than 0.3 of the smallest cell side moves straight
  away from it.
- It gets four rounds and may move at most 0.45 of a cell per axis.
- One still inside is switched off.

In the Cornell box, 40 grid points land inside the two blocks. Their dark
view leaked onto the blocks' outside faces as smudges, most visibly on the
short block's front. 30 of them now move out and 10 are off. A probe column
0.007 from the tall block's back moves to the clearance. "Mostly back faces"
trusts winding: a closed room with every normal flipped would switch off all
its probes, and the bake says so. Offsets and the off flag are in the file
(format version 4; older sidecars are refused with a message to rebake) and
in a per-probe buffer the lookups read.

Probes see what leaves surfaces and the environment, never an analytic light's
own emission, because path-tracer rays do not intersect lights. The rasterizer
shades those lights directly, so nothing is counted twice.

The plan said L2 spherical harmonics. They ring: the fit to a small, bright
source goes negative about 120° from it, and the clamp then zeroed the probes
beside the Cornell box's panel. That left dark, green-tinted blots on the
ceiling around the light (41 against the path tracer's 67 in red). The
octahedral map is convolved from the rays themselves, cannot go negative, and
shares the depth map's layout and lookup.

**Use.** The lighting pass binds two storage buffers after its existing three.
Irradiance is one float4 per texel and depth moments one float2, so no sampler
slot moved. Filtering is bilinear, done by hand. The lookup follows DDGI:

- The point is offset by a quarter of the smallest cell, 0.2 along the normal
  and 0.8 toward the viewer.
- Weights come from the eight surrounding probes, trilinear.
- Each weight is scaled by `((dot(dir, n) + 1) / 2)² + 0.2` for facing, then
  by a cubed Chebyshev bound against the depth moments. Weights under 0.2 are
  crushed.

The result replaces `irradiance_map × env.intensity`. It is absolute, so it
holds with no environment at all. AO still multiplies the ambient term.

**Reflections.** With the same eight weights, the lookup also returns the
ambient reflection. It replaces the prefiltered environment, which saw only the
sky. Each probe is read along the reflection vector r with one parallax step:
its depth along r gives how far the reflected surface is, and the probe is read
toward `p + r × that distance` instead of along r from where it stands. The
result blends from the sharp radiance map toward the irradiance map as
roughness rises, standing in for the environment's mip chain. It is the
fallback wherever screen-space reflections find nothing on screen: off-frame,
or the sides of things the camera cannot see. SSR binds the same three
buffers, so it subtracts exactly what lighting added before adding what it
hit. On the Cornell box's mirror wall, the patch that SSR could not fill
showed the sky, a pink (96, 64, 69) against the path tracer's (51, 22, 16). It
now shows a blurred dark red (58, 22, 18). The glass pass reflects the probes
the same way.

A GPU test holds it together: a uniform probe field must light a mirror floor
and a matte wall exactly as a uniform sky of the same radiance does, SSR hits
and misses included. Making SSR subtract the sky instead of the probes, or
lighting ignore the probe reflection, each fails it.

The forward (glass) pass binds the same two buffers after its own three and
runs the same lookup. A clear pane barely changes, because its diffuse lobe is
scaled by `1 - spec_trans`. Frosted, tinted or partly transmissive surfaces
take the probes like opaque ones.

Fog is different: it needs ambient light at each of 32 march steps per pixel,
and the surface lookup there would be thousands of reads a pixel. So fog binds
a third buffer with one float4 per probe: the mean of its interior irradiance
texels, which is the mean radiance arriving from all directions (exact for a
uniform field). It reads that trilinearly from the eight surrounding probes,
with no visibility test. That leaks through walls thinner than a cell, which
fog's softness hides. It replaces the sky's "mean of up and down" ambient.
In the example room, fog under the roof stops glowing as if under open sky.
At 1080p the fog pass costs about 4.4 ms with probes against 3.7 ms without
(Metal, 16-frame average).

**Revision.** The sidecar records a hash of everything the bake saw:
triangles, placement, material parameters and textures, lights, and the
environment's pixels, rotation and intensity. The camera is not included. The
renderer compares it on every new scene key. In the GUI it also compares after
in-place material and light edits, which do not change the key. The
triangles-and-texels half is cached per key, so a slider drag rehashes only
parameters. On a mismatch the grid stays loaded but unused, and the renderer
prints that ambient falls back to the environment. `--probes 0`,
`probes=False` and the viewport's "Baked GI" box turn it off. The box is
greyed out, with a tooltip saying why, when no matching grid is loaded.

**Measured** on the render page's `baked-gi` step (800×450, against
`--deterministic` 256 spp, 8-bit mean |diff|):

| Scene | Environment ambient | Baked probes | Bake |
|---|---|---|---|
| Cornell box (8×8×8 probes, 4096 rays) | 8.56 (signed +5.60) | 1.95 (+0.81) | 3.5 s |
| Room with instanced crates (8×5×8, 4096 rays) | 30.80 (+30.62) | 2.09 (−0.01) | 8.4 s |

Before probe reflections the probes gave 2.85 and 4.91, and before
relocation and the adaptive radiance lobe 2.02 and 2.16. The room gains most:
every surface has a dielectric sheen, which used to reflect the sky.

In the room, the back wall goes from 108 to 71 against the path tracer's 67,
the ceiling from 111 to 67 against 62, and the floor between the crates from 88
to 56 against 44 (green channel); those figures are from before probe
reflections. At 1920×1080 on the Cornell box the lighting pass costs about
1.2 ms with probes (diffuse and reflections) against 0.66 ms without, and SSR
about 1.0 ms either way (Metal, 16-frame average, fence-timed). With no sidecar, or `--probes 0`, beauty, label EXR, COCO and
classes are byte-identical to the build before. With probes, labels are
unchanged.

**Limits.**

- The resolution is the grid's: a cell is an eighth of the scene by default,
  so contact shading at a crate's foot still comes from AO. Indirect shadow
  between close objects is exaggerated where a probe stands between them. On
  the Cornell box's green wall beside the short block, a soft dip of about
  8 levels in the path tracer is a sharp one of 23 at `--grid 8` and 13 at
  `--grid 16`.
- Light leaks through walls thinner than the depth maps resolve.
- The CPU path tracer has no next-event estimation for emissive meshes, so
  scenes lit that way need more rays for a clean bake.
- The bake is static. A moved light or crate makes the grid stale; nothing is
  updated incrementally.

**Found on the way.** The CPU path tracer rendered roughness-0 Principled
metals black, because GGX sampling returned a zero pdf. The Cornell box's red
wall is one, through its MTL's Ks. The GPU path tracer has always floored
roughness at 0.015; the CPU now does the same (`66cf83f`).
