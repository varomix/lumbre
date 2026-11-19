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
lighting at 1080p. Fog still treats area lights as points for the light they
scatter; it is shadowed (below).

Cylinder lights agree across renderers. The path tracer, CPU and GPU, samples
only the half of a cylinder facing the point. It used to count the far side
through the near one, which doubled the light. The USD importer places a
`CylinderLight` by its centre, as UsdLux defines it, rather than its end.
Realtime, both path tracers and the closed form for a thin tube agree within a
few percent.

Incremental USD import remains on the broader roadmap. Baked GI and area light
shadows landed since; see below.

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

## Area light shadows

Quad, disc and cylinder lights take omni shadow slots like point and sphere
lights: the first four point or area lights in scene order get six 512² faces
each, rendered from the light's centre. Each light also carries a shadow
radius:

- sphere or disc: its own radius;
- quad: the radius of a disc with the same area;
- cylinder: the radius of a disc with the same side-on area, 2rh.

One number for every direction, so a long tube's shadow is as soft across it
as along it. A point light's radius is 0, and it keeps the old 3×3 hard
lookup.

Softening is PCSS on the omni map (`omni_visibility_soft` in
`shading.slang`):

- An 8-tap blocker search over the light's radius around the point averages
  the occluders' distance d_b.
- With the receiver at d_r, the penumbra radius is R (d_r − d_b) / d_b.
- A PCF kernel of that radius filters the hard shadow, capped at 128 texels
  (a quarter of a face): an 8-tap spiral first, and 32 more only where those
  eight disagree.

Four details:

- The search and, where the kernel fits on the point's own face, the filter
  run in the map's own texels. One projection per point, not one per tap.
- Each tap is compared with the receiving plane's depth at that tap, not the
  centre's, so a tilted receiver's own near side is not taken for a blocker.
  The plane's depth gradient across the map comes from projecting two points a
  few texels along the surface.
- Near a face seam, or on a plane edge-on to the light, the filter instead
  places taps in world space on the receiving surface and projects each to
  whichever face sees it.
- The tap spiral (Vogel) is turned by a hash of the shadow-map texel the point
  falls in. With one fixed spiral, every pixel's sparse taps crossed an
  occluder's edge in step, and penumbras came out mottled with a faint ring.
  Turned, it is fine grain that TAA averages. It is anchored in the world, so
  it does not crawl, and deterministic, so a still renders the same twice.
- The blocker search reads raw depths through a second, non-comparison binding
  of the omni map, sampler slot 15 in lighting and 13 in the forward pass. The
  storage buffers moved up one; the fog pass does not need it.

Fog's march already shadows lights through their omni slot, so area lights now
cast shafts. They are hard: fog is soft enough to bear it.

**Verified.** A GPU test with a closed form: a disc light of radius 0.5, 2 up,
over a half-plane occluder at height 1 whose edge lies under the light.
Measured as occluded over unoccluded floor, which cancels the shading:

- full shadow from x = −0.5 back;
- half at x = 0;
- full light from x = +0.5 on;
- partial in between.

The old hard lookup fails the penumbra checks, and so did a 32-texel cap,
which halved this penumbra. On `assets/render_tests/area_shadows.usda`
(panel over a box, disk over a floating ball, tube over a post), against
`--deterministic` 256 spp:

| Region | Path tracer | Before | Now |
|---|---|---|---|
| Ball's shadow core | 79 | 113 (no shadow) | 86 |
| Floor right of the box | 118 | 134 | 109 |
| Floor behind the post | 99 | 104 | 93 |

The ball's core is a little light: PCSS filters the shadow from the light's
centre, and a kernel as wide as a round occluder's shadow lifts its core. The
box's sides and the post read dark in realtime (20 against 69 on the box's
front), but that is bounce from the lit floor. With baked probes the box's
front is 68 and the post's side 90 against 97. Scenes without area lights
render byte-identical to `a234295`, labels everywhere unchanged.

**Cost.** About 0.3–0.4 ms per shadowed area light at 1920×1080 (Metal, the
minimum of five 16-frame averages):

| Scene | No soft shadows | First version | Now |
|---|---|---|---|
| `area_shadows.usda`, 3 lights | 1.38 ms | 3.53 ms | 2.27 ms |
| `area_lights.usda`, 4 lights | 1.69 ms | 5.04 ms | 3.29 ms |

Where it went, measured one change at a time:

- The blocker search was most of the cost, and that is the texture reads, not
  the projections. Moving the search to map space saved a little; halving it
  to 8 taps (the turned spiral estimates blocker distance as well with 8)
  saved 0.25–0.4 ms.
- The 8-tap first pass of the filter settles most points without the full
  spiral.
- Shading first and skipping the shadow where the light gives nothing
  helps where lights are far or behind.

A min-depth pyramid of the omni maps, to let lit points skip the search
outright, was built and measured and taken out. In these scenes an occluder
sits near every light, so the pyramid's coarse cells almost always held one
and the shortcut rarely fired. It cost more than it saved, and its rebuild
quadrupled the shadow pass. A first version of the filter's 8-tap pass used
every fourth tap of the 32-spiral. Four golden angles is about 190°, so those
eight fell in two narrow wedges; turned, they could miss a penumbra's lit
side, and the penumbra test caught it narrowed to 40%. Output stays within
0.06 of the first version's mean, with a different grain.

**Limits.** Shadows are cast from the light's centre: a surface the centre
cannot see but some of the light can (a box face just below a panel's plane)
is shadowed more than it should be, softened only by the kernel. Four omni
slots are shared by point and area lights in scene order.

### Shaped like the light, and grain TAA can average (after `c2132c7`)

Hands-on checks in the GUI found two faults in the post's shadow (the tube
station): it looked blocky up close, TAA made no difference, and it ended in a
hard line.

- **Kernel shape.** Every area light was filtered as a disc of the same area
  (`light_shadow_radius`), so the 1.6 m × 4 cm tube shadowed like a 0.2 m
  ball: a short, dark trapezoid, where the path tracer shows a wide, faint fan
  from the post's foot. The "line" was that trapezoid's far end, not a seam.
  Now `light_shadow_axes` gives each light's shape as the half-axes of an
  ellipse: a quad's half edges, a disc's radius in its plane, a tube's half
  length and its radius, a sphere's radius across the view. The axes are slid
  along the light's direction onto the receiving plane, then mapped to texels
  through the Jacobian the slope solve already measures. The blocker search
  covers that ellipse, and the filter is the ellipse scaled by
  (receiver − blocker) / blocker. Each axis is capped at `PCSS_MAX_TEXELS`.
- **Grain.** `pcss_spin` hashed the shadow-map texel and never changed, so
  TAA had nothing to average. Close up, where one texel covers several pixels,
  the grain read as blocks. It is now interleaved gradient noise per pixel,
  moved each TAA frame (`forward_params.z`, 0 without TAA). Without TAA that
  is fine dither; with TAA, smooth. Stills stay deterministic.
- **Seams.** Search taps that fall off the point's face, or every tap when the
  slope solve fails, are now placed on the surface in world space and
  projected to the face that sees them. They were skipped before, losing
  blockers held by the neighbouring face.

Close up on the post (`assets/render_tests/area_shadow_post.usda`, TAA),
against `--deterministic` 256 spp:

| Region | Path tracer | Before | Now |
|---|---|---|---|
| Floor just behind the post | 100 | 79 | 100 |
| Far end of the old trapezoid | 90 | 84 | 89 |
| Fan, left of the post | 105 | 112 | 109 |
| Right at the foot | 98 | 74 | 113 |

The foot is the one miss. An occluder touching the receiver is where PCSS's
average blocker depth is least true. The stations view is otherwise as before
(mean |diff| from the path tracer 1.44 → 1.42), and scenes without area
lights render byte-identical, with and without TAA.

Cost: per-pixel rotation means neighbouring pixels no longer share texel reads.
At 1080p, `area_shadows.usda` lighting goes 2.38 → 2.74 ms and
`area_lights.usda` 3.20 → 3.90 ms.

## Environment reflections

The split-sum maps now describe the same GGX as the analytic BRDF and the
path tracer (`core/principled.odin`): alpha = roughness. Before, the
prefiltered environment and the BRDF LUT squared roughness once more, so
environment reflections were far sharper than a light's highlight on the
same surface. The fix has three parts:

- **One alpha.** `importance_sample_ggx` (common.slang) takes alpha =
  roughness. The LUT's geometry term is now the exact uncorrelated Smith G1
  that `smith_G1` and `ggx_G1` use, not the Schlick-k fit, which put its scale
  up to 0.15 high.
- **Filtered importance sampling** in `env_specular_fs`. Each tap reads the
  source mip whose texel covers the tap's share of the lobe,
  ½ log2(Ω_sample / Ω_texel) + 1, with Ω_sample = 1 / (N · D/4) and an
  equirect texel's solid angle shrinking with sin θ. It is never finer than
  the mip matching the 256 × 128 target, which the mirror level also reads.
  Before, every level read one fixed coarse mip (32 × 16 for a 2K HDRI): wide
  lobes came out blotchy, with a pinch at the poles.
- **Exact diffuse.** `env_irradiance_fs` sums a 64 × 32 grid of directions
  (each weighted by cosine and solid angle) instead of 256 cosine-sampled
  ones. The samples aliased against a bright sun and drew stripes across
  every surface facing it.

Measured on `assets/render_tests/glossy_env.usda` (960 × 540, 80 px box per
sphere, 8-bit mean |diff| against 256 spp with `--roughness-cutoff 2`, since
the default cutoff renders roughness ≥ 0.95 as diffuse):

| Roughness | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
|---|---|---|---|---|---|---|
| Metal, before | 2.7 | 11.1 | 10.6 | 14.9 | 11.0 | 8.4 |
| Metal, now | 2.2 | 8.2 | 5.5 | 7.3 | 8.3 | 8.7 |
| Red plastic, before | 3.5 | 2.7 | 3.1 | 2.9 | 1.8 | 1.6 |
| Red plastic, now | 5.4 | 4.5 | 3.6 | 2.7 | 2.1 | 1.8 |

At 0.2 and 0.4 the metal's mean now matches (91 vs 90, 84 vs 82). What is
left there is SSR, which mirrors the neighbours at any roughness. Rough metal
reads 6–9 levels bright, the split sum's view-equals-normal assumption.
Dropping the prefilter's n·l weight made it worse (+11). The red spheres read
brighter than the path tracer because the metal spheres above them shadow the
HDRI's sun there, and environment light has no visibility; a lone red sphere
matches within 1–4 levels.

Two GPU tests in `realtime/environment_test.odin` hold this. Both check
against CPU integrals of `ggx_D` and `ggx_G1`: the LUT at four (n·v,
roughness) points, and the prefiltered map of a half-bright sky just above
and below the horizon at roughness 0.4 and 0.6. Mutation-tested: the old
alpha fails both, and the Schlick-k term fails the LUT.

Every frame lit by an environment changes. Labels do not.

## Shadow slot ranking

The four omni and eight spot shadow slots used to go to the first lights in
scene order. `shadow_slots_assign` (shadows.odin) now re-ranks them every
frame by how much each light matters to the view:

- Score: power (as the cluster pass measures it, `max(rgb)`, times the area
  for area shapes) over the squared distance from the eye, floored at the
  light's own extent.
- A light whose reach sphere (`cluster_light_reach`) misses the view frustum
  ranks below every light whose reach does not, and among those by power
  alone, so spare slots still go somewhere useful.
- Spots and omnis rank separately. Ties go to the earlier light.
- The chosen lights take slots in scene order, so a slot moves, and its map
  is redrawn, only when the chosen set changes. Camera moves that keep the
  set redraw nothing, as before.
- No history: a still renders the same alone or in a sequence. A light can
  gain or lose its shadow as the camera crosses a ranking boundary; there is
  no hysteresis, since hysteresis would make a frame depend on earlier ones.

`lights_convert` still fills in the scene-order default, for callers with no
camera. The light buffer always matches `light_scratch` at the start of a
frame, so `renderer_render` re-uploads it only when a slot moved.

The spot, point, area-shadow, area-light and baked-GI room scenes all fit the
limits and render byte-identical to `8459a20`. On
`assets/render_tests/shadow_ranking.usda` (eight bulbs beside eight posts,
listed farthest first) the shadows move from the far end of the row, where
they were too small to read, to the four posts nearest the camera.
`test_shadow_slots_follow_importance` covers the ranking; it fails with
either the scene-order or the visibility-only comparator.

## Area light Fresnel

The two path-tracer mismatches the handoff listed, measured again (800 ×
450, against `--deterministic` 512 spp on the area-light scene and 256 on
the Cornell box):

**Area-light scene floor.** Splitting the scene by light found it. Sky alone
matches. A tube over a matte floor matches an analytic integral within 1–3%
in both renderers (the CPU path tracer reads 5% under, the known gap). A
point light over the glossy floor matches the path tracer exactly. A tiny
sphere light in the same place does not: the floor near the camera read 37
against 47.

The LTC specular took Fresnel from the table, averaged over the whole GGX
lobe. That is exact only for a light covering the lobe. In the lobe's grazing
tail v·h is small and Schlick's term several times the average, and on a dark
dielectric that is nearly all of the reflection. `area_light` now takes
Fresnel at the light's own direction: the direction of the specular
polygon's vector form factor, mapped back out of the cosine's space by M. It
blends toward the table average by the share of the lobe the light covers
(the normalized LTC integral), so a light covering the lobe is unchanged.

| Region | Before | Now | Path tracer |
|---|---|---|---|
| Orb test, near floor | 37 | 49 | 47 |
| Tube test, near floor | 63 | 80 | 78 |
| Area scene, orb reflection | 151 | 145 | 140 |
| Area scene, near floor | 81 | 97 | 93 |
| Area scene, tube reflection | 143 | 138 | 139 |

`test_gpu_small_sphere_light_matches_a_point_light_at_grazing` holds it: a
tiny sphere gives 91% of the equal point light at an 82° light and 56° view,
where it gave 36%. The remaining 9% is the fitted lobe's shape, which falls
short of GGX toward the horizon (0.67–0.86 of it for light past 70° at
roughness 0.25). Refitting with the lobe's tilt as a fourth parameter cut the
fit's mean L1 error from 0.17 to 0.12, but rendered worse: it overshoots the
backscatter side, which lit the far floor at 20 against 8. An L1 objective
was worse still at grazing. The fit stays as it was.

**Cornell box short block.** Its front reads 80 against 77 overall, within
1–2 levels except the bottom quarter, where it meets the floor: 77 against
68. The path tracer darkens about 30 px up from the contact; SSAO covers the
last 10, and the probes are a cell (an eighth of the box) apart. A larger AO
radius does not help overall: mean |diff| on the Cornell box and the room
moves 2.0 → 1.9 and 2.2 → 2.5 at three times the radius. (This was left for
the GTAO step, which found it is not occlusion at all; see "GTAO".) The "78 vs 66" figure in the handoff no longer reproduces at
any commit since probe reflections; it was probably that bottom strip.

## Glossy reflections

SSR traced one mirror ray per pixel and faded out between roughness 0.2 and
0.45. Glossy surfaces showed either a sharp mirror image or none, and in the
area-light scene the 0.25 floor mirrored the box and ball crisply.

**Now** there are two passes:

- **`ssr_fs` (trace)**, one ray per pixel of a half-resolution target, drawn
  from GGX's distribution of
  visible normals (Heitz 2017), so the rays spread as the lobe does:
  stretched along the view at grazing angles, which is what streaks a
  reflection down a floor. The two sample numbers are interleaved gradient
  noise, which spreads well over a 3 × 3 neighbourhood, moved per TAA frame.
  A hit is read from a blur pyramid, `refraction`'s mips built right after
  lighting, at the level matching the ray's share of the lobe at that
  distance, capped at level 3. Past that, a level's square takes in the
  floor under the object and the sky beside it and turns the reflection a
  flat grey. Output: colour × trust, trust. Below roughness 0.04 it traces
  nothing: mirrors need every pixel, so the resolve traces their exact ray at
  full resolution. The march lives in `ssr_march.slang`, shared by both.
- **`ssr_resolve_fs`** averages each glossy pixel's ray with its
  neighbours', weighted by how alike the surfaces are (off-plane distance,
  normal, roughness). That is 3 × 3 with TAA, and for a still 5 × 5 with the
  outer ring thinned to a checkerboard. The spacing grows with roughness up
  to 3 px. The hits replace their share of the environment's reflection;
  a pixel whose rays all missed keeps exactly what lighting gave it.
- The fade moves to 0.5 → 0.75.

**Why not several rays per pixel:** eight cost 23 ms at 1080p against the
mirror ray's 2.35. Four, for stills, cost 19.6 and still showed grain. One
per full-resolution pixel cost 7.6 ms in the baked-GI room, whose surfaces
are all roughness 0.5 and so skipped SSR before (0.45 ms); half resolution
quarters the trace. A
fade by blur level, which handed distant reflections back to the
environment, killed the reflection at the blocks' feet too.

Measured on `assets/render_tests/glossy_floor_{015,03,05}.usda` (800 × 450,
whole-image 8-bit mean |diff| against 256 spp, `--roughness-cutoff 2`):

| Roughness | Before | Still | TAA |
|---|---|---|---|
| 0.15 | 6.4 | 2.7 | 2.3 |
| 0.3 | 5.0 | 3.2 | 2.5 |
| 0.5 | 4.0 | 3.0 | 2.6 |

Other scenes against the path tracer: area lights 2.8 → 1.8, and Cornell
box, room and sphere sweep unchanged (2.0, 2.2, 4.4 → 4.3). Cost at 1080p
(SSR pass, with TAA / still):

| Scene | Before | Now |
|---|---|---|
| 0.15 floor | 2.35 | 1.8 / 2.6 |
| Baked-GI room (all roughness 0.5) | 0.45 | 3.9 / 5.1 | A still
keeps a fine grain; TAA removes it. `test_gpu_glossy_reflections_spread`
checks that a 0.3 floor picks up a strip's light beside its mirror image, that
0.6 still reflects, and that a still renders the same twice. It fails with
the mirror ray for every roughness, or with the old fade.

**Found on the way: the path tracer's GGX sampler was wrong** (`e4266ba`).
Both path tracers' `sample_ggx_vndf` built its basis around the y axis and
scaled the far half of the visible disc by sqrt(1 − s1²) instead of v.z, so
its samples did not follow the pdf the estimator divides by. At alpha 0.25
and 85° the specular albedo estimate was 0.59 against 0.86: glossy
reflections seen low came out about 30% dark. On the sphere sweep it moved
the reference by at most 3 levels. Render-page references made before it are
slightly off on glossy surfaces seen at a grazing angle.

While building the floor scene, a block's reflection appeared to vanish on a
0.05 strip in the path tracer. That was geometry, not a bug: a floor point
reflecting a block lies between the camera's x and the block's, so the
image falls on the neighbouring strip, and the 0.2 strip blurs it as it
should.

## GTAO

`ao_fs` is now ground-truth ambient occlusion (Jimenez et al. 2016) in place
of the SSAO it replaces:

- **Slices.** Two slices through each pixel (planes containing the view
  direction), at angles rotated per pixel by the 4 × 4 ordered pattern that
  `ao_blur_fs` already cancels: 32 directions per 4 × 4 block.
- **Horizons.** On each slice it marches six steps each way, quadratically
  spaced out to the radius, keeping the highest horizon. An occluder counts
  fully within 0.4 of the radius and fades to nothing at it.
- **Integral.** The cosine-weighted visible arc between the two horizons is
  integrated in closed form against the normal projected into the slice.
- **Normalization.** The result is divided by what the same two slices
  give with nothing in the way. Two angles per pixel only average to exactly
  1 on an open surface once the blur has mixed all sixteen rotations in. On
  a floor seen low it cannot, because the rows above and below are too far
  off in depth to blend.
- **No tuned exponent.** SSAO raised its sample count to the power 1.5;
  GTAO's answer is visibility itself (`AO_INTENSITY` = 1).

A new debug view shows it: `--raster-view 10`, "AO" in the viewport.

**Measured** (800 × 450, mean |diff| against 256 spp): Cornell box 2.0 →
1.9, baked-GI room 2.2 → 2.3, area lights 1.8 → 1.9, sphere sweep 4.3 → 4.2.
A larger radius (2.5× or 5×) does not help: the room gets worse, 2.4 and
2.6, since its probes already carry the occlusion at their cell's scale.
Cost at 1080p: 1.72 → 1.28 ms in the room, 0.82 → 0.95 in the Cornell box.

So GTAO is not measurably closer to the path tracer than the SSAO was on
these scenes. What it brings is a result with a meaning, with no exponent to
tune, that can be tested: `test_gpu_gtao_sees_half_the_sky_at_a_wall` holds
the bounds of a tall wall's foot. A half-space hides at most half the
cosine-weighted sky, visibility rises with distance from the wall, and the
open floor is 1. It fails with the horizons swapped or the normal ignored.

**The Cornell block's foot is not AO's.** Even at five times the radius GTAO
darkens only the last 10 px of the short block's front, where the path
tracer's darkening starts about 25 px up. That gradient is indirect light
varying finer than the probe grid, not contact occlusion. "Area light
Fresnel" above said the GTAO step would take it; it cannot. A finer grid
(`--grid 16`) or a screen-space GI term would.

Left as found: flat floors read about 0.96 in the AO view, not 1. It does
not change with the radius, so it is not the horizon search; depth or
normal precision at a low angle is the likely cause.

## Fog: area lights, colour, multiple scattering

Three changes to `fog_fs`:

- **Area lights by solid angle.** An area light lights a step in the air
  with its radiance times the solid angle it fills from there
  (`area_solid_angle` in shading.slang): exact for a sphere, exact for a quad
  and for a tube's facing rectangle (two triangles, Van Oosterom–Strackee),
  and for a disc a spherical cap with the same small-angle solid angle, which
  is exact on its axis. The point approximation (area · cos / d²) gave three
  times the right in-scattering for rays through a 1 m panel, and without
  bound nearer in. The 12-gon's exact solid angle cost the pass twice its
  time, so discs take the cap.
- **Dimming and multiple scattering**, both local estimates with H the
  distance light travels to leave the fog (the layer's falloff height, capped
  at the scene's size) and τ = σH:
  - The sky and the sun arrive at each step dimmed by e^−τ, and a local
    light by e^−σd for its distance d, up to H.
  - Every order after the first is isotropic, with albedo a and a chance
    f = 1 − e^−τ of scattering again. Summed: a f / (1 − a f) of the
    dimmed light.
  - Under a uniform sky with a = 1 that is exactly the sky again, as it must
    be. Before, two omissions (no dimming and no multiple scattering) happened
    to cancel there. `--fog-multiple 0` leaves single scattering with the
    dimming, which is darker.
- **Colour.** `fog_colour` is the scattering albedo (`--fog-colour r,g,b`,
  and a colour picker in the viewport). What the fog does not scatter it
  absorbs.

Tests: `test_gpu_fog_lit_by_an_area_light` checks a uniform fog beside and
through a quad light against the integral along the ray, with the solid
angle from a 48 × 48 grid over the panel, within 5%. The point approximation
fails it at 3×. `test_gpu_fog_matches_beer_lambert` now also checks, in
uniform fog, the exact uniform-sky answer with multiple scattering, the
dimmed single scattering, and a (1, 0.5, 0) colour channel by channel. It
fails without the series's 1/(1 − af) or without the dimming. One open
point: in that test's height-varying fog, single scattering reads about 0.03
brighter than the estimate at its pixel. Not traced.

Cost at 1080p, TAA: the area-light scene in dense fog goes 11.7 → 19 ms; the
point-shadow room, with no area lights, 3.7 → about 5. Taking the point form
wherever it was within 1% saved nothing, so the cost is the larger shader,
not the arithmetic.

## Glass seen through glass

Glass refracted one copy of the opaque image, taken before any glass was
drawn. A pane behind a pane was invisible through it. A glass object's
front and back faces, drawn in one pass without depth writes, came out in
whatever order its triangles did.

`draw_forward` now lays glass down farthest first. Each object draws its back
faces over a fresh copy of what is drawn so far, then its front faces over
another (`forward_layer`; `forward_params.w` tells `forward_fs` which faces,
by the geometric normal). Back faces only pass light through, tinted. What
they reflect goes back into the glass, which screen space cannot follow.
Shaded as outside surfaces, they added a second sky reflection and a diffuse
glow, and total internal reflection at a ball's rim. Past
`GLASS_LAYERED_MAX` (16) objects, all back faces go down over one copy and
all front faces over a second.

On `assets/render_tests/glass_layers.usda` the green pane now shows through
the clear one, as in the path tracer. Mean |diff| against 512 spp is 6.0 →
5.9; the ball stays milkier than the reference, because screen-space
refraction samples the bright floor just behind it where a real ball inverts
the view. The glass pass costs 0.50 → 0.97 ms at 1080p with three glass
objects. Scenes without glass render byte-identical.
`test_gpu_glass_is_seen_through_glass` needs a white wall to come through a
clear pane and a red pane red; with one copy it came through white.
