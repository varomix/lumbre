# Lumbre

A **biased GPU path tracer** for Apple Silicon, written in Odin + Metal.

Inspired by Redshift — trades physical accuracy for speed using
irradiance caching, photon mapping, and aggressive bias controls.
Targets 10-100× faster convergence than unbiased renderers.

## Status

**Stage 4A + 4B complete** — Pure-Odin EXR writer (FP16, multi-layer)
and `--aovs` for multi-AOV EXR output. Builds on the Stage 3
Principled BSDF and Stage 2 biased GI.

## Quick start

```bash
scripts/setup_dev.sh   # once: checks for and fetches the toolchain
odin build .
./lumbre --scene assets/cornell_box.obj --width 1920 --height 1080 --spp 100 --output frame.exr
```

Or with AOVs:

```bash
./lumbre --scene assets/cornell_box.obj --width 1920 --height 1080 --spp 100 --aovs --output frame.exr
```

Or with production denoising:

```bash
./lumbre --scene assets/cornell_box.obj --width 1920 --height 1080 --spp 32 --denoise --output frame.exr
```

Or use the test script:

```bash
scripts/render_cornell.sh --quick
```

## Usage

```
lumbre [options]

  --scene, -s <file.obj>     Load OBJ scene
  --width, -w <int>          Image width (default 1024)
  --height, -h <int>         Image height (default 576)
  --spp <int>                Samples per pixel (default 50)
  --depth <int>              Max bounces (default 20)
  --max-radiance <float>     Firefly clamp (default 1000)
  --roughness-cutoff <float> Bias: treat rough Principled as diffuse (default 0.95)
  --glossy-bias <float>      Bias: damp Principled roughness toward mirror (default 0)
  --output, -o <file.png|exr> Output file (default render.png)
  --hdri <file.hdr>          Equirectangular HDRI environment (dome light)
  --hdri-rotation <deg>      Rotate the HDRI about the Y axis (default 0)
  --hdri-intensity <float>   Scale the HDRI radiance (default 1)
  --sun-dir <x,y,z>          Add a directional/sun light (direction it travels)
  --sun-intensity <f|r,g,b>  Sun radiance, scalar or RGB (default 1)
  --sun-angle <deg>          Sun angular diameter for soft shadows (default 0.53)
  --aovs                     Write AOV layers to .exr output
  --cpu                      Force CPU renderer
  --gpu                      Force GPU renderer
  --quality <preset>         Quality preset: draft, preview, final
  --gi-dist <float>          Cache lookup distance (default auto)
  --photon-radius <float>    Photon search radius (default auto)
  --raster                   Render with the realtime rasterizer (see --help for its switches)
  --frame <t>                USD time code to render (default: the stage's start time)
  --bake-probes <scene>      Bake irradiance probes for the rasterizer into <scene>.probes
  --grid <int>               With --bake-probes: probes along the longest axis (default 8)
  --probe-rays <int>         With --bake-probes: rays per probe (default 2048)
  --probes <0|1>             With --raster: use the scene's baked probes (default 1)
  --debug <mode>             Debug: 1=albedo, 2=normal, 3=depth,
                             4=primitive id, 5=direct, 6=light count,
                             7=direct candidates, 8=shadow visibility,
                             9=indirect, 10=GI cache hits,
                             11=photon contribution,
                             12=GI cache samples,
                             13=GI cache confidence,
                             14=UV, 15=albedo texture
```

With `--test`, renders a random sphere scene that showcases every light
type (quad, sphere, disc, cylinder, point, and spot) for testing.

## Scripted datasets

Scripts author stages with OpenUSD's own `pxr` API, vendored with Lumbre, and
render them headlessly with the realtime rasterizer and its label pass:

```bash
./lumbre --script python/examples/random_dataset.py -- out/ds 20 7
```

```python
import random
import lumbre
from lumbre import randomize as rnd
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open("assets/usd_scene_tests/semantic_classes.usda")
camera = UsdGeom.Camera.Define(stage, "/World/DatasetCam")
for frame in range(20):
    rng = random.Random(frame)          # one seed per frame: each reproduces alone
    rnd.orbit_camera(camera, target=(0, 0, 0), rng=rng, distance=(10, 16))
    rnd.clear(stage, "/World/Distractors")
    rnd.scatter_distractors(stage, "/World/Distractors", rng, count=8)
    lumbre.render(stage, "out/ds.png", frame=frame, labels=True, camera="DatasetCam")
```

Each `render` writes `ds.0007.png`, `ds.0007.labels.exr` (instance, semantic,
depth, normal) and `ds.0007.coco.json`. `frame=` only numbers the files; to
read an animated stage at a USD time code, pass `time=` (e.g.
`lumbre.render(stage, "out/anim.png", time=12, labels=True)`). A stage read at
a time code also gets a `motion` layer in its label EXR: per-pixel velocity in
pixels since the previous frame, x right and y down. The stage is read as it is at the call,
unsaved edits included, and never modified. A prim's class for segmentation is
the custom string attribute `semantic:class`, inherited down the hierarchy.

`lumbre.randomize` covers UsdPreviewSurface materials, light intensity, colour
and aim, camera orbits, poses, and distractors. Every helper is plain `pxr`
authoring that takes an explicit `random.Random`, so a randomized stage can be
saved and rendered again to the same pixels.

For large scatters, `rnd.scatter_instances(stage, "/World/Scatter", rng,
count=1000)` authors one `PointInstancer` over a few randomly coloured
prototypes instead of a thousand prims. The rasterizer draws it instanced, and
every point is still its own object in the labels, with the path
`/World/Scatter[<point>]/<prototype>`.

`lumbre-gui` plays animated stages: the Timeline panel has play/pause, step,
loop and a frame slider (Space, arrows, Home and End while it has focus), and
`lumbre-gui --scene anim.usda --frame 12` or `--play` opens at a frame or
starts playing. The viewport's Motion channel shows the velocity, and the
renderer, channel, look and effect controls live in the Viewport Settings
panel.

In `lumbre-gui`'s script editor, `lumbre.stage()` returns the loaded scene as
an editable `Usd.Stage` and `lumbre.show(stage)` puts an edited stage in the
viewport without saving it.

The `pxr` bindings are vendored by `scripts/vendor_pxr.sh`, which must run
after `native/usd_shim/build.sh` (that script re-copies dylibs the bindings
need repointed). `scripts/vendor_python.sh` re-runs it itself.

Class IDs persist beside each labelled output stem in `ds.classes.json`. Use
`classes=["chair", "table"]` to declare a fixed dataset vocabulary. Camera selection
accepts full prim paths; ambiguous short names are rejected. COCO annotation IDs
are dataset identities, with `instance_id` retaining the value stored in the mask.

See [realtime stabilization and throughput](docs/realtime-stabilization.md) for
resource reuse, queued capture, backend validation, and benchmark commands.

Test the scripting layer end to end with `scripts/test_scripting.sh`
(`--no-gui` skips the one test that opens a window). It needs `./lumbre` and
`./lumbre-gui` built, and keeps its outputs in a temp dir when anything fails.

## Baked GI

The rasterizer's diffuse ambient light comes from the environment alone unless
the scene has baked irradiance probes. Bake them with the CPU path tracer, and
every later `--raster` run, `lumbre.render` call or `lumbre-gui` session on that
scene picks them up:

```bash
./lumbre --bake-probes scene.usda --grid 8     # writes scene.probes
./lumbre -s scene.usda --raster -o out.png    # bounce light, occlusion, colour bleed
```

```python
lumbre.bake_probes(stage)                      # sidecar next to the stage's file
lumbre.render(stage, "out/gi.png")             # uses it; probes=False does not
```

A bake is deterministic and records what it saw: geometry, materials, lights,
and the environment. Change any of those and the probes no longer apply, so
rendering falls back to environment lighting and says so until you bake again.
`python/examples/baked_gi.py` builds a room with a `PointInstancer` scatter,
bakes it, and renders it both ways. Design notes and measurements are in
[realtime stabilization](docs/realtime-stabilization.md#baked-gi).

## Features

- GPU path tracer via Metal hardware ray tracing (Apple M-series)
- CPU fallback with Odin thread pool
- OBJ + MTL scene import with `map_Kd` texture loading (PNG, JPEG)
- Scene graph with TRS transforms and instancing
- Direct light sampling with multiple importance sampling
- Full light set: rectangle (quad), disc, sphere, and cylinder area lights;
  point, spot, and distant/sun lights; and an importance-sampled HDRI dome
- HDRI environment lighting (equirectangular `.hdr`) with 2D importance
  sampling, MIS against BSDF sampling, and Y-axis rotation
- 5 material types: Lambertian, Metal, Dielectric, Principled, Emissive
- Disney-style Principled BSDF (GGX + Fresnel + diffuse) with proper
  MIS for NEE on glossy surfaces
- Roughness cutoff and glossy bias controls
- Scene-scale biased GI defaults and quality presets
- GI cache confidence rejection with diagnostic debug views
- Firefly clamping
- EXR writer (FP16, multi-layer AOVs) — `.exr` output
- Animated USD: render any time code (`--frame`, `lumbre.render(time=)`),
  screen-space motion vectors in the label EXR, TAA that follows moving
  objects, and playback in `lumbre-gui`

## Requirements

- Apple Silicon Mac (M1+)
- Odin compiler (nightly)
- macOS 14+ (for Metal ray tracing)
- [OpenImageDenoise](https://www.openimagedenoise.org/) 2.x for `--denoise`

### OpenImageDenoise setup

`--denoise` uses OIDN's HDR `RT` filter with the renderer's albedo and normal
feature buffers. This preserves glossy reflections, refraction boundaries, and
fine texture detail much better than the previous edge-aware blur.

The OIDN library is loaded at runtime, so the normal build does not need an
external linker setup. Install an ARM64 macOS OIDN 2.x distribution, then either
copy the *contents* of its `lib` directory (including OIDN device libraries)
to `lib/darwin/oidn/`, or point to its main library explicitly:

```bash
LUMBRE_OIDN_LIBRARY=/path/to/libOpenImageDenoise.dylib ./lumbre --denoise ...
```

If OIDN cannot be loaded or reports an error, Lumbre emits a clear diagnostic
and writes the original, unfiltered beauty image rather than producing a
partially filtered result.
