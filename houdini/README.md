# Lumbre Houdini plugin

This directory contains the Houdini 21 Hydra render-delegate integration. It
is deliberately isolated from the root Odin CLI build: `odin build .` and
`./lumbre` remain the standalone renderer workflow.

## Current status

`HdLumbreRendererPlugin` is a discoverable HDK plugin with a lifecycle-safe
Hydra delegate. It now hosts Lumbre's renderer through the Houdini-safe bridge
dylib (`libLumbreBridge.dylib`, built from `lumbre_bridge/`, which imports only
Lumbre's USD-free `core`). The delegate loads the bridge, uploads synced meshes
as world-space triangles, derives the camera/resolution from the render-pass
state, renders on the CPU, and publishes the result into Hydra's color AOV. If
the bridge is unavailable it falls back to a diagnostic gradient.

The viewport renders **progressively** with Lumbre's Metal path tracer: the
bridge keeps one GPU renderer, the scene's GPU resources and an accumulation
buffer for the life of the delegate, and each Hydra execute adds a batch of
samples. A camera move, a resolution or render-setting change, or a scene edit
starts the image over; it stops refining at the target sample count, which is
also when `husk` considers the frame done. The colour AOV is linear radiance
(HDR), so Houdini's display transform applies once. The CPU path remains a
fallback via `lumbre_bridge_set_use_gpu`.

Implicit prims (Sphere, Cube, Cone, Cylinder, Capsule, Plane) are tessellated
by Hydra's implicit-surface scene index before they reach the delegate, and
material bindings resolve for the `full` purpose (falling back to all-purpose),
as a final-quality renderer's should. Hydra meshes preserve
their material binding, face-varying UVs, and authored or computed-smooth
normals. Catmull-Clark and Loop meshes are refined at level 2 through Houdini's
OpenSubdiv runtime, including face-varying UV seams.

Materials read both **UsdPreviewSurface** and **MaterialX `standard_surface`**
(the delegate requests the `mtlx` render context, falling back to the universal
one). The delegate fills the same field set the CLI's `usd_shim` importer does —
base colour, roughness, metalness, specular + specular colour, IOR, opacity,
transmission + transmission colour (glass), coat, and subsurface. Base colour,
normal, and emission textures are followed by connection; roughness and
metalness are resolved as separate maps with an explicit channel and scale/bias,
folding the multiply/invert/separate nodes crossed on the way (a gloss map
inverted into roughness, an ORM texture's G/B channels, ...). The bridge holds
each material as a `core.Imported_Material` and lets core pack roughness +
metalness into one ORM texture and run `imported_material_to_principled` — the
same conversion the CLI uses. Package assets (`.usdz[…]`, `.exr`) are decoded
through Houdini's Hio and pushed into those descriptors.

Lights and the HDRI dome go through the **same** UsdLux→Lumbre conversion as the
CLI. The delegate forwards each light's authored parameters (intensity,
exposure, colour, `normalize`, shape sizes, shaping cone) plus its world
transform through the bridge, and `core/usd_light.odin`
(`usd_make_light_from_params` / `usd_dome_to_environment`) applies the
area-normalization, spot-cone, and radiance math once, for both front ends. A
`UsdLuxDomeLight` becomes `Scene.environment`, rebuilt only when the dome
actually changes: an HDRI texture (`.hdr` via stb, `.exr` via the pure-Odin
reader in `core/exr_read.odin`) tinted by the dome colour, or a uniform
constant-colour environment when the dome has no texture. Additional AOVs and
depth-of-field remain future work.

## Render settings

| Setting | Default | Meaning |
|---|---|---|
| `samples` | 128 | Samples per pixel before the image stops refining |
| `samples_per_update` | 4 | Samples added per viewport update |
| `max_depth` | 20 | Maximum bounces per path |

They appear in the viewport's display options and can be authored on a
RenderSettings prim for `husk`.

## Local build

The default target is the Houdini install marked `Current`
(`/Applications/Houdini/Current`, Houdini 22.0.387 here). The build reads that
install's C++ standard and Python version from its HDK makefile, so Houdini 21
and 22 both build. Override the install when needed:

```bash
HOUDINI_INSTALL=/Applications/Houdini/Houdini22.0.387 \
  houdini/scripts/build_plugin.sh
```

The plugin links the Houdini USD, OpenSubdiv and Python libraries it uses. It
used to rely on Houdini having loaded them already, which Houdini 22's `husk`
had not, and the plugin then failed to load without an error.

`build_bridge.sh` works around an Odin regression (since Odin commit
`9aa84b5e3`) where macOS shared-library links pass `-init` with literal quotes:
when the link fails that way it re-runs Odin's printed link command through a
shell.

`build_plugin.sh` first builds the bridge dylib (`build_bridge.sh`, which runs
a C smoke test and asserts the dylib links no USD), then compiles the Hydra
plugin against Houdini's HDK. To build only the bridge:

```bash
houdini/scripts/build_bridge.sh
```

Both scripts install under `houdini/install/` (`usd_plugins/HdLumbre/` and
`lib/`). The plugin finds the bridge via an rpath to `install/lib`. Launch Houdini through the wrapper:

```bash
houdini/scripts/launch_houdini.sh
```

It clears the three standalone-runtime variables that conflict with Houdini
(see below), sources `houdini_setup`, and adds Lumbre's Hydra plugin and
Houdini configuration paths.

To render a stage headlessly through the same Hydra path the viewport uses:

```bash
houdini/scripts/husk_lumbre.sh -o out.exr --res 1280 720 scene.usd
```

Set `LUMBRE_HOUDINI_DEBUG=1` for per-frame delegate and bridge diagnostics
(mesh and material sync, parsed material values, converted lights).
To inspect plugin discovery on the first launch:

```bash
TF_DEBUG=PLUG_* houdini/scripts/launch_houdini.sh
```

The startup log must include the Lumbre `HdLumbre/resources` directory and a
line for `HdLumbre/resources/plugInfo.json`. Then select **Lumbre** from the Solaris
Stage viewport's render-delegate menu. `TF_DEBUG` is optional after discovery
is working.

## Runtime boundary

The plugin compiles against the installed Houdini HDK/USD libraries. It must
never link to `lib/darwin/libusd_shim.dylib` or Lumbre's vendored OpenUSD
libraries; the standalone CLI owns those dependencies.

Do not set `DYLD_LIBRARY_PATH`, `DYLD_FALLBACK_LIBRARY_PATH`, or `PYTHONPATH`
to `/Users/varomix/dev/OpenUSD` when launching Houdini. Those variables make
Houdini load an incompatible OpenUSD/MaterialX runtime and prevent Solaris
from starting correctly.
