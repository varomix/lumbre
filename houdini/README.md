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

GPU beauty by default: the bridge renders with Lumbre's Metal ray tracer (the
CPU path is a fallback via `lumbre_bridge_set_use_gpu`). Hydra meshes preserve
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

## Local build

The default target is the locally installed Houdini 21.0.751. Override it when
needed:

```bash
HOUDINI_INSTALL=/Applications/Houdini/Houdini21.0.751 \
  houdini/scripts/build_plugin.sh
```

`build_plugin.sh` first builds the bridge dylib (`build_bridge.sh`, which runs
a C smoke test and asserts the dylib links no USD), then compiles the Hydra
plugin against Houdini's HDK. To build only the bridge:

```bash
houdini/scripts/build_bridge.sh
```

Both scripts install under `houdini/install/` (`usd_plugins/HdLumbre/` and
`lib/`). The plugin finds the bridge via an rpath to `install/lib`. From a
shell where `houdinifx` is already available, launch through the small wrapper:

```bash
houdini/scripts/launch_houdini.sh
```

It clears the three standalone-runtime variables that conflict with Houdini,
then adds Lumbre's Hydra plugin and Houdini configuration paths.
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
