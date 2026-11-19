# Realtime renderer handoff (2026-11-12, updated 2026-11-18)

Where the realtime (SDL_GPU rasterizer) work stands, how to work on it safely,
and what comes next. Read this first, then `docs/realtime-stabilization.md`
(the running design record) and `plans/REALTIME_PHASE04.md` (the phase plan).
`plans/` is gitignored, so the phase plans exist only in the original checkout.
In a fresh clone, the baked-GI section at the end of this file is the plan.

## State

Everything below is pushed to `origin/main`. Four sessions landed these, oldest
first:

| Commit | What |
|---|---|
| `b592071` | Port the path tracer's Metal calls to Odin `dev-2026-11`'s regenerated darwin bindings (the tree did not compile before this) |
| `9fd58ae` | TAA (16-point Halton jitter, depth reprojection, YCoCg variance clip) and per-pass GPU timing (`realtime/profile.odin`) |
| `a158a95` | Spot light shadows: 8 × 1024² perspective layers |
| `1fa6e8c` | Render test page `docs/render-tests/index.html` + `scripts/render_tests.py` |
| `94079a3` | Restore file modes flipped to 755 by three earlier commits (see Gotchas) |
| `f24c459` | Point/sphere light shadows: 4 lights × 6 faces of 512², shared `draw_shadow_layers` cache |
| `a6d16b1` | Render page: point shadows |
| `a0f8a68` | Area lights with LTC: `realtime/ltc` (table fitted to Lumbre's own GGX by `tools/ltc_fit`), `realtime/area_lights.odin` |
| `f4a8c9f` | Render page: area lights |
| `3a5b8a8` | Path tracer cylinder light fix (only the facing half emits; was 2× too bright) and USD `CylinderLight` centred (was placed by its end) |
| `046c780` | Render page: cylinder fix |
| `60edf7e` | This handoff |
| `66cf83f` | CPU path tracer floors GGX roughness at 0.015 like the GPU one (roughness-0 metals rendered black; the Cornell box's red wall is one) |
| `73574be` | Baked GI: `--bake-probes`, `core/probes.odin`, `realtime/probes.odin`, DDGI-style lookup in the lighting pass |
| (the commit after `73574be`) | Baked GI scripting (`lumbre.bake_probes`, `render(probes=)`), example, tests, render page, docs |
| `7607e59` | SSR: no more stripes in reflections seen at a grazing angle (Cornell box's mirror wall) |
| `7362353` | Probes in the forward (glass) and fog passes |
| `448e813` | Probe reflections: 16×16 radiance per probe (file format 3), SSR misses fall back to it |
| `a234295` | Probe relocation out of solids and away from surfaces, inactive probes, radiance lobe sized to the ray count (file format 4) |
| `27f9663` | Area light shadows: omni slots for quad/disc/cylinder, PCSS softened by light size, fog shafts |
| `c2132c7` | Area light shadows at half the cost: map-space search and filter, 8 search taps, 8-tap first pass, shade first |
| (the commit after `c2132c7`) | Area light shadows shaped like the light (a tube's kernel is long and thin), tap spiral turned per pixel and per TAA frame so TAA averages the grain |
| `12db412` | This handoff: Phase 04 closed, quality pass planned |
| `8459a20` | Environment prefilter and LUT use the BRDF's alpha = roughness and exact Smith; filtered importance sampling; exact diffuse irradiance |
| `8063e20` | Shadow slots go to the lights that matter most to the view, re-ranked each frame |
| `c38af30` | Area light specular takes Fresnel at the light's direction, not the lobe average |
| `e4266ba` | Path tracers' GGX VNDF sampler fixed (grazing glossy reflections were ~30% dark) |
| `07edd9f` | Glossy SSR: a VNDF ray per pixel, read from a blur pyramid, shared among neighbours by a resolve pass |
| `a46d266` | Glossy SSR rays traced at half resolution; mirrors traced in the resolve |
| `b7056a8` | GTAO in place of SSAO, and an AO debug view (`--raster-view 10`) |
| `6277c70` | Fog: area lights by solid angle, light dimmed on its way in, multiple scattering, colour |
| `6c27f0a` | Glass laid down per object, back faces then front, each over a fresh copy: glass is seen through glass |
| `e97e4d1` | Glass casts tinted, partial shadows: glass maps beside every shadow map |

Phase 04 is done: steps A, B and C (baked GI, steps 12–14). TAA and
local-light shadows were listed as out of scope in the Phase 04 plan, but are
done too. Baked GI's design and numbers are in `docs/realtime-stabilization.md`
under "Baked GI".

## Verify before and after any change

```sh
odin test core && odin test output && odin test realtime && odin test realtime/ltc && odin test importers
python3 scripts/check_shader_bindings.py
bash scripts/test_realtime_backend.sh metal   # GPU tests under Metal validation + 102 scripting checks
odin build . -out:lumbre && odin build gui -out:lumbre-gui
```

GPU tests only: `LUMBRE_GPU_DRIVER=metal odin test realtime -define:GPU_TESTS=true -define:ODIN_TEST_THREADS=1`.
Expected now: realtime 41 (64 with GPU), core 15, ltc 4, output 7, importers 4, scripting
115 (`--no-gui`, as the backend script runs it) or 131 with the GUI test.

Regression practice used all session: build HEAD in a temporary
`git worktree` (binaries must sit in the repo root, because they load
`@loader_path/lib`). Render the same scenes with both builds and `cmp` the
PNGs. Byte-identical is the bar for anything a change is not meant to touch.
The path tracer is reproducible with `--deterministic --spp 16`. Headless
raster runs are `./lumbre -s scene.usda --raster -w W -h H -o out.png`.
`-s` is required.

For a new feature, add a step to `STEPS` in `scripts/render_tests.py`, run
`python3 scripts/render_tests.py --step <id>`, and commit the images.
"Before" variants name a commit and are built once into `.render_tests/`
(gitignored).

`LUMBRE_RASTER_PROFILE=1` prints per-pass GPU times. They are approximate:
SDL_GPU has no timestamp queries, so each pass is fenced. Each pass carries
about 0.3 ms of overhead, and a single frame is a cold first frame. Use
`--taa 1`, which averages 16 frames, for stable numbers.

## Gotchas (each cost time once)

- **File modes.** Nearly every file in the checkout is 755 on disk while the
  repo stores 644, so `git status` shows ~700 phantom `M` entries and
  `git add` stages a mode change. Before every commit, run
  `git diff --cached --summary | grep "mode change"`, then
  `git update-index --chmod=-x <files>`. Don't touch `core.fileMode`.
- **Shader blobs.** `scripts/build_shaders.sh` with the pinned
  `.tools/slang/bin/slangc` rewrites every committed blob differently.
  Compile only the shader you changed:
  `.tools/slang/bin/slangc realtime/shaders/X.slang -target metal|spirv -entry fragmentMain -stage fragment -o realtime/shaders/build/X.fragment.msl|spv`.
  Still list new shaders in `build_shaders.sh` and in `realtime/shader.odin`.
- **Slang → MSL.** `.Load()` on a combined `Sampler2D` generates Metal that
  does not compile. Use point `SampleLevel` at texel centres.
- **Bindings.** Samplers come before storage buffers, contiguously, per stage.
  A new sampler slot shifts the buffers, the Odin bind arrays, the
  `Shader_Bindings` counts and `EXPECTED_LIGHTING` in the checker. Lighting
  uses 16 samplers (Metal's limit; the glass maps are storage textures), forward
  14, fog 5, SSR trace 4, SSR resolve 9.
- **Uniform mirrors.** `Lighting_Uniforms` (camera.odin) and `LightingUniforms`
  (shading.slang) must match field for field. The same goes for `Light_GPU`
  and `Light`, and for the constants duplicated between shadows.odin/ltc.odin
  and shading.slang (`MAX_*_SHADOWS`, resolutions, `OMNI_FACE_TAN`, LTC
  `SIZE` and `MIN_ROUGHNESS`).
- **Headless defaults.** `main.odin` sets `taa = false` for headless runs.
  Dataset output is FXAA and byte-stable. `--taa 1` makes each still 16 frames
  from a cleared history. The viewport defaults to TAA.
- **The default sky** lights any scene without a DomeLight. For clean
  measurements, add `def DomeLight "black" { float inputs:intensity = 0 }`.
- **Mutation-test new GPU tests.** Break the feature, confirm the test fails,
  restore. Two tests this session passed with the feature broken until
  tightened.
- **Probe sidecars go stale silently on purpose.** A `.probes` file records a
  revision of what it was baked from. Any change, including to the revision
  scheme itself, makes it inactive with one printed line, and lighting falls
  back to the environment. If probes "do nothing", look for that line, then
  rebake.
- **The `lumbre` Python package runs from a staged copy** in
  `lib/darwin/python3.12/lumbre`, made by `scripts/vendor_python.sh` and
  tracked in git. After editing `python/lumbre/`, copy it across
  (`cp -R python/lumbre lib/darwin/python3.12/`) and commit both, or scripts
  see the old API.
- **COCO ids hash the output file name**, so two renders of the same frame
  to different names differ in `id`/`image_id` only. Compare annotations
  without those.
- **Toolchain.** Homebrew Odin `dev-2026-11` moved Foundation to
  `core:sys/darwin/Foundation` and renamed Metal binding enums. `core` is
  ported. Any new Metal code must use the new names.

## Shadow and light model now

- Sun: 4 cascades at 2048² (unchanged).
- Spot lights: the 8 that matter most to the view own a layer, stored in
  `Light_GPU.params.w` (`shadow_slots_assign`, re-ranked each frame).
- Point and area lights (sphere, quad, disc, cylinder): the 4 that matter
  most own a 6-face slot. Faces are layers of a 2D array, picked by dominant axis, not a
  cube map. Area lights' shadows are PCSS, softened by the light's size; see
  "Area light shadows" in `docs/realtime-stabilization.md`.
- Local maps depend on the light and the casters only. They are redrawn when
  the `geometry_uploads`/`instance_uploads` stamp or the matrix changes. A
  camera move redraws a slot only when it changes which lights are chosen.
- Area lights (quad, disc, sphere, cylinder) use LTC on surfaces and are
  two-sided, like the path tracer. Fog takes the solid angle they fill; the
  cluster reach still treats them as points. Fog shadows them (hard).

## Known gaps and loose ends

- **GUI checks done** (2026-11-18, by hand): TAA settling, local shadows while
  navigating, the Baked GI checkbox and its tooltip.
- **Vulkan is on hold.** It has not been run; only Metal was exercised.
- The CPU path tracer read about 4% under the GPU one on the tube test.
  Not investigated.
- The path tracer's light-selection power for cylinders still uses the full
  lateral area. It is only a selection weight, so not a bias.
- Per-frame noise rotation for AO and fog was tried and reverted: no
  measurable gain, since their 4×4 blurs already cancel the patterns.
- **Baked GI limits:** the lighting, forward (glass) and fog passes all read
  probes; fog uses per-probe means without visibility, so it can leak through
  thin walls. Probe reflections are 16×16 per probe: a mirror's fallback is
  blurred, with soft blotches where one probe's view differs from its
  neighbour's. Indirect shadow between close objects is exaggerated at the
  default grid (the Cornell box's green wall beside the short block);
  `--grid 16` halves it. Probes are relocated once, at bake time; a
  probe that cannot get out of a solid within its cell is switched off.
  The CPU path tracer has no NEE for emissive meshes, so scenes lit that way
  need more rays. The GUI loads a sidecar when a file is opened, not when it
  is rebaked while open (reopen the file). `lumbre.bake_probes` is CLI-host
  only; the GUI script editor refuses it by name. The Cornell box's short block
  front is 9 levels bright in its bottom quarter, where it meets the floor:
  indirect light finer than the probe grid. GTAO at any radius does not
  reach it; `--grid 16` or screen-space GI would.

## Learned in the quality pass (2026-11-18)

- **Metal allows 16 samplers per stage.** `lighting_fs` and `forward_fs` are at
  it. New inputs there must be storage textures read with `Load` (as the glass
  maps are), or buffers. Adding samplers compiles on the Slang side and fails
  only when Metal builds the library at startup.
- **The path tracer is not always the reference:**
  - Its default `--roughness-cutoff 0.95` renders rough Principled surfaces
    as diffuse. Pass `--roughness-cutoff 2` for glossy references.
  - Its direct light cannot pass through glass, so its glass shadows are
    opaque; realtime's are now correct and differ from it.
  - Before `e4266ba` its GGX sampler was biased at grazing angles. Render-page
    references made before that are slightly dark on glossy surfaces seen
    low.
- **Check the geometry before calling the path tracer wrong.** A reflection
  that "vanished" on a glossy strip had landed on the neighbouring strip.
- **Old builds for comparison:** `render_tests.binary_for('<rev>', None)` in
  `scripts/render_tests.py` builds and caches one under `.render_tests/bin/`.
  It is quicker than a worktree by hand, and scenes byte-compare against it.
- **GPU profiling is noisy** (±20% run to run, `LUMBRE_RASTER_PROFILE=1`). Time
  twice, and old against new in the same sitting.
- **Mutation-test every new GPU test.** This session's tests each fail with
  the old behaviour; several first drafts did not.
- **A stand-in texture must cover every index the shader asks for.** The
  1 × 1 glass stand-in has one layer; an unclamped omni face index read zero
  and blacked out every point light in scenes without glass.

## Open loose ends from the quality pass

- SSR stills (no TAA) keep a fine grain; TAA removes it.
- GTAO reads about 0.96 on flat floors in the AO view; not the horizon
  search (radius-independent), probably depth or normal precision.
- Fog: in the Beer-Lambert test's height-varying fog, single scattering is
  about 0.03 brighter than the closed-form estimate at its pixel; not traced.
  The fog pass is 30-60% slower than before the area-light change.
- Fog sun shafts pass glass untinted; the probe bake treats glass as opaque.
- Past `GLASS_LAYERED_MAX` (16) glass objects, layering falls back to all
  back faces then all front faces. Glass balls read milkier than the path
  tracer (screen-space refraction's limit).
- The Cornell short block's bright foot is indirect light finer than the probe
  grid; try `--grid 16` or screen-space GI.
- Rough metal under an HDRI reads 6-9 levels bright: the split sum's
  view-equals-normal assumption.

## Next

Phase 04 is complete and checked in the GUI. Vulkan is on hold. The next
pass is realtime quality, taking the open rendering items from the Phase 04
backlog, in this order:

1. ✅ Environment prefilter and the analytic BRDF agree on roughness
   ("Environment reflections" in `docs/realtime-stabilization.md`).
2. ✅ Shadow slots go to the most important lights, not the first in scene order
   ("Shadow slot ranking").
3. ✅ The two path-tracer mismatches ("Area light Fresnel"). The floor was
   Fresnel in the LTC specular, fixed. The short block's front is within 1–2
   levels except at its foot: indirect light finer than the probe grid.
4. ✅ Glossy reflections ("Glossy reflections"). SSR now fades out by 0.75.
5. ✅ GTAO in place of the full-res SSAO ("GTAO"). Not measurably closer to the
   path tracer, but visibility with a meaning and a test; AO view added.
6. ✅ Fog: area lights by solid angle, a settable colour, multiple scattering
   ("Fog: area lights, colour, multiple scattering"). The pass is 30–60% slower.
7. ✅ Glass seen through glass, and tinted, partial shadows from glass ("Glass
   seen through glass", "Glass shadows").
8. **Motion vectors** -- next, not started. See "Motion vectors: where to
   start" below.

Progress on each lands in `docs/realtime-stabilization.md` and here.

## Motion vectors: where to start

The backlog item is "motion vectors (Phase 02) -- needs time-sampled
transforms first", next to "time-sampled / animated transforms as a render
input; import reads default time only". Nothing for it exists yet. What the
last session found before handing off:

- **Import reads default time.** `native/usd_shim/usd_shim.cpp` takes each
  prim's transform with `GetLocalTransformation` (around line 454) at
  `UsdTimeCode::Default()`. The PointInstancer path (around line 1889) already
  falls back to `EarliestTime()` for positions authored only as samples, and
  `ComputeInstanceTransformsAtTime` takes a time; that is the pattern to
  extend. The Odin side is `importers/usd.odin` and `importers/scene.odin`;
  the scene graph is `core/scene_graph.odin`.
- **A suggested order:**
  1. ✅ Shim: an import time code, and the stage's start/end/fps, through the C
     API (`usd_shim.h`, "Time"): `usd_shim_get_time_info`,
     `usd_shim_get_local_transform_at`, `usd_shim_get_point_instancer_at`
     (a NaN time is default time; `USD_SHIM_TIME_DEFAULT` in Odin) and
     `usd_shim_transform_might_be_time_varying` (own ops only, not
     ancestors). The time-less functions read default time as before; all
     22 USD test scenes render byte-identical. Tests:
     `importers/usd_time_test.odin`, mutation-tested. See "Rebuilding the
     USD shim" below.
  2. Importer and CLI: a frame or time to render (`--frame`) and, alongside
     each node's world transform, the previous frame's. Instances of a
     PointInstancer need theirs too.
  3. Realtime: a previous-transform instance buffer beside the current one
     (`scene_upload.odin`), and a screen-space velocity target from the
     G-buffer or label pass. It must be jitter-free like the other label
     channels: labels never see TAA (`labels.odin`).
  4. Output: a motion channel in the label EXR (`output/`), and the Python API
     (`python/lumbre/`; remember to copy it to `lib/darwin/python3.12/lumbre`).
  5. TAA can then reproject moving objects by their velocity. Today it
     reprojects by depth and camera motion only, so moving objects ghost.
- **Tests to aim for:** a cube translated between two time samples gives
  velocity equal to its projected displacement, and zero for static
  geometry; the label EXR round-trips it; a static scene is byte-identical.
- **A scene with motion:** `assets/usd_scene_tests/animated_transforms.usda`,
  frames 1-24 at 24 fps: `mover` (translate as samples only, +0.25 X per
  frame), `spinner` (15° Y per frame), an animated `parent` with a static
  `child`, a `scatter` PointInstancer (positions as samples only, +0.05 Y per
  frame), and static geometry. Today's importer collapses `mover` and
  `parent` to the origin: at default time, sample-only ops have no value.
  Step 2 must pick a time when the stage has a range (start time, or
  `--frame`).

## Rebuilding the USD shim (2026-11-18)

The OpenUSD build the shim compiles against was gone, so it was rebuilt from
source. To rebuild the shim again:

```sh
PYTHON_ROOT=/Users/varomix/dev/OpenUSD-python3.12 bash native/usd_shim/build.sh
USD_ROOT=/Users/varomix/dev/OpenUSD bash scripts/vendor_pxr.sh   # always after build.sh
```

- **OpenUSD v26.08** (`pxrInternal_v0_26_8`), source in
  `/Users/varomix/dev/OpenUSD-src`, installed to `/Users/varomix/dev/OpenUSD`
  (the scripts' default `USD_ROOT`) by
  `build_usd.py --onetbb --no-usdview --no-tests --no-examples --no-tutorials --no-docs --build-python-info <py> <include> <libpython> 3.12`.
  `--onetbb` is required: the vendored set and the shim's CMakeLists assume
  oneTBB (`libtbb.12.dylib`); the default is TBB 2020.
- **Python 3.12.15** (python-build-standalone via `uv python install 3.12`),
  copied to `/Users/varomix/dev/OpenUSD-python3.12` with
  `install_name_tool -id @rpath/libpython3.12.dylib` on its libpython, so the
  vendored copy resolves like conda's did. The old engine was conda 3.12.2;
  `scripts/vendor_python.sh` was rerun with
  `PYSRC=/Users/varomix/dev/OpenUSD-python3.12/lib/python3.12`. This Python
  builds nearly every extension module into libpython, so `lib-dynload` holds
  two modules and libz/libffi are no longer vendored.
- `vendor_python.sh` deletes all of `lib/darwin/python3.12` first and then
  re-vendors pxr and restages `lumbre` itself; a failed run leaves it empty.
- 26.x installs the bindings to `lib/python3.12/site-packages/pxr`;
  `vendor_pxr.sh` now looks there too.
- The vendored dylibs keep LC_RPATHs into the build tree (as before); every
  load command is `@loader_path`, so nothing resolves through them.

