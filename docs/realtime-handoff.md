# Realtime renderer handoff (2026-11-12)

Where the realtime (SDL_GPU rasterizer) work stands, how to work on it safely,
and what comes next. Read this first, then `docs/realtime-stabilization.md`
(the running design record) and `plans/REALTIME_PHASE04.md` (the phase plan).
`plans/` is gitignored, so the phase plans exist only in the original checkout.
In a fresh clone, the baked-GI section at the end of this file is the plan.

## State

`main` is 15 commits ahead of `origin/main`, nothing pushed. The last session
landed these, oldest first:

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

Phase 04 steps A and B are done. Step C (baked GI, steps 12–14) is not started.
TAA and local-light shadows were listed as out of scope in the Phase 04 plan,
but are now done.

## Verify before and after any change

```sh
odin test core && odin test output && odin test realtime && odin test realtime/ltc
python3 scripts/check_shader_bindings.py
bash scripts/test_realtime_backend.sh metal   # GPU tests under Metal validation + 102 scripting checks
odin build . -out:lumbre && odin build gui -out:lumbre-gui
```

GPU tests only: `LUMBRE_GPU_DRIVER=metal odin test realtime -define:GPU_TESTS=true -define:ODIN_TEST_THREADS=1`.
Expected now: core 6, realtime 39 (50 with GPU), ltc 4, output 7, scripting 102.

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
  uses 15 samplers, forward 13, fog 5.
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
- **Toolchain.** Homebrew Odin `dev-2026-11` moved Foundation to
  `core:sys/darwin/Foundation` and renamed Metal binding enums. `core` is
  ported. Any new Metal code must use the new names.

## Shadow and light model now

- Sun: 4 cascades at 2048² (unchanged).
- Spot lights: the first 8 in scene order own a layer, stored in
  `Light_GPU.params.w`.
- Point and sphere lights: the first 4 own a 6-face slot. Faces are layers of
  a 2D array, picked by dominant axis, not a cube map.
- Local maps depend on the light and the casters only. They are redrawn when
  the `geometry_uploads`/`instance_uploads` stamp or the matrix changes, never
  on camera moves.
- Area lights (quad, disc, sphere, cylinder) use LTC on surfaces and are
  two-sided, like the path tracer. Fog and the cluster reach still treat them
  as points. Quad, disc and cylinder lights cast no shadows.

## Known gaps and loose ends

- **Not checked in the GUI.** TAA settling (`rt_settle` in `gui/viewport.odin`),
  the TAA / "GPU pass times" checkboxes, and local shadows while navigating
  have only been tested headless.
- **Vulkan** has not been run; only Metal was exercised.
- The area light scene's floor reflection under the sphere light reads
  brighter in realtime (153 vs 137 sRGB), and the near floor darker (86 vs 97,
  missing bounce light).
- The CPU path tracer read about 4% under the GPU one on the tube test.
  Not investigated.
- The path tracer's light-selection power for cylinders still uses the full
  lateral area. It is only a selection weight, so not a bias.
- Per-frame noise rotation for AO and fog was tried and reverted: no
  measurable gain, since their 4×4 blurs already cancel the patterns.
- Area lights in fog, shadows for quad/disc/cylinder lights, and importance
  ranking of shadowed lights (currently scene order) are all open.

## Next: baked GI (Phase 04 step C)

This is the main realtime gap. Every render-page comparison shows realtime
darker in shadow than the path tracer. The plan's decisions
(`plans/REALTIME_PHASE04.md`, steps 12–14):

1. **Bake** (`lumbre --bake-probes scene --grid N`). An axis-aligned probe grid
   over the scene bounds. The path tracer traces irradiance to L2 SH per probe,
   with biased caches off so the bake is deterministic, plus depth moments for
   visibility. Output is a sidecar keyed by geometry and light revision.
2. **Consume.** Load the sidecar when its revision matches. Trilinear SH with
   visibility weighting replaces or scales the IBL diffuse term. If probes are
   stale or missing, fall back to environment lighting and say so.
3. **Script and docs.** `lumbre.bake_probes(...)`, an example scene,
   documentation, and a render-page step. Compare against the
   `--deterministic` Cornell box, and check that two bakes are byte-identical.

Starting points: `realtime/environment.odin` and `ibl()` in
`realtime/shaders/shading.slang` for where diffuse ambient enters, and
`realtime/revisions.odin` for revision keys. On the path tracer side,
`core/render_gpu_darwin.odin` holds the GPU entry point that a bake kernel
would sit beside.
