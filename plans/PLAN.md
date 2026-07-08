# Lumbre — Biased GPU Path Tracer for Metal

> A Redshift-like partially-biased GPU renderer built in Odin + Metal for Apple Silicon.

## Philosophy

Lumbre is a **biased / partially-biased path tracer**, not an unbiased one.
We aggressively trade physical accuracy for speed, using the same techniques
that make Redshift fast:

1. **Irradiance Caching** — World-space cache of diffuse indirect lighting;
   interpolate between cache points instead of tracing full paths per pixel.
2. **Photon Mapping** — Shoot photons from lights, store in spatial index,
   look up during rendering. Handles caustics efficiently.
3. **Screen-space + temporal denoising** — Reuse samples across pixels and
   frames via edge-avoiding filtering.
4. **Aggressive clamping** — Clamp fireflies, limit ray lengths, roughness
   cutoffs. Bias controls exposed to the user.
5. **Early ray termination** — Russian roulette with aggressive cutoff;
   stop tracing paths whose contribution drops below a threshold.

**Result**: Converges to a usable image 10-100x faster than an unbiased
path tracer, at a small cost in accuracy.

---

## Stages

### Stage 0 — Current State

- Sphere-only scene, procedurally generated
- GPU path tracer via Metal mega-kernel + CPU fallback (Odin threads)
- 3 materials (Lambertian / Metal / Dielectric)
- CPU-built BVH, Metal bounding-box acceleration structure
- PNG output only, no scene file I/O

---

### Stage 1 — Mesh & Scene Infrastructure

Add triangle meshes, OBJ import, and the scene graph needed to replace
the random sphere scene.

**Status: in progress.** Mesh OBJ scenes now render through the GPU triangle
path, Cornell-box debug renders are covered by `scripts/render_cornell.sh`,
and direct lighting now uses explicit emissive-triangle sampling with MIS.

| Area | Status | Details |
|---|---|---|
| **Triangle + Mesh types** | Done | `Triangle` carries positions, normals, UVs, and material index; `Mesh` stores triangle slices and material data. |
| **OBJ parser** | Done | Loads positions, normals, UVs, face indices, material groups, and basic MTL data without external dependencies. |
| **Metal triangle AS** | Done | Uses `MTLPrimitiveAccelerationStructure` with triangle geometry and Metal hardware triangle intersections. |
| **Mesh GPU upload** | Partial | Uploads packed per-triangle vertex/index/material buffers. Multi-mesh argument buffers are still pending. |
| **Scene graph** | Pending | Add `SceneNode` with TRS/matrix transforms, mesh references, material overrides, and instancing. |
| **Area lights** | Partial | Emissive mesh triangles are extracted and sampled directly. Explicit quad/sphere `Light` primitives still need host and GPU paths. |
| **Direct light sampling + MIS** | Done | Next-event estimation samples emissive triangles, traces shadow rays, and combines direct light and BSDF emissive hits with a power heuristic. |
| **Firefly clamping** | Done | `max_radiance` is exposed through the CLI and applied in CPU/GPU paths. |
| **Roughness cutoff** | Pending | `roughness_cutoff` exists in config but is not wired into material sampling yet. |

**Next Stage 1 work:**
- Build the actual scene graph and transform/instancing path.
- Add explicit quad and sphere light primitives in addition to emissive mesh
  triangles.
- Wire `roughness_cutoff` into GPU material sampling.
- Replace per-triangle material duplication with indexed material lookup once
  multi-mesh upload is in place.

---

### Stage 2 — Biased GI Core

This is where Lumbre diverges from unbiased renderers. The two main
techniques: irradiance caching for indirect diffuse, photon mapping
for caustics.

#### 2A — Irradiance Cache (GPU)

```
Goal: Replace brute-force indirect diffuse bounces with a cached world-space
      representation. Reduces the number of indirect rays by ~90%.
```

- **Cache point structure**: `struct { pos: float3, normal: float3, irradiance: float3 }`
- **Storage**: Fixed-size ring buffer on GPU (e.g. 64K-256K points).
- **Placement**: During rendering, when a diffuse bounce occurs, check if a
  nearby cache point exists (distance + normal dot-product threshold).
  If not, evaluate indirect lighting via path tracing and store result.
- **Lookup**: Before tracing a diffuse bounce, query cache. Weight nearby
  points by distance, normal similarity, and visibility (occlusion check).
- **Spatial index**: Uniform grid or hash grid for O(1) nearest-neighbor
  lookups on GPU.
- **Bias controls**: `gi_cache_distance`, `gi_cache_normal_angle` exposed
  to user (controls interpolation quality vs. speed).
- **Convergence**: Cache points converge over samples. Results are biased:
  high-frequency indirect detail is blurred (like Redshift's GI cache).
- **Brute-force fallback**: For scenes where GI cache is too coarse, provide
  brute-force path tracing with aggressive `max_diffuse_bounces` (2-3) and
  `gi_clamp`.

#### 2B — Photon Mapping (GPU)

```
Goal: Efficient caustics + focused indirect lighting where path tracing
      struggles (specular-diffuse-specular paths).
```

**Pass 1 — Photon emission**:
- Compute kernel dispatches N photons from light sources (e.g. 1M).
- Each photon traces through the scene (specular bounces + first diffuse
  hit), stores position + direction + power in a buffer.
- Russian roulette termination.

**Pass 2 — Photon index build**:
- Hash grid spatial index over photon positions (GPU compute).

**Pass 3 — Rendering**:
- During path tracing at a diffuse hit, query nearest photons.
- Estimate flux density (kernel density estimation).
- Blend photon contribution with path-traced direct illumination.

**Controls**: `photon_count`, `photon_radius`, `photon_bounces`.

---

### Stage 3 — Materials + Textures

| Area | Details |
|---|---|
| **Principled BSDF** | Disney-style: diffuse (Lambertian + Fresnel), GGX microfacet specular, metallic, roughness, clearcoat, sheen. Split into CPU (reference) + MSL (GPU) implementations. |
| **Emissive** | Emissive color + intensity on any material. Acts as light source for photon emission and direct sampling. |
| **Texture maps** | UV per vertex. `Texture` type wrapping `MTLTexture`. Load via `stb/image` (PNG, JPEG, etc.). Samplers with filtering + mip-maps. |
| **Roughness/glossy bias** | User parameter to fake glossy reflections with a cheaper approximation below a certain roughness threshold. |

---

### Stage 4 — Production I/O

| Area | Details |
|---|---|
| **EXR writer** | Wrap `tinyexr` via Odin foreign bindings. Write half-float EXR with zip compression. |
| **EXR AOVs / passes** | GPU kernel outputs multiple float4 buffers per sample -> accumulate -> write each as EXR layer: `beauty`, `diffuse`, `specular`, `normal`, `depth`, `emission`, `alpha`. |
| **USD static mesh import** | Write Odin foreign bindings to USD C API (`#foreign` + `#link pxr`). Read `UsdGeomMesh` -> vertex data, normals, UVs. Read `UsdGeomCamera`, `UsdPreviewSurface`. |
| **Animated USD** | Time-sampled attributes. For each frame `t`, read updated vertex positions / transforms / camera. Render sequence. |
| **Render sequences** | `--frame-range 1-100`. Loop, update scene, render, write EXR per frame. |
| **OBJ + MTL** | OBJ loader + MTL material file parser for scenes without USD. |

---

### Stage 5 — Screen-space Denoising & Temporal Reuse

| Area | Details |
|---|---|
| **Bilateral / guided filter** | Post-process compute shader: edge-avoiding blur using pixel normal + depth as guide. Smooths remaining noise while preserving detail. |
| **Temporal accumulation** | For animated sequences: reproject samples from previous frame using motion vectors. Accumulate over time. Bias: lag/ghosting on fast motion, controlled by `temporal_blend` parameter. |
| **Adaptive sampling** | Per-tile variance estimate. Stop sampling low-variance tiles early. Spend budget on noisy regions. |
| **Progressive refinement** | Render at 1spp -> apply denoiser -> accumulate more samples -> re-denoise. Interactive feedback for look-dev. |

---

### Stage 6 — CLI + Performance

| Area | Details |
|---|---|
| **CLI interface** | `lumbre --scene scene.usda --width 1920 --height 1080 --spp 1000 --gi-cache --photons 500000 --frame-range 1-100 --output /frames/frame_####.exr` |
| **Bias control knobs** | Expose all bias parameters: `gi_cache_distance`, `gi_cache_normal_angle`, `max_radiance`, `max_diffuse_bounces`, `roughness_cutoff`, `photon_count`, `photon_radius`, `temporal_blend`. |
| **GPU BVH** | Compute kernel for Morton code + radix sort -> build acceleration structure entirely on GPU. Removes CPU bottleneck for large scenes. |
| **Metal optimization** | Threadgroup sizing, argument buffer pooling, pipeline caching. |
| **Tile rendering** | Split image into tiles for large resolutions + memory management. |

---

## Architecture (Target)

```
lumbre/
├── main.odin                        <- CLI entry, scene load, render dispatch
├── cli.odin                         <- Arg parsing, bias control params
├── scene/
│   ├── scene.odin                   <- Scene graph, nodes, instances
│   ├── mesh.odin                    <- Triangle/Mesh types, vertex buffers
│   └── loader/
│       ├── obj.odin                 <- OBJ + MTL parser
│       └── usd.odin                 <- USD C API bindings + reader
├── core/
│   ├── types.odin                   <- Primitive, Material, Light, Camera
│   ├── transform.odin              <- TRS transforms
│   ├── texture.odin                <- Texture loading + MTL sampler
│   └── film.odin                   <- Tile buffer, accumulation, AOVs
├── bsdf/
│   ├── principled.odin             <- CPU reference
│   └── principled.metal            <- GPU MSL implementation
├── light/
│   ├── light.odin                  <- Light types + sampling
│   └── photon.odin                 <- Photon emission + hash grid
├── gi/
│   ├── irradiance_cache.odin       <- GI cache host code
│   ├── irradiance_cache.metal      <- GPU GI cache kernel
│   ├── photon_map.metal            <- GPU photon tracing + lookup
│   └── denoise.metal               <- Bilateral / temporal filter
├── render_cpu.odin                 <- CPU path tracer (reference)
├── render_gpu_darwin.odin          <- Metal GPU dispatch
├── shaders/
│   └── pathtrace.metal             <- Main kernel (biased path tracing + AOVs)
├── output/
│   ├── exr.odin                    <- EXR writer (tinyexr bindings)
│   └── png.odin                    <- PNG writer (stb_image)
├── bvh.odin                        <- CPU BVH builder
├── camera.odin                     <- Camera model
├── rng.odin                        <- CPU RNG
└── vec3_utils.odin                 <- Math utilities
```

## Key Design Decisions

- **Biased by default**: All "cheats" enabled by default. User can dial them
  back for more accuracy at the cost of speed.
- **GPU-first**: MSL kernel does the heavy lifting. CPU code is orchestration,
  scene loading, and fallback reference.
- **No OptiX / CUDA**: Metal-only. Apple Silicon native.
- **USD optional**: OBJ + MTL for simple scenes. USD for production pipelines.
- **tinyexr for EXR**: Single-header C lib, easy to wrap in Odin.
- **Hash grids for spatial queries**: Simpler than k-d trees on GPU. Good
  enough for GI cache + photon lookups.
