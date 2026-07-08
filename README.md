# Lumbre

A **biased GPU path tracer** for Apple Silicon, written in Odin + Metal.

Inspired by Redshift — trades physical accuracy for speed using
irradiance caching, photon mapping, and aggressive bias controls.
Targets 10-100× faster convergence than unbiased renderers.

## Status

**Stage 2 complete** — mesh import, scene graph, explicit lights,
roughness cutoff, indexed materials, scene-scale biased GI,
irradiance cache confidence rejection, photon mapping, quality presets,
and GI diagnostics.

## Quick start

```bash
odin build .
./lumbre --scene assets/cornell_box.obj --width 1920 --height 1080 --spp 100
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
  --roughness-cutoff <float> Bias: treat rough materials as diffuse (default 0.95)
  --output, -o <file.png>    Output file (default render.png)
  --cpu                      Force CPU renderer
  --gpu                      Force GPU renderer
  --quality <preset>         Quality preset: draft, preview, final
  --gi-dist <float>          Cache lookup distance (default auto)
  --photon-radius <float>    Photon search radius (default auto)
  --debug <mode>             Debug: 1=albedo, 2=normal, 3=depth,
                             4=primitive id, 5=direct, 6=light count,
                             7=direct candidates, 8=shadow visibility,
                             9=indirect, 10=GI cache hits,
                             11=photon contribution,
                             12=GI cache samples,
                             13=GI cache confidence
```

Without `--scene`, renders a random sphere scene with explicit
quad + sphere lights for testing.

## Features

- GPU path tracer via Metal hardware ray tracing (Apple M-series)
- CPU fallback with Odin thread pool
- OBJ + MTL scene import
- Scene graph with TRS transforms and instancing
- Direct light sampling with multiple importance sampling
- Explicit quad and sphere light primitives
- 5 material types: Lambertian, Metal, Dielectric, Principled, Emissive
- Roughness cutoff bias control
- Scene-scale biased GI defaults and quality presets
- GI cache confidence rejection with diagnostic debug views
- Firefly clamping

## Requirements

- Apple Silicon Mac (M1+)
- Odin compiler (nightly)
- macOS 14+ (for Metal ray tracing)
