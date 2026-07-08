# GPU Raytracer — Metal Hardware RT on Apple M3 Max

## Architecture

**GPU compute + hardware ray tracing via `vendor:darwin/Metal`**

Each sphere becomes an AABB bounding box in a Metal primitive acceleration
structure. A custom MSL intersection function performs the actual ray-sphere
math. The kernel handles material evaluation and path tracing.

## Files

| File | Role |
|---|---|
| `main.odin` | GPU host: device setup, scene upload, AS build, dispatch, readback |
| `raytrace.metal` | MSL shader: kernel + intersection function + GPU RNG + material logic |

## Data Flow

```
CPU (main.odin)                       GPU (raytrace.metal)
─────────────────                     ──────────────────
random_scene()
  ↓
Sphere[] → sphere_buffer (f32)        kernel reads sphere_buffer[]
Sphere[] → bbox_buffer (AABB f32)
  ↓
MTLAccelerationStructure(build AS)    kernel reads AS via intersector
  ↓
MTLIntersectionFunctionTable           intersection_function() called per AABB hit
  ↓
MTLComputePipeline (MSL compiled)     kernel dispatched as (width×height) grid
  ↓
dispatchThreads()                     each thread: path trace / sample / accumulate
  ↓
read output_buffer → stbi.write_png   kernel writes float4 → output_buffer
```

## Shader Design (raytrace.metal)

- **Intersection function** `[[intersection(bounding_box)]]`:
  Receives ray + sphere primitive index, does ray-sphere test, returns hit
  distance (or miss).
- **Kernel** `[[kernel]] raytraceKernel`:
  One thread per pixel. Each thread runs all samples. Per sample: generate ray
  → trace through AS → hit data → compute normal → material scatter → recurse
  (loop) → accumulate. GPU RNG: per-thread `uint` state, PCG-style.

## Precision

- CPU scene generation: `f64` unchanged
- GPU buffers: `f32` (spheres, camera, constants)
- Shader: `float` (f32) — 2× faster than f64, sufficient for path tracing

## Performance Target

1024×576, 50spp: ~1–2s (vs ~22s CPU)
