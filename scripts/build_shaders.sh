#!/usr/bin/env bash
# Compiles the realtime renderer's Slang shaders to every format SDL_GPU can
# consume, and commits the results.
#
# Shaders are built OFFLINE and the outputs are checked in, so a plain
# `odin build gui` needs no Slang installed — the Odin side `#load`s the blobs
# at compile time. Run this only when a .slang file changes.
#
#   scripts/build_shaders.sh
#
# One entry point per output file: Slang emits a whole translation unit per
# (entry, stage) pair, and SDL_GPU's shader object is per-stage anyway.
#
# MSL is emitted as source rather than a .metallib because SDL compiles it at
# device-creation time and that keeps the artifact readable in review. SPIR-V
# is what the Vulkan and D3D12 backends want.

set -euo pipefail

cd "$(dirname "$0")/.."

src_dir="realtime/shaders"
out_dir="$src_dir/build"

if ! command -v slangc >/dev/null 2>&1; then
  echo "slangc not found. Install the Vulkan SDK or the Slang release." >&2
  exit 1
fi

mkdir -p "$out_dir"

# name:entry:stage — every shader the renderer loads must be listed here.
shaders=(
  "triangle:vertexMain:vertex"
  "triangle:fragmentMain:fragment"
)

for spec in "${shaders[@]}"; do
  IFS=: read -r name entry stage <<<"$spec"
  src="$src_dir/$name.slang"

  echo "==> $name [$stage] $entry"
  slangc "$src" -target metal -entry "$entry" -stage "$stage" \
    -o "$out_dir/$name.$stage.msl"
  slangc "$src" -target spirv -entry "$entry" -stage "$stage" \
    -o "$out_dir/$name.$stage.spv"
done

echo "==> wrote $(ls "$out_dir" | wc -l | tr -d ' ') files to $out_dir"
