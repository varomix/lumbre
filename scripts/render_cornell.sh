#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

width=320
height=240
spp=16
depth=8
mode="quick"
renderer="gpu"
skip_build=0
tag="$(date +%Y%m%d_%H%M%S)"
out_dir="dev_renders/iter_$tag"

usage() {
  cat <<'USAGE'
Usage: scripts/render_cornell.sh [options]

Options:
  --quick              Render core debug set plus beauty (default)
  --full               Render all current debug modes plus beauty
  --cpu                Force CPU renderer for beauty only
  --gpu                Force GPU renderer (default)
  --skip-build         Reuse existing ./lumbre binary
  --width N            Image width (default 320)
  --height N           Image height (default 240)
  --spp N              Samples per pixel for beauty/direct (default 16)
  --depth N            Max depth (default 8)
  --tag NAME           Output folder suffix (default timestamp)
  --out-dir DIR        Output directory (default dev_renders/iter_<tag>)
  -h, --help           Show this help

Outputs are written to the chosen output directory with stable names:
  01_albedo.png
  02_primitive_id.png
  03_light_count.png
  04_direct_candidates.png
  05_shadow_visibility.png
  06_direct.png
  07_beauty.png
  08_normal.png (--full)
  09_depth.png (--full)
  10_indirect.png (--full)
  11_gi_cache_hits.png (--full)
  12_photon_contribution.png (--full)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) mode="quick"; shift ;;
    --full) mode="full"; shift ;;
    --cpu) renderer="cpu"; shift ;;
    --gpu) renderer="gpu"; shift ;;
    --skip-build) skip_build=1; shift ;;
    --width) width="$2"; shift 2 ;;
    --height) height="$2"; shift 2 ;;
    --spp) spp="$2"; shift 2 ;;
    --depth) depth="$2"; shift 2 ;;
    --tag)
      tag="$2"
      out_dir="dev_renders/iter_$tag"
      shift 2
      ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$out_dir"

if [[ "$skip_build" -eq 0 ]]; then
  echo "==> Building ./lumbre"
  odin build . -out:lumbre
fi

common=(--scene assets/cornell_box.obj --width "$width" --height "$height" --depth "$depth")
render_flag=(--gpu)
if [[ "$renderer" == "cpu" ]]; then
  render_flag=(--cpu)
fi

run_render() {
  local name="$1"
  local debug="$2"
  local samples="$3"
  local output="$out_dir/$name.png"

  echo "==> $name -> $output"
  if [[ "$debug" == "beauty" ]]; then
    ./lumbre "${render_flag[@]}" "${common[@]}" --spp "$samples" --output "$output"
  else
    ./lumbre "${render_flag[@]}" "${common[@]}" --spp "$samples" --debug "$debug" --output "$output"
  fi
}

if [[ "$renderer" == "cpu" ]]; then
  run_render "07_beauty_cpu" "beauty" "$spp"
else
  run_render "01_albedo" 1 1
  run_render "02_primitive_id" 4 1
  run_render "03_light_count" 6 1
  run_render "04_direct_candidates" 7 1
  run_render "05_shadow_visibility" 8 1
  run_render "06_direct" 5 "$spp"

  if [[ "$mode" == "full" ]]; then
    run_render "08_normal" 2 1
    run_render "09_depth" 3 1
    run_render "10_indirect" 9 "$spp"
    run_render "11_gi_cache_hits" 10 "$spp"
    run_render "12_photon_contribution" 11 "$spp"
  fi

  run_render "07_beauty" "beauty" "$spp"
fi

echo
echo "Done. Outputs:"
find "$out_dir" -maxdepth 1 -type f -name '*.png' | sort
