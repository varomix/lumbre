#!/usr/bin/env python3
"""Renders the visual test cases behind docs/render-tests/index.html.

Each realtime milestone gets a STEP below: a few cases, each one scene rendered
in several VARIANTS -- a feature off and on, the build before the change, the
path tracer as reference. The script renders every variant, records its
wall time and (for the rasterizer) the per-pass GPU times, and writes
docs/render-tests/data.js, which the page reads. Open the page straight from
disk; it needs no server.

    python3 scripts/render_tests.py              # everything, current ./lumbre
    python3 scripts/render_tests.py --build      # rebuild ./lumbre first
    python3 scripts/render_tests.py --step taa   # only one step

A variant with `rev` renders with that commit's build instead, made once in a
git worktree under .render_tests/ and kept there. Use it for "before" images
of a change that has no switch.

To add a step: append to STEPS, run the script, commit the images and data.js
with the change they document.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "render-tests"
CACHE = ROOT / ".render_tests"

W, H = 800, 450

STEPS = [
    {
        "id": "taa",
        "title": "Temporal antialiasing",
        "commit": "9fd58ae",
        "summary": (
            "The shaded view jitters the camera through 16 Halton points and "
            "blends each frame into an HDR history, reprojected from depth and "
            "clipped to the current frame's neighbourhood. Replaces FXAA while on. "
            "Headless stills render 16 frames from a cleared history."
        ),
        "cases": [
            {
                "id": "edges",
                "title": "Thin geometry and shadow edges",
                "scene": "assets/render_tests/taa_edges.usda",
                "look_for": (
                    "Rails and posts should be continuous lines with no stair-steps or "
                    "gaps. The sphere's silhouette should be smooth, and so should the "
                    "fence shadows on the floor. FXAA softens edges but leaves broken "
                    "thin lines. TAA should rebuild them without blurring flat areas."
                ),
                "variants": [
                    {"id": "none", "label": "No antialiasing", "args": ["--raster", "--fxaa", "0"]},
                    {"id": "fxaa", "label": "FXAA (headless default)", "args": ["--raster"]},
                    {"id": "taa", "label": "TAA", "args": ["--raster", "--taa", "1"]},
                ],
                "compare": [["fxaa", "taa"], ["none", "taa"]],
            },
        ],
    },
    {
        "id": "spot-shadows",
        "title": "Spot light shadows",
        "commit": "a158a95",
        "summary": (
            "The first eight spot lights each get a 1024² perspective depth map "
            "fitted to the cone. Lighting, glass and fog sample it. A map is "
            "redrawn only when its light or the geometry changes."
        ),
        "cases": [
            {
                "id": "spot",
                "title": "One spot over a box and a ball",
                "scene": "assets/render_tests/spot_shadow.usda",
                "look_for": (
                    "Before: the spot lights straight through the box and ball. After: "
                    "both cast shadows, and the box's shadow should match the path "
                    "tracer's in shape and position. Realtime shadows are darker because "
                    "there is no bounce light yet. Expect no acne on the floor and no "
                    "light leaking at the base of either object."
                ),
                "variants": [
                    {"id": "before", "label": "Before (9fd58ae)", "args": ["--raster"], "rev": "9fd58ae"},
                    {"id": "after", "label": "After", "args": ["--raster"]},
                    {"id": "fog", "label": "After, fog on", "args": ["--raster", "--fog", "1", "--fog-density", "1.5"]},
                    {"id": "reference", "label": "Path traced, 256 spp", "args": ["--spp", "256"]},
                ],
                "compare": [["before", "after"], ["after", "reference"]],
            },
        ],
    },
    {
        "id": "point-shadows",
        "title": "Point and sphere light shadows",
        "commit": "f24c459",
        "summary": (
            "The first four point or sphere lights each get six 512² faces, stored "
            "as layers of a depth array. The shader picks the face by the dominant "
            "axis from the light. Spot and point maps share drawing and caching."
        ),
        "cases": [
            {
                "id": "room",
                "title": "One bulb low in a room, objects on every side",
                "scene": "assets/render_tests/point_shadow.usda",
                "look_for": (
                    "Shadows should fall in every direction from the bulb: the left box's "
                    "up the left wall, the far ball's onto the back wall, the pillar's "
                    "across the floor. Shapes and positions should match the path tracer. "
                    "The reference is softer because its bulb has a 5 cm radius, and "
                    "lighter because of bounce light. Check for seams or gaps where a "
                    "shadow crosses from one face to the next."
                ),
                "variants": [
                    {"id": "before", "label": "Before (94079a3)", "args": ["--raster"], "rev": "94079a3"},
                    {"id": "after", "label": "After", "args": ["--raster"]},
                    {"id": "fog", "label": "After, fog on", "args": ["--raster", "--fog", "1", "--fog-density", "1"]},
                    {"id": "reference", "label": "Path traced, 256 spp", "args": ["--spp", "256"]},
                ],
                "compare": [["before", "after"], ["after", "reference"]],
            },
        ],
    },
]

PASS_LINE = re.compile(r"^\s+(\w+)\s+([0-9.]+)\s*$")


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True, **kw)


def binary_for(rev, current):
    """The build to render with: the working tree's, or `rev`'s, built once."""
    if rev is None:
        return current
    target = CACHE / "bin" / rev
    exe = target / "lumbre"
    if exe.exists():
        return exe
    print(f"  building {rev} (once; cached in {target.relative_to(ROOT)})")
    tree = CACHE / "src" / rev
    if not tree.exists():
        run(["git", "worktree", "add", "--detach", str(tree), rev])
    target.mkdir(parents=True, exist_ok=True)
    # The binary finds its libraries at @loader_path/lib, so the cache
    # directory borrows the checkout's.
    link = target / "lib"
    if not link.exists():
        link.symlink_to(ROOT / "lib")
    subprocess.run(["odin", "build", ".", f"-out:{exe}"], cwd=tree, check=True)
    return exe


def render(exe, scene, args, out):
    env = dict(os.environ, LUMBRE_RASTER_PROFILE="1")
    cmd = [str(exe), "-s", scene, "-w", str(W), "-h", str(H), "-o", str(out), *args]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    seconds = time.perf_counter() - start
    if proc.returncode != 0 or not out.exists():
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"render failed: {' '.join(cmd)}")
    passes = []
    frames = 0
    in_table = False
    for line in proc.stdout.splitlines():
        if "GPU pass times" in line:
            in_table = True
            m = re.search(r"over (\d+) frame", line)
            frames = int(m.group(1)) if m else 0
            continue
        if in_table:
            m = PASS_LINE.match(line)
            if not m:
                in_table = False
                continue
            passes.append([m.group(1), float(m.group(2))])
    shown = ["lumbre", "-s", scene, "-w", str(W), "-h", str(H), *args]
    return {"seconds": round(seconds, 2), "passes": passes, "frames": frames, "command": " ".join(shown)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="rebuild ./lumbre from the working tree first")
    parser.add_argument("--binary", default=str(ROOT / "lumbre"), help="build to render the current tree with")
    parser.add_argument("--step", action="append", help="only these step ids (repeatable)")
    opts = parser.parse_args()

    current = pathlib.Path(opts.binary)
    if opts.build:
        print("building ./lumbre")
        subprocess.run(["odin", "build", ".", f"-out:{current}"], cwd=ROOT, check=True)
    if not current.exists():
        raise SystemExit(f"{current} not found; pass --build")

    images = OUT / "images"
    images.mkdir(parents=True, exist_ok=True)
    data_path = OUT / "data.js"

    # Keep results of steps not re-rendered this run.
    previous = {}
    if data_path.exists():
        text = data_path.read_text()
        payload = text[text.index("{"): text.rindex("}") + 1]
        previous = {s["id"]: s for s in json.loads(payload)["steps"]}

    head = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    steps_out = []
    for step in STEPS:
        if opts.step and step["id"] not in opts.step:
            if step["id"] in previous:
                steps_out.append(previous[step["id"]])
            continue
        print(f"{step['title']}")
        cases_out = []
        for case in step["cases"]:
            variants_out = []
            for v in case["variants"]:
                exe = binary_for(v.get("rev"), current)
                name = f"{step['id']}--{case['id']}--{v['id']}.png"
                print(f"  {case['id']} / {v['id']}")
                result = render(exe, case["scene"], v["args"], images / name)
                variants_out.append({
                    "id": v["id"], "label": v["label"], "image": f"images/{name}",
                    "rev": v.get("rev") or head, **result,
                })
            cases_out.append({**{k: case[k] for k in ("id", "title", "scene", "look_for", "compare")},
                              "variants": variants_out})
        steps_out.append({**{k: step[k] for k in ("id", "title", "commit", "summary")},
                          "rendered_at": time.strftime("%Y-%m-%d %H:%M"), "rendered_from": head,
                          "cases": cases_out})

    data = {"width": W, "height": H, "steps": steps_out}
    data_path.write_text(
        "// Generated by scripts/render_tests.py -- edit STEPS there, not this file.\n"
        f"window.RENDER_TESTS = {json.dumps(data, indent=1)};\n"
    )
    print(f"wrote {data_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
