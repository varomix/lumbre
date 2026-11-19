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
    {
        "id": "area-lights",
        "title": "Area lights (LTC)",
        "commit": "a0f8a68",
        "summary": (
            "Quad, disc, sphere and cylinder lights are integrated over their "
            "shapes with linearly transformed cosines fitted to Lumbre's own "
            "GGX. Before, each was a single point carrying its area."
        ),
        "cases": [
            {
                "id": "shapes",
                "title": "One light of each shape over a glossy floor",
                "scene": "assets/render_tests/area_lights.usda",
                "look_for": (
                    "Before, the floor shows hot pinpoint reflections and the matte "
                    "objects are over-lit up close. After, reflections should spread "
                    "into soft pools shaped by each light, and brightness should track "
                    "the path tracer. The panel (left) and tube (front) are close. The "
                    "orb's reflection on the floor reads brighter in realtime, and the "
                    "near floor darker, since realtime has no bounce light. When a0f8a68 landed, the path tracer drew the "
                    "tube twice as bright and both renderers shifted it right; 3a5b8a8 "
                    "fixed both. See the next step."
                ),
                "variants": [
                    {"id": "before", "label": "Before (a6d16b1)", "args": ["--raster"], "rev": "a6d16b1"},
                    {"id": "after", "label": "After", "args": ["--raster"]},
                    {"id": "reference", "label": "Path traced, 512 spp", "args": ["--spp", "512"]},
                ],
                "compare": [["before", "after"], ["after", "reference"], ["before", "reference"]],
            },
        ],
    },
    {
        "id": "cylinder-fix",
        "title": "Cylinder lights: facing half, centred",
        "commit": "3a5b8a8",
        "summary": (
            "The path tracer lit from the whole surface of a cylinder light, seeing "
            "its far side through the near one, which doubled its light. The USD "
            "importer also placed the tube by its end instead of its centre. Both "
            "are fixed, in the CPU and GPU path tracers and in the import both "
            "renderers share."
        ),
        "cases": [
            {
                "id": "tube",
                "title": "The area light scene's tube, path traced",
                "scene": "assets/render_tests/area_lights.usda",
                "look_for": (
                    "Before, the tube's light on the ball and floor is about twice as "
                    "bright, and its pool sits to the right. After, it is centred under "
                    "the tube and roughly halved, and the realtime render matches it. "
                    "The rect, disk and sphere lights should not change at all."
                ),
                "variants": [
                    {"id": "before", "label": "Path traced before (a0f8a68)", "args": ["--spp", "512"], "rev": "a0f8a68"},
                    {"id": "after", "label": "Path traced after", "args": ["--spp", "512"]},
                    {"id": "realtime", "label": "Realtime after", "args": ["--raster"]},
                ],
                "compare": [["before", "after"], ["realtime", "after"]],
            },
        ],
    },
    {
        "id": "baked-gi",
        "title": "Baked GI (irradiance probes)",
        "commit": "73574be",
        "summary": (
            "lumbre --bake-probes traces a grid of probes with the CPU path tracer "
            "into a <scene>.probes sidecar: per probe, octahedral maps of "
            "cosine-convolved irradiance and of depth moments. The realtime "
            "lighting pass takes its diffuse ambient from the eight probes around "
            "each point, weighted for visibility, instead of from the "
            "environment alone, and reflects their radiance where screen-space "
            "reflections find nothing on screen."
        ),
        "cases": [
            {
                "id": "cornell",
                "title": "Cornell box",
                "scene": "assets/cornell_box.obj",
                "bake": ["--grid", "8", "--probe-rays", "4096"],
                "look_for": (
                    "Environment-only lighting cannot see the ceiling panel, an "
                    "emissive mesh: the room is lit flat and blue-grey by the sky "
                    "through the open front. With probes the panel's light bounces "
                    "around the box: the walls warm up, red and green bleed onto the "
                    "blocks' sides, and the floor darkens under and between them, "
                    "close to the path tracer. The red wall is a mirror. Where it should "
                    "show block sides the camera cannot see, screen-space reflections "
                    "find nothing; without probes that falls back to the sky (a pink "
                    "patch), with them to the probes' radiance (a blurred, dark red "
                    "block, as in the path tracer). Probes are "
                    "8x8x8, 4096 rays each, a 3 s bake."
                ),
                "variants": [
                    {"id": "env", "label": "Environment ambient (--probes 0)", "args": ["--raster", "--probes", "0"]},
                    {"id": "probes", "label": "Baked probes", "args": ["--raster"]},
                    {"id": "reference", "label": "Path traced, deterministic, 256 spp", "args": ["--deterministic", "--spp", "256"]},
                ],
                "compare": [["env", "probes"], ["probes", "reference"], ["env", "reference"]],
            },
            {
                "id": "room",
                "title": "Room lit through its open side, crates from a PointInstancer",
                "scene": "assets/render_tests/baked_gi_room.usda",
                "bake": ["--grid", "8", "--probe-rays", "4096"],
                "look_for": (
                    "The sun lands on the front of the floor; the rest of the room "
                    "is lit by what that patch and the sky bounce around. Without "
                    "probes every wall gets the whole sky and the room reads evenly "
                    "lit to the back. With probes it falls off toward the back wall, "
                    "the ceiling picks up light from the sunlit floor, the side walls "
                    "tint what faces them, and the crates sit in their own occlusion. "
                    "Built by python/examples/baked_gi.py."
                ),
                "variants": [
                    {"id": "env", "label": "Environment ambient (--probes 0)", "args": ["--raster", "--probes", "0"]},
                    {"id": "probes", "label": "Baked probes", "args": ["--raster"]},
                    {"id": "reference", "label": "Path traced, deterministic, 256 spp", "args": ["--deterministic", "--spp", "256"]},
                ],
                "compare": [["env", "probes"], ["probes", "reference"], ["env", "reference"]],
            },
        ],
    },
    {
        "id": "probes-fog-glass",
        "title": "Baked probes in fog and glass",
        "commit": "after 7607e59",
        "summary": (
            "Fog scatters the probes' mean radiance at each step of its march, "
            "trilinear between the eight around it, instead of the sky's. The "
            "forward pass gives transmissive surfaces the same probe lookup as "
            "opaque ones; a clear pane has almost no diffuse lobe, so that shows "
            "on frosted or tinted ones."
        ),
        "cases": [
            {
                "id": "room-fog",
                "title": "Fog in the room lit through its open side",
                "scene": "assets/render_tests/baked_gi_room.usda",
                "bake": ["--grid", "8", "--probe-rays", "4096"],
                "look_for": (
                    "Without probes the fog inside the room glows blue-white, lit "
                    "by the whole sky as if there were no roof. With them it takes "
                    "the room's own light: dimmer and warmer toward the back, "
                    "tinted by the red and blue walls near them."
                ),
                "variants": [
                    {"id": "env", "label": "Fog, environment ambient (--probes 0)", "args": ["--raster", "--fog", "1", "--probes", "0"]},
                    {"id": "probes", "label": "Fog, baked probes", "args": ["--raster", "--fog", "1"]},
                    {"id": "clear", "label": "No fog, baked probes", "args": ["--raster"]},
                ],
                "compare": [["env", "probes"], ["clear", "probes"]],
            },
        ],
    },
    {
        "id": "area-shadows",
        "title": "Area light shadows (PCSS)",
        "commit": "after a234295",
        "summary": (
            "Quad, disc and cylinder lights now take one of the four omni shadow "
            "slots, as point and sphere lights did. Every area light's shadow is "
            "softened by its size: a blocker search finds how far the occluders "
            "are, and a PCF kernel takes the penumbra width that gives. Taps sit "
            "on the receiving surface, each projected to the face that sees it."
        ),
        "cases": [
            {
                "id": "stations",
                "title": "A panel over a box, a disk over a ball, a tube over a post",
                "scene": "assets/render_tests/area_shadows.usda",
                "look_for": (
                    "Before, only the floor's falloff: the area lights cast no "
                    "shadows. After, the box's shadow is sharp where it meets the "
                    "floor and widens away from it; the floating ball casts a soft "
                    "round shadow; the post's shadow smears out from its foot. The "
                    "ball's shadow core is a little lighter than the path tracer's, "
                    "PCSS's usual error for a round occluder. The box's sides and "
                    "the post read dark in realtime: they are lit mostly by the "
                    "floor's bounce, which needs baked probes."
                ),
                "variants": [
                    {"id": "before", "label": "Before (a234295)", "args": ["--raster"], "rev": "a234295"},
                    {"id": "after", "label": "After", "args": ["--raster"]},
                    {"id": "fog", "label": "After, fog on", "args": ["--raster", "--fog", "1", "--fog-density", "1"]},
                    {"id": "reference", "label": "Path traced, deterministic, 256 spp", "args": ["--deterministic", "--spp", "256"]},
                ],
                "compare": [["before", "after"], ["after", "reference"]],
            },
        ],
    },
    {
        "id": "area-shadow-shape",
        "title": "Area light shadows shaped like the light",
        "commit": "after c2132c7",
        "summary": (
            "Two fixes from hands-on checks in the GUI. The soft shadow's kernel "
            "now has the light's own shape: a tube's is long and thin, a quad's "
            "follows its edges. Before, every area light was filtered as a disc "
            "of the same area, so a 1.6 m tube shadowed like a 0.2 m ball. And "
            "the tap spiral now turns per pixel and per TAA frame. Before, it "
            "turned per shadow-map texel and stayed put, so TAA had nothing to "
            "average, and close up the grain read as blocky."
        ),
        "cases": [
            {
                "id": "post",
                "title": "Close up on the tube over the post",
                "scene": "assets/render_tests/area_shadow_post.usda",
                "look_for": (
                    "Before, a short, dark, grainy trapezoid behind the post, "
                    "ending in a hard line. After, a wide, faint fan from the "
                    "post's foot, like the path tracer's, and smooth with TAA. "
                    "Without TAA the grain is fine dither, not blocks. Right at "
                    "the foot the path tracer is a little darker: PCSS's usual "
                    "miss for an occluder touching the receiver."
                ),
                "variants": [
                    {"id": "before", "label": "Before (c2132c7), TAA", "args": ["--raster", "--taa", "1"], "rev": "c2132c7"},
                    {"id": "after-fxaa", "label": "After, no TAA", "args": ["--raster"]},
                    {"id": "after", "label": "After, TAA", "args": ["--raster", "--taa", "1"]},
                    {"id": "reference", "label": "Path traced, deterministic, 256 spp", "args": ["--deterministic", "--spp", "256"]},
                ],
                "compare": [["before", "after"], ["after", "reference"]],
            },
            {
                "id": "stations",
                "title": "All three stations",
                "scene": "assets/render_tests/area_shadows.usda",
                "look_for": (
                    "The post's shadow is now a faint fan, as in the path tracer, "
                    "where it was a dark streak. The box's and the ball's are "
                    "much as before: a square panel and a disk are close to the "
                    "disc they were filtered as."
                ),
                "variants": [
                    {"id": "before", "label": "Before (c2132c7)", "args": ["--raster"], "rev": "c2132c7"},
                    {"id": "after", "label": "After", "args": ["--raster"]},
                    {"id": "reference", "label": "Path traced, deterministic, 256 spp", "args": ["--deterministic", "--spp", "256"]},
                ],
                "compare": [["before", "after"], ["after", "reference"]],
            },
        ],
    },
    {
        "id": "glossy-env",
        "title": "Environment reflections as wide as the BRDF's",
        "commit": "after 12db412",
        "summary": (
            "The prefiltered environment and the BRDF LUT took alpha = "
            "roughness^2, the analytic BRDF and the path tracer alpha = roughness, "
            "so environment reflections were much sharper than a light's highlight "
            "on the same surface. Both now use the path tracer's GGX, with its "
            "exact Smith term in the LUT. The prefilter picks a source mip per tap "
            "from the lobe's width, and the diffuse map is an exact sum over the "
            "environment instead of 256 sampled directions."
        ),
        "cases": [
            {
                "id": "sweep",
                "title": "Metal and plastic at roughness 0 to 1 under an HDRI",
                "scene": "assets/render_tests/glossy_env.usda",
                "look_for": (
                    "Before, metal at roughness 0.2 and 0.4 is nearly a mirror, "
                    "rough spheres are blotchy with a pinch at the top, and the red "
                    "spheres carry dark stripes where they face the sun. After, "
                    "blur grows with roughness as in the path tracer, and the "
                    "stripes and pinch are gone. At roughness 0.2 the neighbours' "
                    "reflections stay sharp: those are SSR, which does not blur "
                    "yet. Rough metal still reads a little bright, the split sum's "
                    "view-equals-normal assumption. The path tracer's red spheres "
                    "show a sharp sunlit crescent: the metal spheres shadow the "
                    "HDRI's sun, which environment lighting cannot. The reference "
                    "turns off the path tracer's roughness cutoff, which renders "
                    "roughness 1 as diffuse."
                ),
                "variants": [
                    {"id": "before", "label": "Before (12db412)", "args": ["--raster"], "rev": "12db412"},
                    {"id": "after", "label": "After", "args": ["--raster"]},
                    {"id": "reference", "label": "Path traced, 256 spp, no roughness cutoff", "args": ["--spp", "256", "--roughness-cutoff", "2"]},
                ],
                "compare": [["before", "after"], ["after", "reference"], ["before", "reference"]],
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
            # A case may need probes baked first; its sidecar sits next to the
            # scene (gitignored) and is rebuilt every run, with this build.
            bake = None
            if "bake" in case:
                print(f"  {case['id']} / bake")
                start = time.perf_counter()
                run([str(current), "--bake-probes", case["scene"], *case["bake"]])
                bake = {"seconds": round(time.perf_counter() - start, 2),
                        "command": " ".join(["lumbre", "--bake-probes", case["scene"], *case["bake"]])}
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
                              "variants": variants_out, **({"bake": bake} if bake else {})})
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
