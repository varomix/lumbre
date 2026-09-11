"""lumbre.randomize: a seeded, randomized scene rendered for two frames.

    ./lumbre --script python/tests/test_randomize.py -- <out_dir> <seed>

Checks nothing byte-level itself; scripts/test_scripting.sh runs it twice with
one seed and once with another and compares, then renders the exported
<out_dir>/frame1.usda with `--raster --labels` against ds.0001.*. A randomized
stage is a plain USD stage, so saving it must not change a pixel.

It also carries a prim with no material bound, whose look must be the default
rather than whichever material came first (fixed in ce58b74).
"""

import json
import os
import random
import sys
import zlib

import lumbre
from lumbre import randomize as rnd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

out_dir, seed = sys.argv[1], int(sys.argv[2])
failures = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
UsdGeom.Xform.Define(stage, "/World")

floor = UsdGeom.Cube.Define(stage, "/World/floor")
UsdGeom.XformCommonAPI(floor).SetScale(Gf.Vec3f(8, 0.05, 8))
UsdGeom.XformCommonAPI(floor).SetTranslate(Gf.Vec3d(0, -0.05, 0))
rnd.set_semantic_class(floor, "floor")
# Distinctly blue, and the first material bound in traversal: under the old
# bug, the unbound cube below would have rendered blue too.
rnd.bind_preview_surface(floor, rnd.preview_surface(stage, "/World/Looks/floor", base_color=(0.1, 0.35, 0.9)))

target = UsdGeom.Sphere.Define(stage, "/World/target")
rnd.set_semantic_class(target, "target")
target_mat = rnd.preview_surface(stage, "/World/Looks/target")
rnd.bind_preview_surface(target, target_mat)

bare = UsdGeom.Cube.Define(stage, "/World/bare")
UsdGeom.XformCommonAPI(bare).SetTranslate(Gf.Vec3d(-2.5, 0.5, 0))
rnd.set_semantic_class(bare, "bare")

sun = UsdLux.DistantLight.Define(stage, "/World/sun")
camera = UsdGeom.Camera.Define(stage, "/World/camera")

for frame in range(2):
    rng = random.Random(seed * 1000 + frame)
    eye = rnd.orbit_camera(camera, target=(0, 0.5, 0), rng=rng, distance=(8, 10), elevation=(15, 40))
    check(f"frame {frame}: camera placed within the distance range",
          8 <= Gf.Vec3d(*eye).GetLength() <= 10.1)
    rnd.aim_light(sun, rng)
    rnd.randomize_light(sun, rng, intensity=(2, 5), color=((0.8, 0.8, 0.8), (1, 1, 1)))
    rnd.randomize_surface(target_mat, rng, metallic=(0, 1))
    rnd.random_pose(target, rng, translate=((-1, 1, -1), (1, 1, 1)))
    rnd.clear(stage, "/World/Distractors")
    rnd.scatter_distractors(stage, "/World/Distractors", rng, count=6,
                            bounds=((-3, 0.3, -3), (3, 0.3, 3)))
    kids = [p for p in stage.GetPrimAtPath("/World/Distractors").GetChildren() if p.GetName() != "Looks"]
    check(f"frame {frame}: clear + scatter leaves exactly 6 distractors", len(kids) == 6)
    check(f"frame {frame}: every distractor binds a material",
          all(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0] for p in kids))

    rnd.clear(stage, "/World/Scatter")
    scatter = rnd.scatter_instances(stage, "/World/Scatter", rng, count=40, prototypes=3,
                                    bounds=((-3.5, 0.1, -3.5), (3.5, 0.1, 3.5)), size=(0.15, 0.3),
                                    semantic_class="scatter")
    check(f"frame {frame}: scatter_instances authors 40 points over 3 prototypes",
          len(scatter.GetProtoIndicesAttr().Get()) == 40 and len(scatter.GetPrototypesRel().GetTargets()) == 3)

    files = lumbre.render(stage, f"{out_dir}/ds.png", frame=frame, labels=True, width=320, height=240)
    with open(files[2]) as f:
        coco = json.load(f)
    names = {c["name"] for c in coco["categories"]}
    check(f"frame {frame}: classes include target, floor, bare, distractor, scatter",
          {"target", "floor", "bare", "distractor", "scatter"} <= names)

    # Every visible point is its own object, traceable to its point index.
    scatter_id = next(c["id"] for c in coco["categories"] if c["name"] == "scatter")
    points = [a.get("prim_path", "") for a in coco["annotations"] if a["category_id"] == scatter_id]
    indices = [p[len("/World/Scatter["):p.index("]")] for p in points if p.startswith("/World/Scatter[")]
    check(f"frame {frame}: {len(points)} visible scatter points, each labelled with its own point path",
          0 < len(points) <= 40 and len(indices) == len(points) and len(set(indices)) == len(indices)
          and all(0 <= int(i) < 40 for i in indices))

stage.GetRootLayer().Export(f"{out_dir}/frame1.usda")



def read_png_rgba(path):
    """Decodes the 8-bit RGBA, non-interlaced PNGs the raster runner writes.
    The vendored stdlib has zlib but no image library."""
    with open(path, "rb") as f:
        data = f.read()
    pos, idat = 8, b""
    while pos < len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height = int.from_bytes(body[0:4], "big"), int.from_bytes(body[4:8], "big")
            assert body[8] == 8 and body[9] == 6 and body[12] == 0, "expected RGBA8, no interlace"
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw, bpp, stride = zlib.decompress(idat), 4, width * 4
    rows, prev, i = [], bytearray(stride), 0
    for _ in range(height):
        filt, line = raw[i], bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b, c = prev[x], (prev[x - bpp] if x >= bpp else 0)
            if filt == 1: line[x] = (line[x] + a) & 255
            elif filt == 2: line[x] = (line[x] + b) & 255
            elif filt == 3: line[x] = (line[x] + (a + b) // 2) & 255
            elif filt == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    return width, rows


# The unbound cube must wear the default grey, not the first material bound in
# the scene (the blue floor). Its box also holds floor and maybe a distractor,
# so require a healthy share of grey pixels rather than sampling one.
with open(f"{out_dir}/ds.0001.coco.json") as f:
    coco = json.load(f)
cats = {c["id"]: c["name"] for c in coco["categories"]}
bare_boxes = [a["bbox"] for a in coco["annotations"] if cats[a["category_id"]] == "bare"]
check("the unbound cube is visible in frame 1", len(bare_boxes) == 1)
if bare_boxes:
    x0, y0, w, h = bare_boxes[0]
    _, rows = read_png_rgba(f"{out_dir}/ds.0001.png")
    grey = total = 0
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            r, g, b = rows[y][x * 4], rows[y][x * 4 + 1], rows[y][x * 4 + 2]
            total += 1
            grey += max(r, g, b) - min(r, g, b) < 40 and max(r, g, b) > 60
    share = grey / max(total, 1)
    print(f"     grey share inside the unbound cube's box: {share:.2f}")
    check("the unbound cube renders in the default grey, not the first material", share >= 0.3)

if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
