"""lumbre.render: file round trip, frames, camera filter, errors.

    ./lumbre --script python/tests/test_render.py -- <out_dir>

Writes <out_dir>/script/sc.* from semantic_classes.usda. scripts/test_scripting.sh
renders the same file with `--raster --labels` into <out_dir>/raster/ and
requires the two to be byte-identical: the same stage, reached through Python
and the stage cache or through a path, must produce the same pixels and labels.
"""

import json
import os
import sys

import lumbre
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = sys.argv[1]
failures = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


# 1. The file, for the runner's byte comparison. Resolution matches the runner's.
st = Usd.Stage.Open(os.path.join(REPO, "assets", "usd_scene_tests", "semantic_classes.usda"))
files = lumbre.render(st, f"{out_dir}/script/sc.png", labels=True, width=320, height=240)
check("file stage writes beauty, label EXR and COCO",
      [os.path.basename(f) for f in files] == ["sc.png", "sc.labels.exr", "sc.coco.json"])

# 2. An in-memory stage, edited and rendered again each frame.
mem = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(mem, UsdGeom.Tokens.y)
ball = UsdGeom.Sphere.Define(mem, "/World/ball")
ball.GetPrim().CreateAttribute("semantic:class", Sdf.ValueTypeNames.String, custom=True).Set("ball")
UsdLux.DistantLight.Define(mem, "/World/sun")
for name, z in (("near", 6.0), ("far", 12.0)):
    cam = UsdGeom.Camera.Define(mem, f"/World/{name}")
    UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(0, 0, z))

lefts = []
image_ids = set()
for frame in range(3):
    UsdGeom.XformCommonAPI(ball).SetTranslate(Gf.Vec3d(frame - 1.0, 0, 0))
    out = lumbre.render(mem, f"{out_dir}/mem/mem.png", frame=frame, labels=True,
                        width=160, height=120, camera="near")
    names = [os.path.basename(f) for f in out]
    check(f"frame {frame}: numbered names, filtered camera left out",
          names == [f"mem.{frame:04d}.png", f"mem.{frame:04d}.labels.exr", f"mem.{frame:04d}.coco.json"])
    with open(out[2]) as f:
        coco = json.load(f)
    check(f"frame {frame}: COCO image id is unique", coco["images"][0]["id"] not in image_ids)
    image_ids.add(coco["images"][0]["id"])
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    boxes = [a["bbox"] for a in coco["annotations"] if cats.get(a["category_id"]) == "ball"]
    check(f"frame {frame}: the ball is annotated once", len(boxes) == 1)
    lefts.append(boxes[0][0] + boxes[0][2] / 2 if boxes else None)

check("the ball's box moves right as the ball does",
      None not in lefts and lefts[0] < lefts[1] < lefts[2])

# The script's prim handles must survive rendering: the stage stays cached.
check("prim handles stay valid after render", ball.GetPrim().IsValid())

# 3. Errors raise with a reason.
for bad, needle in ((dict(camera="nope"), "nothing rendered"), (dict(width=0), "bad resolution")):
    try:
        lumbre.render(mem, f"{out_dir}/bad/bad.png", **bad)
        check(f"{bad} raises", False)
    except RuntimeError as e:
        check(f"{bad} raises ({e})", needle in str(e))

if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
