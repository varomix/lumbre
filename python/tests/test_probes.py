"""lumbre.bake_probes and probe use in lumbre.render.

    ./lumbre --script python/tests/test_probes.py -- <out_dir>

Bakes the baked_gi example's room (small grid, few rays: this checks plumbing
and determinism, not image quality) and renders it with and without probes.
"""

import json
import os
import sys

import lumbre
from pxr import Usd, UsdLux

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python", "examples"))
import baked_gi  # noqa: E402

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
failures = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


def annotations(path):
    """COCO annotations without their ids, which hash the output file name."""
    with open(path) as f:
        coco = json.load(f)
    return [{k: v for k, v in a.items() if k not in ("id", "image_id")} for a in coco["annotations"]]


def read(path):
    with open(path, "rb") as f:
        return f.read()


stage = baked_gi.build_stage(os.path.join(out_dir, "room.usda"))
sidecar = os.path.join(out_dir, "room.probes")

# 1. The default output is the sidecar; an explicit one is the same bake.
a = lumbre.bake_probes(stage, grid=4, rays=128)
check("default output is the sidecar next to the stage", a["file"] == sidecar and os.path.exists(sidecar))
# 6.2 x 3.2 x 6.1 with the walls: 4 along x, the others in proportion, rounded up.
check("the grid follows the room's proportions", a["dims"] == [4, 3, 4])
b = lumbre.bake_probes(stage, output=os.path.join(out_dir, "again.probes"), grid=4, rays=128)
check("two bakes are byte-identical", read(a["file"]) == read(b["file"]))
check("both bakes carry the same revision", a["revision"] == b["revision"])

# 2. Rendering picks the sidecar up; probes=False leaves it out. Labels never
# depend on lighting.
gi = lumbre.render(stage, os.path.join(out_dir, "gi.png"), labels=True, width=160, height=90)
env = lumbre.render(stage, os.path.join(out_dir, "env.png"), labels=True, width=160, height=90, probes=False)
check("probes change the beauty image", read(gi[0]) != read(env[0]))
check("probes leave the label EXR alone", read(gi[1]) == read(env[1]))
check("probes leave the COCO annotations alone", annotations(gi[2]) == annotations(env[2]))
explicit = lumbre.render(stage, os.path.join(out_dir, "explicit.png"), width=160, height=90,
                         probes=os.path.join(out_dir, "again.probes"))
check("an explicit probe file renders like the sidecar", read(explicit[0]) == read(gi[0]))

# 3. Edit the light: the bake no longer applies, and rendering falls back to the
# environment -- exactly what probes=False gives.
UsdLux.DistantLight(stage.GetPrimAtPath("/World/Sun")).GetIntensityAttr().Set(2.5)
stale = lumbre.render(stage, os.path.join(out_dir, "stale.png"), width=160, height=90)
off = lumbre.render(stage, os.path.join(out_dir, "off.png"), width=160, height=90, probes=False)
check("stale probes fall back to environment lighting", read(stale[0]) == read(off[0]))

# 4. An in-memory stage has nowhere to put a sidecar.
mem = Usd.Stage.CreateInMemory()
mem.DefinePrim("/World/Ball", "Sphere")
try:
    lumbre.bake_probes(mem, grid=2, rays=16)
    check("an in-memory stage without output= is refused", False)
except RuntimeError as e:
    check("an in-memory stage without output= is refused", "output=" in str(e))
c = lumbre.bake_probes(mem, output=os.path.join(out_dir, "mem.probes"), grid=2, rays=16)
check("an in-memory stage bakes to an explicit output", os.path.exists(c["file"]))

if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
