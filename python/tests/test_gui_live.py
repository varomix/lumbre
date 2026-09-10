"""lumbre.stage / lumbre.show in the GUI's script host.

    ./lumbre-gui --scene assets/usd_scene_tests/semantic_classes.usda \\
                 --run-script python/tests/test_gui_live.py

Run from a scratch working directory: it checks no look sidecar is written into
the working directory for a stage that has no file. Opens a window briefly.
--run-script exits before the event loop, so this checks app state and the USD
panels, not the viewport's pixels.
"""

import os

import lumbre
from lumbre import randomize as rnd
from pxr import Gf, Usd, UsdGeom

failures = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


check("host is the GUI", lumbre.host() == "gui")
check("a loaded file reports source=file", lumbre.stats()["source"] == "file")
n_prims = len(lumbre.prims())

st = lumbre.stage()
check("stage() returns the same object twice", lumbre.stage() is st)
check("stage() is the loaded file", st.GetRootLayer().realPath.endswith("semantic_classes.usda"))

ball = UsdGeom.Sphere.Define(st, "/World/added")
UsdGeom.XformCommonAPI(ball).SetTranslate(Gf.Vec3d(0, 3, 0))
rnd.set_semantic_class(ball, "added")
lumbre.show(st)

info = lumbre.stats()
check("after show: source=script", info["source"] == "script")
check("after show: the file path is kept", info["scene"].endswith("semantic_classes.usda"))
paths = [p["path"] for p in lumbre.prims()]
check("the USD panel lists the unsaved prim", "/World/added" in paths)
check("the prim count grew by exactly one", len(paths) == n_prims + 1)
check("stage() after show is the shown stage", lumbre.stage() is st)
with open(st.GetRootLayer().realPath) as f:
    check("the file on disk is untouched", "added" not in f.read())

mem = Usd.Stage.CreateInMemory()
UsdGeom.Cube.Define(mem, "/Box")
lumbre.show(mem)
check("an in-memory stage shows with no scene path", lumbre.stats()["scene"] == "")
check("the USD panel lists only /Box", [p["path"] for p in lumbre.prims()] == ["/Box"])
check("save_look refuses a stage with no file", lumbre.save_look()["ok"] is False)
check("no stray look file in the working directory", not os.path.exists(".lumbrelook"))

try:
    lumbre.render(mem, "never.png")
    check("render() is CLI-only", False)
except RuntimeError as e:
    check("render() raises in the GUI, by name", "'render'" in str(e))

if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
