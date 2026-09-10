"""A small randomized, labelled dataset from an existing USD asset.

    ./lumbre --script python/examples/random_dataset.py -- [out_prefix] [frames] [seed]

Defaults: out/ds, 10 frames, seed 0. Writes, per frame, a beauty PNG, a label
EXR (instance, semantic, depth, normal) and a COCO file.

The stage is built in memory and *references* semantic_classes.usda rather
than copying it, then overrides what it needs in its own root layer — the
asset on disk is never touched. That is ordinary USD composition, and it is
what a real pipeline does with a library of assets.
"""

import os
import random
import sys

import lumbre
from lumbre import randomize as rnd
from pxr import Gf, Usd, UsdGeom

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET = os.path.join(REPO, "assets", "usd_scene_tests", "semantic_classes.usda")

out_prefix = sys.argv[1] if len(sys.argv) > 1 else "out/ds"
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

# The asset's /World (its defaultPrim) composes in under our own /World.
world = stage.DefinePrim("/World", "Xform")
world.GetReferences().AddReference(ASSET)
stage.SetDefaultPrim(world)

# The asset leaves its sphere unlabelled; this dataset calls it a ball. The
# opinion lives in this stage's layer and wins over the referenced one.
rnd.set_semantic_class(stage.GetPrimAtPath("/World/unlabelled"), "ball")

floor = UsdGeom.Cube.Define(stage, "/World/Floor")
UsdGeom.XformCommonAPI(floor).SetScale(Gf.Vec3f(12, 0.05, 12))
UsdGeom.XformCommonAPI(floor).SetTranslate(Gf.Vec3d(0, -1.05, 0))
rnd.set_semantic_class(floor, "floor")
rnd.bind_preview_surface(floor, rnd.preview_surface(stage, "/World/Looks/Floor", base_color=(0.5, 0.5, 0.48)))

table_look = rnd.preview_surface(stage, "/World/Looks/Table")
rnd.bind_preview_surface(stage.GetPrimAtPath("/World/Furniture/table"), table_look)

camera = UsdGeom.Camera.Define(stage, "/World/DatasetCam")
camera.CreateFocalLengthAttr(28.0)
sun = stage.GetPrimAtPath("/World/sun")
chair = stage.GetPrimAtPath("/World/Furniture/chair")

total = 0
for frame in range(frames):
    # One generator per frame, seeded from both the run and the frame, so any
    # single frame can be regenerated without replaying the ones before it.
    rng = random.Random(seed * 100003 + frame)

    rnd.orbit_camera(camera, target=(0, 0, 0), rng=rng, distance=(11, 16), elevation=(12, 40))
    rnd.aim_light(sun, rng, elevation=(30, 70))
    rnd.randomize_light(sun, rng, intensity=(2.0, 4.0))
    rnd.randomize_surface(table_look, rng, roughness=(0.2, 0.9))
    rnd.random_pose(chair, rng, translate=((3, 0, -2), (5, 0, 2)), rotate=((0, 0, 0), (0, 360, 0)))

    rnd.clear(stage, "/World/Distractors")
    rnd.scatter_distractors(stage, "/World/Distractors", rng, count=6,
                            bounds=((-6, -0.6, -6), (6, -0.6, 6)), size=(0.3, 0.7))

    files = lumbre.render(stage, f"{out_prefix}.png", frame=frame, labels=True, camera="DatasetCam")
    total += len(files)
    print(f"frame {frame + 1}/{frames}: {', '.join(os.path.basename(f) for f in files)}")

print(f"done: {frames} frames, {total} files under {os.path.dirname(os.path.abspath(out_prefix))}")
