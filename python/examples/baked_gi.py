"""Baked GI: a room lit through its open side, with a scatter of crates.

    ./lumbre --script python/examples/baked_gi.py -- [out_dir] [grid] [rays]

Defaults: out/baked_gi, an 8-probe grid, 2048 rays per probe. Writes
<out_dir>/room.usda, its sidecar room.probes, and two renders of it:
room_env.png lit the old way, with ambient light from the environment alone,
and room_gi.png with the baked probes.

The sun comes in through the open front and lands on the floor; everything
else in the room is lit by what that patch and the sky bounce around. The
environment-only render cannot show that -- every surface facing up sees the
whole sky, ceiling or not -- and the probes can.

The crates are a UsdGeom.PointInstancer: a scatter is the usual way to fill
a set, and the probes see its copies as ordinary geometry.
"""

import os
import random
import sys

import lumbre
from lumbre import randomize as rnd
from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

W, H, D = 6.0, 3.0, 6.0  # room interior, metres: x, y, z. The +z side is open.
T = 0.1                  # wall thickness


def slab(stage, path, size, centre, colour):
    cube = UsdGeom.Cube.Define(stage, path)
    api = UsdGeom.XformCommonAPI(cube)
    api.SetScale(Gf.Vec3f(size[0] / 2, size[1] / 2, size[2] / 2))
    api.SetTranslate(Gf.Vec3d(*centre))
    name = path.rsplit("/", 1)[-1]
    rnd.bind_preview_surface(cube, rnd.preview_surface(stage, f"/World/Looks/{name}", base_color=colour))
    return cube


def build_stage(path, seed=7):
    """Authors the room to `path` and returns the opened stage."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    wall = (0.72, 0.70, 0.66)
    slab(stage, "/World/Room/Floor", (W, T, D), (0, -T / 2, 0), (0.55, 0.52, 0.48))
    slab(stage, "/World/Room/Ceiling", (W, T, D), (0, H + T / 2, 0), wall)
    slab(stage, "/World/Room/Back", (W, H, T), (0, H / 2, -D / 2 - T / 2), wall)
    slab(stage, "/World/Room/Left", (T, H, D), (-W / 2 - T / 2, H / 2, 0), (0.75, 0.25, 0.18))
    slab(stage, "/World/Room/Right", (T, H, D), (W / 2 + T / 2, H / 2, 0), (0.20, 0.35, 0.70))

    # The scatter: one crate prototype, points on a jittered grid over the
    # back half of the floor.
    inst = UsdGeom.PointInstancer.Define(stage, "/World/Crates")
    UsdGeom.Scope.Define(stage, "/World/Crates/Prototypes")
    crate = UsdGeom.Cube.Define(stage, "/World/Crates/Prototypes/Crate")
    crate.GetSizeAttr().Set(0.5)
    UsdGeom.XformCommonAPI(crate).SetTranslate(Gf.Vec3d(0, 0.25, 0))
    rnd.bind_preview_surface(crate, rnd.preview_surface(stage, "/World/Looks/Crate", base_color=(0.80, 0.62, 0.35)))
    rnd.set_semantic_class(inst.GetPrim(), "crate")
    inst.CreatePrototypesRel().SetTargets([crate.GetPath()])

    rng = random.Random(seed)
    points, orients = [], []
    for ix in range(4):
        for iz in range(3):
            x = -2.1 + ix * 1.4 + rng.uniform(-0.25, 0.25)
            z = -2.4 + iz * 1.0 + rng.uniform(-0.2, 0.2)
            points.append(Gf.Vec3f(x, 0, z))
            angle = rng.uniform(0, 90)
            orients.append(Gf.Quath(Gf.Rotation(Gf.Vec3d(0, 1, 0), angle).GetQuat()))
    inst.CreatePositionsAttr(Vt.Vec3fArray(points))
    inst.CreateOrientationsAttr(Vt.QuathArray(orients))
    inst.CreateProtoIndicesAttr(Vt.IntArray([0] * len(points)))

    # Low sun through the open front, onto the floor.
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1.5)
    sun.CreateAngleAttr(0.53)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-35, 15, 0))

    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateFocalLengthAttr(18.0)
    api = UsdGeom.XformCommonAPI(cam)
    api.SetTranslate(Gf.Vec3d(0.4, 1.4, D / 2 + 0.8))
    api.SetRotate(Gf.Vec3f(-8, 0, 0))

    stage.GetRootLayer().Save()
    return stage


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/baked_gi"
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rays = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    os.makedirs(out_dir, exist_ok=True)

    stage = build_stage(os.path.join(out_dir, "room.usda"))
    baked = lumbre.bake_probes(stage, grid=grid, rays=rays)
    print("baked", baked["dims"], "probes into", baked["file"])

    # The sidecar next to room.usda is picked up by itself; probes=False
    # renders the same stage without it.
    lumbre.render(stage, os.path.join(out_dir, "room_env.png"), width=640, height=360, probes=False)
    lumbre.render(stage, os.path.join(out_dir, "room_gi.png"), width=640, height=360)


if __name__ == "__main__":
    main()
