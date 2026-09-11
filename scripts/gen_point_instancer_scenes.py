# Authors the PointInstancer test pair in assets/usd_scene_tests:
#
#   point_instancer.usda       a scatter of three prototypes, one of which nests
#                              a second PointInstancer, with a masked point
#   point_instancer_flat.usda  the same scene with every point written out as
#                              an Xform over its own copy of the prototype
#
# The flat twin mirrors the importer's transform order (instancer, point,
# prototype root), so the two must render byte-identical beauty and labels.
#
#   ./lumbre --script scripts/gen_point_instancer_scenes.py
import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, Vt

OUT = "assets/usd_scene_tests"
DEFAULT = Usd.TimeCode.Default()


def set_class(prim, name):
    prim.CreateAttribute("semantic:class", Sdf.ValueTypeNames.String, custom=True).Set(name)


def translate(prim, v):
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*v))


def point_transforms(instancer):
    xforms = instancer.ComputeInstanceTransformsAtTime(
        DEFAULT, DEFAULT,
        UsdGeom.PointInstancer.ExcludeProtoXform,
        UsdGeom.PointInstancer.IgnoreMask,
    )
    mask = instancer.ComputeMaskAtTime(DEFAULT)
    indices = instancer.GetProtoIndicesAttr().Get()
    return [(i, indices[i], xforms[i]) for i in range(len(xforms)) if not mask or mask[i]]


def author_box(stage, path):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(0.4)
    translate(cube.GetPrim(), (0, 0.2, 0))


def author_ball(stage, path):
    ball = UsdGeom.Sphere.Define(stage, path)
    ball.GetRadiusAttr().Set(0.25)
    translate(ball.GetPrim(), (0, 0.25, 0))
    set_class(ball.GetPrim(), "ball")


def author_peg(stage, path):
    peg = UsdGeom.Cube.Define(stage, path)
    peg.GetSizeAttr().Set(0.1)
    translate(peg.GetPrim(), (0, 0.05, 0))


def author_row(stage, path, inner_points=None):
    """A prototype holding its own PointInstancer of pegs, or, given the
    pegs' point transforms, the flattened equivalent."""
    row = UsdGeom.Xform.Define(stage, path)
    translate(row.GetPrim(), (0, 0.05, 0))
    if inner_points is None:
        inner = UsdGeom.PointInstancer.Define(stage, path + "/Inner")
        translate(inner.GetPrim(), (0, 0.1, 0))
        UsdGeom.Scope.Define(stage, path + "/Inner/Prototypes")
        author_peg(stage, path + "/Inner/Prototypes/Peg")
        inner.CreatePrototypesRel().SetTargets([Sdf.Path(path + "/Inner/Prototypes/Peg")])
        inner.CreateProtoIndicesAttr(Vt.IntArray([0, 0, 0]))
        inner.CreatePositionsAttr(Vt.Vec3fArray([(-0.15, 0, 0), (0, 0, 0), (0.15, 0, 0)]))
        inner.CreateScalesAttr(Vt.Vec3fArray([(1, 1, 1), (1, 1.5, 1), (1, 2, 1)]))
        return inner
    inner = UsdGeom.Xform.Define(stage, path + "/Inner")
    translate(inner.GetPrim(), (0, 0.1, 0))
    for index, _, xform in inner_points:
        point = UsdGeom.Xform.Define(stage, f"{path}/Inner/p{index:02d}")
        point.AddTransformOp().Set(xform)
        author_peg(stage, f"{path}/Inner/p{index:02d}/Peg")
    return None


PROTOS = ["Box", "Ball", "Row"]


def new_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(0, -0.05, 0))
    UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(5, 0.1, 4))

    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.GetFocalLengthAttr().Set(24)
    UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(0, 3.2, 4.2))
    UsdGeom.Xformable(cam).AddRotateXYZOp().Set(Gf.Vec3f(-38, 0, 0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(3)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 30, 0))
    return stage


def author_scatter_group(stage):
    group = UsdGeom.Xform.Define(stage, "/World/Scatter")
    translate(group.GetPrim(), (0.2, 0, -0.3))
    set_class(group.GetPrim(), "crate")
    return group


def main():
    # Instanced
    stage = new_stage()
    author_scatter_group(stage)
    scatter = UsdGeom.PointInstancer.Define(stage, "/World/Scatter/Points")
    UsdGeom.Scope.Define(stage, "/World/Scatter/Points/Prototypes")
    author_box(stage, "/World/Scatter/Points/Prototypes/Box")
    author_ball(stage, "/World/Scatter/Points/Prototypes/Ball")
    inner = author_row(stage, "/World/Scatter/Points/Prototypes/Row")

    positions, orientations, scales, indices = [], [], [], []
    for i in range(12):
        x, z = i % 4, i // 4
        positions.append((x * 0.9 - 1.35, 0, z * 0.9 - 0.9))
        a = math.radians(30 * i)
        orientations.append(Gf.Quath(math.cos(a / 2), Gf.Vec3h(0, math.sin(a / 2), 0)))
        s = 1 + 0.15 * (i % 3)
        scales.append((s, s, s))
        indices.append(i % 3)
    scatter.CreatePrototypesRel().SetTargets([Sdf.Path(f"/World/Scatter/Points/Prototypes/{p}") for p in PROTOS])
    scatter.CreateProtoIndicesAttr(Vt.IntArray(indices))
    scatter.CreatePositionsAttr(Vt.Vec3fArray(positions))
    scatter.CreateOrientationsAttr(Vt.QuathArray(orientations))
    scatter.CreateScalesAttr(Vt.Vec3fArray(scales))
    scatter.CreateInvisibleIdsAttr(Vt.Int64Array([5]))
    stage.GetRootLayer().Export(f"{OUT}/point_instancer.usda")

    points = point_transforms(scatter)
    inner_points = point_transforms(inner)

    # Flat twin
    flat = new_stage()
    author_scatter_group(flat)
    UsdGeom.Xform.Define(flat, "/World/Scatter/Points")
    for index, proto, xform in points:
        base = f"/World/Scatter/Points/p{index:02d}"
        UsdGeom.Xform.Define(flat, base).AddTransformOp().Set(xform)
        name = PROTOS[proto]
        if name == "Box":
            author_box(flat, f"{base}/Box")
        elif name == "Ball":
            author_ball(flat, f"{base}/Ball")
        else:
            author_row(flat, f"{base}/Row", inner_points)
    flat.GetRootLayer().Export(f"{OUT}/point_instancer_flat.usda")

    print(f"wrote point_instancer.usda ({len(points)} of 12 points visible) and point_instancer_flat.usda")


main()
