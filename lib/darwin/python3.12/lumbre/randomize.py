"""Domain randomization over a ``Usd.Stage``.

Every helper here is ordinary pxr authoring. Nothing is special to Lumbre
except the ``semantic:class`` attribute its label pass reads — which is the
point: a randomized stage is a normal USD stage, so it can be saved, opened in
another DCC, and rendered again later to the same pixels.

Each helper takes an explicit ``random.Random``. Seed one per frame and a frame
is reproducible on its own, without replaying every frame before it:

    import random
    import lumbre
    from lumbre import randomize as rnd

    for frame in range(100):
        rng = random.Random(frame)
        rnd.orbit_camera(camera, target=(0, 0, 0), rng=rng, distance=(6, 10))
        rnd.aim_light(sun, rng)
        rnd.clear(stage, "/World/Distractors")
        rnd.scatter_distractors(stage, "/World/Distractors", rng, count=8)
        lumbre.render(stage, "out/ds.png", frame=frame, labels=True)

Ranges are ``(low, high)`` pairs sampled uniformly. A vector range is a pair of
vectors, sampled per component.
"""

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

__all__ = [
    "SEMANTIC_CLASS_ATTR", "set_semantic_class",
    "preview_surface", "bind_preview_surface", "randomize_surface", "preview_surfaces",
    "randomize_light", "aim_light",
    "look_at", "orbit_camera",
    "random_pose",
    "scatter_distractors", "clear",
]

# Must match SEMANTIC_CLASS_ATTR in importers/usd.odin.
SEMANTIC_CLASS_ATTR = "semantic:class"

_GPRIMS = {
    "Cube": UsdGeom.Cube,
    "Sphere": UsdGeom.Sphere,
    "Cylinder": UsdGeom.Cylinder,
    "Cone": UsdGeom.Cone,
    "Capsule": UsdGeom.Capsule,
}


def _uniform(rng, span):
    low, high = span
    return rng.uniform(low, high)


def _uniform3(rng, span):
    low, high = span
    return tuple(rng.uniform(low[i], high[i]) for i in range(3))


def _prim(obj):
    return obj.GetPrim() if hasattr(obj, "GetPrim") else obj


# ── labels ──────────────────────────────────────────────────────────────────


def set_semantic_class(prim, name):
    """Label ``prim`` and, by inheritance, everything beneath it."""
    prim = _prim(prim)
    attr = prim.CreateAttribute(SEMANTIC_CLASS_ATTR, Sdf.ValueTypeNames.String, custom=True)
    attr.Set(name)
    return attr


# ── materials ───────────────────────────────────────────────────────────────


def preview_surface(stage, path, base_color=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0):
    """Define a UsdPreviewSurface material at ``path``. Returns the material."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_preview_surface(prim, material):
    """Bind ``material`` to ``prim``."""
    prim = _prim(prim)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def preview_surfaces(stage):
    """Every UsdPreviewSurface shader on the stage."""
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            if shader.GetIdAttr().Get() == "UsdPreviewSurface":
                yield shader


def randomize_surface(shader_or_material, rng, color=((0, 0, 0), (1, 1, 1)),
                      roughness=(0.1, 1.0), metallic=None):
    """Resample a UsdPreviewSurface's colour and roughness, and its metallic
    when a range is given. Accepts the shader or a material made by
    :func:`preview_surface`."""
    shader = shader_or_material
    if isinstance(shader_or_material, UsdShade.Material):
        source = shader_or_material.ComputeSurfaceSource()
        shader = source[0] if source and source[0] else None
        if shader is None:
            raise ValueError("material has no connected surface shader")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_uniform3(rng, color)))
    if roughness is not None:
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(_uniform(rng, roughness))
    if metallic is not None:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(_uniform(rng, metallic))


# ── lights ──────────────────────────────────────────────────────────────────


def randomize_light(light, rng, intensity=None, color=None):
    """Resample a UsdLux light's intensity and/or colour."""
    api = UsdLux.LightAPI(_prim(light))
    if intensity is not None:
        api.CreateIntensityAttr().Set(_uniform(rng, intensity))
    if color is not None:
        api.CreateColorAttr().Set(Gf.Vec3f(*_uniform3(rng, color)))


def aim_light(light, rng, elevation=(20.0, 70.0), azimuth=(0.0, 360.0)):
    """Point a light (a DistantLight, typically) down from a random elevation
    and azimuth, in degrees. A light shines along its local -Z, so tilting
    that axis below the horizon is what makes it shine down."""
    xf = UsdGeom.XformCommonAPI(_prim(light))
    xf.SetRotate(Gf.Vec3f(-_uniform(rng, elevation), _uniform(rng, azimuth), 0.0),
                 UsdGeom.XformCommonAPI.RotationOrderXYZ)


# ── cameras ─────────────────────────────────────────────────────────────────


def look_at(xformable, eye, target, up=(0.0, 1.0, 0.0)):
    """Place a camera (or any xformable) at ``eye`` looking at ``target``.

    Replaces the prim's transform ops with a single matrix: a look-at is a
    full orientation, and composing it with ops already authored would put
    the camera somewhere else.
    """
    xformable = UsdGeom.Xformable(_prim(xformable))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(view.GetInverse())


def orbit_camera(camera, target, rng, distance=(5.0, 10.0), elevation=(10.0, 45.0),
                 azimuth=(0.0, 360.0), up=(0.0, 1.0, 0.0)):
    """Put a camera on a random point of a sphere around ``target`` and aim it
    there. Angles in degrees; elevation is above the horizontal plane of a
    Y-up stage. Returns the eye position."""
    r = _uniform(rng, distance)
    el = math.radians(_uniform(rng, elevation))
    az = math.radians(_uniform(rng, azimuth))
    tx, ty, tz = target
    eye = (tx + r * math.cos(el) * math.sin(az),
           ty + r * math.sin(el),
           tz + r * math.cos(el) * math.cos(az))
    look_at(camera, eye, target, up)
    return eye


# ── poses and distractors ───────────────────────────────────────────────────


def random_pose(xformable, rng, translate=None, rotate=None, scale=None):
    """Set a prim's translate, rotate (XYZ degrees) and uniform scale to values
    sampled from the given ranges; ``None`` leaves that component alone.

    Absolute rather than a jitter on what is authored, so resampling a frame
    never drifts: frame 50 does not depend on frames 0 to 49.
    """
    xf = UsdGeom.XformCommonAPI(_prim(xformable))
    if translate is not None:
        xf.SetTranslate(Gf.Vec3d(*_uniform3(rng, translate)))
    if rotate is not None:
        xf.SetRotate(Gf.Vec3f(*_uniform3(rng, rotate)), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    if scale is not None:
        s = _uniform(rng, scale)
        xf.SetScale(Gf.Vec3f(s, s, s))


def scatter_distractors(stage, parent, rng, count, bounds=((-3, 0, -3), (3, 1, 3)),
                        kinds=("Cube", "Sphere", "Cylinder", "Cone"), size=(0.2, 0.6),
                        semantic_class="distractor"):
    """Scatter ``count`` randomly posed, randomly coloured gprims under
    ``parent``. Each gets its own material and ``semantic_class``, so they
    appear in the labels as a class a model can learn to ignore. Returns the
    prims."""
    root = UsdGeom.Scope.Define(stage, parent).GetPrim()
    set_semantic_class(root, semantic_class)
    prims = []
    for i in range(count):
        kind = kinds[rng.randrange(len(kinds))]
        path = f"{parent}/{kind}_{i:03d}"
        gprim = _GPRIMS[kind].Define(stage, path)
        random_pose(gprim, rng, translate=bounds, rotate=((0, 0, 0), (360, 360, 360)), scale=size)
        material = preview_surface(stage, f"{parent}/Looks/{kind}_{i:03d}")
        randomize_surface(material, rng)
        bind_preview_surface(gprim, material)
        prims.append(gprim.GetPrim())
    return prims


def clear(stage, path):
    """Remove the prim at ``path`` and everything beneath it, if present —
    the previous frame's distractors, typically."""
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
