"""Scripting API for Lumbre.

Everything here forwards to the host through a single JSON bridge
(``lumbre_native.call``), so extending the API means adding a command on the
Odin side rather than new C bindings.

Scripts run in two hosts: the script editor in ``lumbre-gui``, and headlessly
under ``lumbre --script file.py``. :func:`host` says which. A command one host
does not offer raises ``RuntimeError`` naming it, rather than failing blind.

Stage authoring is OpenUSD's own ``pxr`` API, vendored with Lumbre — this
module does not re-describe it. Build or edit a ``Usd.Stage`` with ``pxr`` and
hand it to Lumbre.

    import lumbre
    for i, m in enumerate(lumbre.materials()):
        if m["roughness"] > 0.5:
            lumbre.set_material(i, roughness=0.2, metallic=1.0)
    lumbre.frame_all()
"""

import json as _json
import os as _os

import lumbre_native as _native

__all__ = [
    "call", "host", "render",
    "stats", "materials", "material", "set_material",
    "prims", "frame_all", "restart", "settings", "set_settings",
    "render_to_file", "render_status", "render_cancel",
    "save_look", "load_look", "export_look_usd", "pick", "lights", "set_light",
]


def call(command, **payload):
    """Send a raw command to the host. Returns the decoded JSON reply.

    Raises ``RuntimeError`` when the host reports an error, including a
    command it does not offer.
    """
    reply = _native.call(command, _json.dumps(payload))
    result = _json.loads(reply) if reply else None
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"lumbre.{command}: {result['error']}")
    return result


def host():
    """``"gui"`` inside lumbre-gui's script editor, ``"cli"`` under
    ``lumbre --script``."""
    return call("host")["host"]


# ── headless rendering (lumbre --script) ────────────────────────────────────


def render(stage, output, frame=None, labels=None, width=None, height=None,
           camera=None, view=None):
    """Rasterize a ``Usd.Stage`` and write it to disk. Returns the list of
    files written.

    ``stage`` is read as it is at the moment of the call, including unsaved
    edits and session-layer overrides, so a script can vary one stage and
    render it again in a loop. It is flattened on the way in; the stage itself
    is not modified.

    One frame is written per camera in the stage — or only ``camera``, by prim
    name. With ``frame`` set, the number goes into every file name
    (``out.0007.png``) and the COCO image id. ``labels`` adds the label EXR and
    COCO file; it and the resolution default to the command line's
    ``--labels``, ``--width`` and ``--height``.
    """
    from pxr import UsdUtils

    # The stage reaches the renderer through the process-wide stage cache, and
    # stays there after this returns. Erasing it again is not an option:
    # measured, `cache.Erase` destroys the stage even while the script still
    # holds it, so every prim handle the script kept goes invalid. A stage is
    # therefore cached once, on its first render, and reused after that.
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    if not stage_id.IsValid():
        stage_id = cache.Insert(stage)

    # Relative asset paths in a file-backed stage resolve against its layer;
    # an in-memory stage has none, so they resolve against the working dir.
    real = stage.GetRootLayer().realPath
    base_dir = _os.path.dirname(real) if real else _os.getcwd()

    args = {"stage_id": stage_id.ToLongInt(), "output": str(output),
            "base_dir": base_dir + _os.sep}
    for key, val in (("frame", frame), ("labels", labels), ("width", width),
                     ("height", height), ("camera", camera), ("view", view)):
        if val is not None:
            args[key] = val
    return call("render", **args)["files"]


# ── viewport and scene (lumbre-gui) ─────────────────────────────────────────


def stats():
    """Renderer state: accumulated samples, target, resolution, scene path."""
    return call("stats")


def materials():
    """Every material in the scene, as a list of dicts."""
    return call("materials")["materials"]


def material(index):
    """One material by index."""
    return materials()[index]


def set_material(index, **fields):
    """Update a material in place.

    Accepts any of: base_color, roughness, metallic, specular, ior,
    clearcoat, clearcoat_roughness, sheen, anisotropic, transmission,
    subsurface, emission, emission_strength. Colours are 3-element sequences.

    The viewport restarts accumulation but does not rebuild the scene, so this
    is cheap enough to drive from a loop.
    """
    return call("set_material", index=index, fields=fields)


def prims():
    """Read-only USD stage listing: [{'path': ..., 'type': ...}, ...].

    Empty when the loaded scene is not USD.
    """
    return call("prims")["prims"]


def settings():
    """Current render settings."""
    return call("settings")


def set_settings(**fields):
    """Update render settings: spp, max_depth, gi_cache, photons,
    photon_count, roughness_cutoff, glossy_bias, debug_mode."""
    return call("set_settings", fields=fields)


def render_to_file(path, width=None, height=None, spp=None, depth=None,
                   aovs=None, denoise=None):
    """Start an offline render. Returns immediately; poll :func:`render_status`.

    Runs on its own thread with its own GPU renderer, so the viewport keeps
    working, and writes through the same path the CLI uses.
    """
    args = {"path": path}
    for key, val in (("width", width), ("height", height), ("spp", spp),
                     ("depth", depth), ("aovs", aovs), ("denoise", denoise)):
        if val is not None:
            args[key] = val
    return call("render_to_file", **args)


def render_status():
    """Progress of the offline render: running, progress, status, elapsed."""
    return call("render_status")


def render_cancel():
    """Stop the offline render after the current batch."""
    return call("render_cancel")


def lights():
    """Every analytic light in the scene. The dome/HDRI is not included: it
    lives on the scene environment rather than the light list."""
    return call("lights")["lights"]


def set_light(index, **fields):
    """Update a light: intensity, position, direction, radius, height.

    Rewrites just the light buffers, so this is cheap enough to animate.
    Changing a light's *type* is not supported here because it resizes the
    per-kind GPU buffers.
    """
    return call("set_light", index=index, fields=fields)


def pick(u=0.5, v=0.5):
    """Trace one ray through the viewport and report what it hits.

    ``u`` and ``v`` are normalised viewport coordinates with the origin at the
    bottom-left. Returns the material index, hit point and normal, and selects
    that material in the Material panel.
    """
    return call("pick", u=u, v=v)


def save_look():
    """Write the current material overrides beside the scene as a look file."""
    return call("save_look")


def export_look_usd():
    """Author the current materials as a USD overlay layer beside the scene.

    Writes ``<scene>_look.usda``, which sublayers the original and carries only
    `over` prims with the changed shader inputs — the source asset is never
    modified. Only materials imported from a USD material prim can be authored.
    """
    return call("export_look_usd")


def load_look():
    """Re-apply the scene's look file, if one exists."""
    return call("load_look")


def frame_all():
    """Frame the whole scene in the viewport."""
    return call("frame_all")


def restart():
    """Discard accumulated samples and start the image again."""
    return call("restart")
