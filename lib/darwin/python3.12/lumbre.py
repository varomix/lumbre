"""Scripting API for Lumbre.

Everything here forwards to the host through a single JSON bridge
(``lumbre_native.call``), so extending the API means adding a command on the
Odin side rather than new C bindings.

Note this is deliberately *not* ``pxr``: OpenUSD's Python modules are not
shipped with Lumbre, and a renderer script mostly wants materials, camera and
render control rather than stage authoring. Read-only stage queries are
available through :func:`prims`.

    import lumbre
    for i, m in enumerate(lumbre.materials()):
        if m["roughness"] > 0.5:
            lumbre.set_material(i, roughness=0.2, metallic=1.0)
    lumbre.frame_all()
"""

import json as _json

import lumbre_native as _native

__all__ = [
    "call", "stats", "materials", "material", "set_material",
    "prims", "frame_all", "restart", "settings", "set_settings",
    "render_to_file", "render_status", "render_cancel",
    "save_look", "load_look",
]


def call(command, **payload):
    """Send a raw command to the host. Returns the decoded JSON reply."""
    reply = _native.call(command, _json.dumps(payload))
    return _json.loads(reply) if reply else None


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


def save_look():
    """Write the current material overrides beside the scene as a look file."""
    return call("save_look")


def load_look():
    """Re-apply the scene's look file, if one exists."""
    return call("load_look")


def frame_all():
    """Frame the whole scene in the viewport."""
    return call("frame_all")


def restart():
    """Discard accumulated samples and start the image again."""
    return call("restart")
