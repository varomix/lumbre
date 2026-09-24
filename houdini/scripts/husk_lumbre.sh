#!/usr/bin/env bash
# Render a USD stage with Lumbre through Houdini's husk -- the same Hydra path
# the Solaris viewport uses, without the UI.
#
#   houdini/scripts/husk_lumbre.sh -o out.exr [-c /camera] [--res W H] scene.usd
#
# Sampling comes from render settings (samples, samples_per_update,
# max_depth); see houdini/README.md. HOUDINI_INSTALL selects the install.
set -euo pipefail

HOUDINI_INSTALL="${HOUDINI_INSTALL:-/Applications/Houdini/Current}"
HFS="${HOUDINI_INSTALL}/Frameworks/Houdini.framework/Versions/Current/Resources"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../install" && pwd)"

# See launch_houdini.sh: these break Houdini's own USD.
unset PYTHONPATH DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH

# houdini_setup reads unset variables, so it runs without `set -u`.
set +u
cd "${HFS}" && source ./houdini_setup >/dev/null
set -u
cd - >/dev/null

export HOUDINI_PATH="${root}/houdini;&"
export PXR_PLUGINPATH_NAME="${root}/usd_plugins/HdLumbre/resources${PXR_PLUGINPATH_NAME:+:${PXR_PLUGINPATH_NAME}}"

exec husk -R HdLumbreRendererPlugin "$@"
