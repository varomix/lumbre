#!/usr/bin/env bash
cd '/Applications/Houdini/Houdini21.0.751/Frameworks/Houdini.framework/Versions/Current/Resources'; source ./houdini_setup; cd -
# Launch Houdini with Lumbre's local paths. `houdinifx` must already be on
# PATH (for example, after sourcing Houdini's houdini_setup script).
root="/Users/varomix/dev/ODIN_DEV/lumbre/houdini/install"

# unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH PYTHONPATH
export HOUDINI_PATH="$root/houdini${HOUDINI_PATH:+:$HOUDINI_PATH}"
export PXR_PLUGINPATH_NAME="$root/usd_plugins/HdLumbre/resources"

exec houdinifx "$@"
