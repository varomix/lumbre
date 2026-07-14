#!/usr/bin/env bash
# Build the Houdini-only Hydra plugin. This never invokes Odin or modifies the
# root CLI executable; it links only against Houdini's bundled HDK/USD runtime.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLUGIN_ROOT="${REPO_ROOT}/houdini"
HOUDINI_INSTALL="${HOUDINI_INSTALL:-/Applications/Houdini/Houdini21.0.751}"
HFS="${HOUDINI_INSTALL}/Frameworks/Houdini.framework/Versions/21.0/Resources"
HDK_INCLUDE="${HFS}/toolkit/include"
INSTALL_ROOT="${PLUGIN_ROOT}/install"
PLUGIN_INSTALL="${INSTALL_ROOT}/usd_plugins/HdLumbre"

if [[ ! -f "${HDK_INCLUDE}/pxr/imaging/hd/rendererPlugin.h" ]]; then
    echo "Houdini 21 HDK headers were not found under: ${HDK_INCLUDE}" >&2
    echo "Set HOUDINI_INSTALL to a Houdini 21 installation." >&2
    exit 1
fi

# Build the Houdini-safe Lumbre bridge dylib first. The plugin loads it at
# runtime; building it here keeps the install tree self-consistent.
"$(dirname "${BASH_SOURCE[0]}")/build_bridge.sh"

rm -rf "${PLUGIN_INSTALL}" "${INSTALL_ROOT}/houdini" "${INSTALL_ROOT}/packages"
mkdir -p "${PLUGIN_INSTALL}/resources" \
         "${INSTALL_ROOT}/houdini/soho/parameters" \
         "${INSTALL_ROOT}/packages"

# Houdini 21.0.751's `hcustom` wrapper determines the target CPU from
# `uname -p`, which reports the wrong value in this shell and causes an x86
# build attempt on Apple Silicon. Compile with the equivalent documented HDK
# bundle settings explicitly, retaining dynamic symbol lookup so the plugin
# resolves USD/Hydra symbols from Houdini's already-loaded runtime.
CLANGXX="$(xcrun --find clang++)"
SDKROOT="$(xcrun --show-sdk-path)"
"${CLANGXX}" \
    -arch arm64 -std=c++17 -O3 -fPIC -fvisibility=hidden \
    -DH_PYTHON_VERSION=3.11 -DUSE_QT6=1 \
    -D_GNU_SOURCE -DMBSD -DMBSD_COCOA -DARM64 -DMBSD_ARM \
    -DSIZEOF_VOID_P=8 -DFBX_ENABLED=1 -DOPENCL_ENABLED=1 \
    -DOPENVDB_ENABLED=1 -DUSE_VULKAN=1 \
    -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS=1 -DSESI_LITTLE_ENDIAN \
    -DHBOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT=1 \
    -DHBOOST_BIND_GLOBAL_PLACEHOLDERS=1 -DENABLE_THREADS -DUSE_PTHREADS \
    -D_REENTRANT -D_FILE_OFFSET_BITS=64 \
    -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION \
    -isystem "${HDK_INCLUDE}" -isystem "${HDK_INCLUDE}/python3.11" \
    -isysroot "${SDKROOT}" \
    -mmacosx-version-min=10.15 -bundle -flat_namespace -undefined dynamic_lookup \
    -Wl,-rpath,@loader_path/../../lib -Wl,-rpath,"${INSTALL_ROOT}/lib" \
    "${PLUGIN_ROOT}/plugin/renderer_plugin.cpp" \
    -o "${PLUGIN_INSTALL}/libHdLumbre.dylib"

# The local plugin is a Mach-O bundle. Ad-hoc signing keeps macOS from
# rejecting it after each rebuild on Apple Silicon.
codesign --force --sign - "${PLUGIN_INSTALL}/libHdLumbre.dylib"

cp "${PLUGIN_ROOT}/resources/plugInfo.json" "${PLUGIN_INSTALL}/resources/plugInfo.json"
cp "${PLUGIN_ROOT}/houdini/UsdRenderers.json" "${INSTALL_ROOT}/houdini/UsdRenderers.json"
cp "${PLUGIN_ROOT}/houdini/soho/parameters/HdLumbreRendererPlugin_Viewport.ds" \
   "${INSTALL_ROOT}/houdini/soho/parameters/HdLumbreRendererPlugin_Viewport.ds"
sed "s|@LUMBRE_HOUDINI_ROOT@|${INSTALL_ROOT}|g" \
    "${PLUGIN_ROOT}/packages/lumbre.json.in" > "${INSTALL_ROOT}/packages/lumbre.json"

echo "Built Lumbre Houdini plugin: ${PLUGIN_INSTALL}/libHdLumbre.dylib"
echo "Restart Houdini after rebuilding to load the updated plugin."
