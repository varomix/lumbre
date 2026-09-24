#!/usr/bin/env bash
# Build the Houdini-only Hydra plugin. This never invokes Odin or modifies the
# root CLI executable; it links only against Houdini's bundled HDK/USD runtime.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLUGIN_ROOT="${REPO_ROOT}/houdini"
# Defaults to whichever Houdini the installer marked Current. Everything that
# differs between Houdini releases -- C++ standard, Python version -- is read
# from that install's own HDK makefile rather than assumed.
HOUDINI_INSTALL="${HOUDINI_INSTALL:-/Applications/Houdini/Current}"
HFS="${HOUDINI_INSTALL}/Frameworks/Houdini.framework/Versions/Current/Resources"
HDK_INCLUDE="${HFS}/toolkit/include"
HDK_MAKEFILE="${HFS}/toolkit/makefiles/Makefile.osx"
INSTALL_ROOT="${PLUGIN_ROOT}/install"
PLUGIN_INSTALL="${INSTALL_ROOT}/usd_plugins/HdLumbre"

if [[ ! -f "${HDK_INCLUDE}/pxr/imaging/hd/rendererPlugin.h" ]]; then
    echo "Houdini HDK headers were not found under: ${HDK_INCLUDE}" >&2
    echo "Set HOUDINI_INSTALL to a Houdini 21 or 22 installation." >&2
    exit 1
fi

CXX_STD="$(grep -o -- '-std=c++[0-9]*' "${HDK_MAKEFILE}" | head -1)"
PY_VERSION="$(grep -o 'H_PYTHON_VERSION=[0-9.]*' "${HDK_MAKEFILE}" | head -1 | cut -d= -f2)"
HOUDINI_VERSION="$(grep -o 'VERSION=\\"[0-9.]*' "${HDK_MAKEFILE}" | head -1 | grep -o '[0-9.]*$')"
CXX_STD="${CXX_STD:--std=c++17}"
if [[ -z "${PY_VERSION}" ]]; then
    echo "Could not read H_PYTHON_VERSION from ${HDK_MAKEFILE}" >&2
    exit 1
fi
echo "Building against Houdini ${HOUDINI_VERSION:-?} (${CXX_STD}, Python ${PY_VERSION})"

# Link the USD, OpenSubdiv and Python libraries the plugin actually uses.
# It used to leave every symbol for the host to provide
# (-undefined dynamic_lookup), which only works if Houdini happens to have
# loaded all of them before the plugin -- Houdini 22's husk had not loaded
# libpxr_hgi, so the plugin failed to load with no error shown. The rpaths
# are the RESOLVED install paths, not the `Current` symlink, so dyld matches
# the copies Houdini already loaded instead of loading a second USD/Python.
HOUDINI_REAL="$(cd "${HOUDINI_INSTALL}" && pwd -P)"
HOUDINI_LIBS="${HOUDINI_REAL}/Frameworks/Houdini.framework/Versions/Current/Libraries"
HOUDINI_LIBS="$(cd "${HOUDINI_LIBS}" && pwd -P)"
PYTHON_DIR="$(cd "${HOUDINI_REAL}/Frameworks/Python.framework/Versions/${PY_VERSION}" && pwd -P)"

# Build the Houdini-safe Lumbre bridge dylib first. The plugin loads it at
# runtime; building it here keeps the install tree self-consistent.
"$(dirname "${BASH_SOURCE[0]}")/build_bridge.sh"

rm -rf "${PLUGIN_INSTALL}" "${INSTALL_ROOT}/houdini" "${INSTALL_ROOT}/packages"
mkdir -p "${PLUGIN_INSTALL}/resources" \
         "${INSTALL_ROOT}/houdini/soho/parameters" \
         "${INSTALL_ROOT}/packages"

# Houdini's `hcustom` wrapper determines the target CPU from
# `uname -p`, which reports the wrong value in this shell and causes an x86
# build attempt on Apple Silicon. Compile with the equivalent documented HDK
# bundle settings explicitly.
CLANGXX="$(xcrun --find clang++)"
SDKROOT="$(xcrun --show-sdk-path)"
"${CLANGXX}" \
    -arch arm64 "${CXX_STD}" -O3 -fPIC -fvisibility=hidden \
    -DH_PYTHON_VERSION="${PY_VERSION}" -DUSE_QT6=1 \
    -D_GNU_SOURCE -DMBSD -DMBSD_COCOA -DARM64 -DMBSD_ARM \
    -DSIZEOF_VOID_P=8 -DFBX_ENABLED=1 -DOPENCL_ENABLED=1 \
    -DOPENVDB_ENABLED=1 -DUSE_VULKAN=1 \
    -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS=1 -DSESI_LITTLE_ENDIAN \
    -DHBOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT=1 \
    -DHBOOST_BIND_GLOBAL_PLACEHOLDERS=1 -DENABLE_THREADS -DUSE_PTHREADS \
    -D_REENTRANT -D_FILE_OFFSET_BITS=64 \
    -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION \
    -isystem "${HDK_INCLUDE}" -isystem "${HDK_INCLUDE}/python${PY_VERSION}" \
    -isysroot "${SDKROOT}" \
    -mmacosx-version-min=10.15 -bundle \
    -L"${HOUDINI_LIBS}" \
    -lpxr_hd -lpxr_hdsi -lpxr_hf -lpxr_hgi -lpxr_hio -lpxr_pxOsd -lpxr_sdf \
    -lpxr_vt -lpxr_gf -lpxr_tf -lpxr_python -losdCPU \
    "${PYTHON_DIR}/Python" \
    -Wl,-rpath,"${HOUDINI_LIBS}" -Wl,-rpath,"${PYTHON_DIR}" \
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
