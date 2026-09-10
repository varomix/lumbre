#!/usr/bin/env bash
# Vendors OpenUSD's Python bindings (`pxr`) for Lumbre's embedded interpreter.
#
# The interpreter Lumbre embeds is the libpython OpenUSD already links (see
# scripts/vendor_python.sh), so OpenUSD's own `pxr` extension modules are ABI
# compatible with it: authoring from a Lumbre script is the real Usd/Sdf API,
# not a bridge that re-describes it.
#
# The one thing that must not happen is a second copy of USD in the process.
# The modules as built reference `@rpath/libusd_*.dylib` with an rpath into the
# OpenUSD build tree, while the shim loaded the vendored copies under their
# `@loader_path/...` install names. dyld does not treat those as the same
# image, so `from pxr import Usd` loads a second libusd_tf, registers every
# TfType twice, and aborts. Measured, not guessed:
#
#   before: libusd_tf.dylib loaded from lib/darwin/usd AND /Users/.../OpenUSD/lib
#
# So every module is repointed at the vendored dylibs, and the build-tree rpath
# is deleted outright, so a missing dependency fails loudly instead of quietly
# resolving into a second USD.
#
# Only the modules authoring needs are vendored. Imaging, usdview, Hydra and
# MaterialX bindings would drag in GL, Metal and the whole of Hd for a script
# that defines prims and sets attributes.
#
# Layout produced:
#   lib/darwin/python3.12/pxr/<Module>/        __init__.py + _module.so
#   lib/darwin/usd/libusd_{usdUtils,...}.dylib the few dylibs not yet vendored
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
USD_ROOT="${USD_ROOT:-/Users/varomix/dev/OpenUSD}"

SRC="${USD_ROOT}/lib/python/pxr"
DEST="${REPO_ROOT}/lib/darwin/python3.12/pxr"
VENDOR_DIR="${REPO_ROOT}/lib/darwin/usd"

MODULES=(Tf Gf Trace Work Plug Vt Ts Ar Kind Sdf Sdr Pcp Usd UsdGeom UsdShade
         UsdLux UsdSemantics UsdUI UsdUtils)

# Dylibs the modules above need beyond what the shim already vendored. The
# script checks the closure below and stops if this list falls out of date.
EXTRA_DYLIBS=(libusd_usdUtils.dylib libusd_usdSemantics.dylib libusd_usdUI.dylib)

[[ -d "${SRC}" ]] || { echo "no pxr package at ${SRC}" >&2; exit 1; }
[[ -f "${VENDOR_DIR}/libusd_usd.dylib" ]] || {
    echo "vendored USD missing; run native/usd_shim/build.sh first" >&2; exit 1; }

# Repoints every @rpath dependency of `bin` that is vendored at `prefix`, and
# drops rpaths into the build tree. Reports anything left unresolved.
repoint() {
    local bin="$1" prefix="$2" missing=0
    for dep in $(otool -L "${bin}" | tail -n +2 | awk '{print $1}'); do
        [[ "${dep}" == @rpath/* ]] || continue
        local name="${dep#@rpath/}"
        # A module's own id shows up as its first entry.
        [[ "${name}" == "$(basename "${bin}")" ]] && continue
        if [[ -f "${VENDOR_DIR}/${name}" ]]; then
            install_name_tool -change "${dep}" "${prefix}/${name}" "${bin}"
        else
            echo "UNRESOLVED ${name} (needed by ${bin#"${REPO_ROOT}"/})" >&2
            missing=1
        fi
    done
    while read -r rp; do
        install_name_tool -delete_rpath "${rp}" "${bin}"
    done < <(otool -l "${bin}" | awk '/LC_RPATH/{getline; getline; print $2}' | grep "^${USD_ROOT}" || true)
    return ${missing}
}

echo "==> Vendoring ${#EXTRA_DYLIBS[@]} extra dylib(s) into ${VENDOR_DIR}"
for lib in "${EXTRA_DYLIBS[@]}"; do
    cp -f "${USD_ROOT}/lib/${lib}" "${VENDOR_DIR}/"
    install_name_tool -id "@loader_path/${lib}" "${VENDOR_DIR}/${lib}"
done
failed=0
for lib in "${EXTRA_DYLIBS[@]}"; do
    repoint "${VENDOR_DIR}/${lib}" "@loader_path" || failed=1
done

echo "==> Copying ${#MODULES[@]} pxr module(s) into ${DEST}"
rm -rf "${DEST}"
mkdir -p "${DEST}"
cp "${SRC}/__init__.py" "${DEST}/"
for mod in "${MODULES[@]}"; do
    mkdir -p "${DEST}/${mod}"
    # Sources and the extension only: the build tree's .pyc files were
    # compiled for its own paths, and the interpreter writes fresh ones.
    find "${SRC}/${mod}" -maxdepth 1 \( -name '*.py' -o -name '*.so' \) -exec cp {} "${DEST}/${mod}/" \;
done

echo "==> Repointing module dependencies at the vendored dylibs"
# DEST/<Module>/_module.so -> ../../../usd is lib/darwin/usd.
for so in "${DEST}"/*/*.so; do
    repoint "${so}" "@loader_path/../../../usd" || failed=1
done
if [[ ${failed} -ne 0 ]]; then
    echo "Unresolved dependencies above; extend EXTRA_DYLIBS." >&2
    exit 1
fi

echo "==> Re-signing (install_name_tool invalidates ad-hoc signatures)"
for bin in "${DEST}"/*/*.so "${EXTRA_DYLIBS[@]/#/${VENDOR_DIR}/}"; do
    codesign --force --sign - "${bin}"
done

echo "==> Done: $(du -sh "${DEST}" | awk '{print $1}') of pxr in ${DEST}"
