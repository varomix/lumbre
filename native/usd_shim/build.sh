#!/usr/bin/env bash
# One-time (or after usd_shim.cpp changes) build step for the OpenUSD FFI
# shim. Configures + builds libusd_shim.dylib against a local OpenUSD
# install, then vendors the transitive USD/TBB runtime dylibs it needs into
# lib/darwin/usd/ so the repo is self-contained after a checkout (no
# dependency on the build machine's OpenUSD install path at runtime).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
USD_ROOT="${USD_ROOT:-/Users/varomix/dev/OpenUSD}"
BUILD_DIR="${SCRIPT_DIR}/build"
LIB_DARWIN="${REPO_ROOT}/lib/darwin"
VENDOR_DIR="${LIB_DARWIN}/usd"

echo "==> Configuring usd_shim (USD_ROOT=${USD_ROOT})"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSD_ROOT="${USD_ROOT}"

echo "==> Building usd_shim"
cmake --build "${BUILD_DIR}" --config Release -j

mkdir -p "${LIB_DARWIN}" "${VENDOR_DIR}"
cp "${BUILD_DIR}/libusd_shim.dylib" "${LIB_DARWIN}/libusd_shim.dylib"

echo "==> Resolving transitive USD/TBB dependencies via otool -L"
# Walk otool -L output starting from libusd_shim.dylib, following @rpath
# references back into USD_ROOT/lib, until the dependency set stabilizes.
# Uses a newline-delimited "seen" list instead of an associative array for
# compatibility with the ancient bash 3.2 that ships as macOS's /bin/bash.
SEEN_FILE="$(mktemp)"
trap 'rm -f "${SEEN_FILE}"' EXIT
to_process=("${LIB_DARWIN}/libusd_shim.dylib")

PYTHON_LIB_DIR="$(dirname "$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")' 2>/dev/null || echo /opt/miniconda3/lib)")"
# @rpath entries aren't all rooted at USD_ROOT/lib -- e.g. libpython3.12
# resolves against whatever Python installation USD was built against
# (here, miniconda). Search each candidate directory in turn.
RPATH_SEARCH_DIRS=("${USD_ROOT}/lib" "/opt/miniconda3/lib" "${PYTHON_LIB_DIR}")

resolve_dep_path() {
    local dep="$1"
    if [[ "${dep}" == @rpath/* ]]; then
        local name="${dep#@rpath/}"
        for dir in "${RPATH_SEARCH_DIRS[@]}"; do
            if [[ -f "${dir}/${name}" ]]; then
                echo "${dir}/${name}"
                return
            fi
        done
        echo "" # not found under any known rpath dir; caller will skip it
    elif [[ "${dep}" == /System/* || "${dep}" == /usr/lib/* ]]; then
        echo "" # system libs, never vendor
    else
        echo "${dep}"
    fi
}

is_seen() {
    grep -qxF "$1" "${SEEN_FILE}" 2>/dev/null
}

while [[ ${#to_process[@]} -gt 0 ]]; do
    current="${to_process[0]}"
    to_process=("${to_process[@]:1}")
    [[ -z "${current}" ]] && continue
    is_seen "${current}" && continue
    echo "${current}" >> "${SEEN_FILE}"

    [[ -f "${current}" ]] || continue

    while IFS= read -r dep; do
        resolved="$(resolve_dep_path "${dep}")"
        [[ -z "${resolved}" ]] && continue
        is_seen "${resolved}" && continue
        to_process+=("${resolved}")
    done < <(otool -L "${current}" | tail -n +2 | awk '{print $1}')
done

dep_count=$(( $(wc -l < "${SEEN_FILE}") - 1 ))
echo "==> Vendoring ${dep_count} dependency dylib(s) into ${VENDOR_DIR}"
while IFS= read -r path; do
    [[ "${path}" == "${LIB_DARWIN}/libusd_shim.dylib" ]] && continue
    [[ -f "${path}" ]] || continue
    cp -f "${path}" "${VENDOR_DIR}/"
done < "${SEEN_FILE}"

echo "==> Vendoring USD plugin resources (plugInfo.json + schema data)"
# USD's Plug registry discovers schema/resolver registrations by looking
# for a "usd" subdirectory next to wherever each libusd_*.dylib was
# loaded from -- i.e. it expects ${VENDOR_DIR}/usd/<module>/resources/...
# mirroring USD_ROOT/lib/usd/<module>/resources/... Without this, prims
# fail to load with "Failed to find the plugInfo.json file that declares
# the plugin for ArDefaultResolver" (and similarly for every schema
# domain). Vendor the whole resource tree; it's ~8MB, cheap next to the
# ~150MB of dylibs already vendored.
mkdir -p "${VENDOR_DIR}/usd"
cp -R "${USD_ROOT}/lib/usd/." "${VENDOR_DIR}/usd/"

echo "==> Rewriting install names to @loader_path-relative for portability"
for dylib in "${VENDOR_DIR}"/*.dylib; do
    [[ -f "${dylib}" ]] || continue
    name="$(basename "${dylib}")"
    install_name_tool -id "@loader_path/${name}" "${dylib}" 2>/dev/null || true
    for dep in $(otool -L "${dylib}" | tail -n +2 | awk '{print $1}'); do
        if [[ "${dep}" == @rpath/* ]]; then
            dep_name="${dep#@rpath/}"
            if [[ -f "${VENDOR_DIR}/${dep_name}" ]]; then
                install_name_tool -change "${dep}" "@loader_path/${dep_name}" "${dylib}" 2>/dev/null || true
            fi
        fi
    done
done

# Odin links the lumbre binary with one LC_RPATH: @loader_path (the dir
# holding the executable, i.e. the repo root). For dyld to find the shim
# there, its install name must carry the lib/darwin prefix. Without this
# the binary aborts at startup with
#   dyld: Library not loaded: @rpath/libusd_shim.dylib
# unless DYLD_LIBRARY_PATH is set by hand.
install_name_tool -id "@rpath/lib/darwin/libusd_shim.dylib" "${LIB_DARWIN}/libusd_shim.dylib" 2>/dev/null || true

# otool -L lists the dylib's own install name as its first entry; skip it
# so we don't try to rewrite the shim as a dependency of itself.
shim_id="$(otool -D "${LIB_DARWIN}/libusd_shim.dylib" | tail -n +2)"
for dep in $(otool -L "${LIB_DARWIN}/libusd_shim.dylib" | tail -n +2 | awk '{print $1}'); do
    [[ "${dep}" == "${shim_id}" ]] && continue
    if [[ "${dep}" == @rpath/* ]]; then
        dep_name="${dep#@rpath/}"
        if [[ -f "${VENDOR_DIR}/${dep_name}" ]]; then
            install_name_tool -change "${dep}" "@loader_path/usd/${dep_name}" "${LIB_DARWIN}/libusd_shim.dylib" 2>/dev/null || true
        else
            echo "WARNING: ${dep_name} not vendored, libusd_shim.dylib still references @rpath/${dep_name}" >&2
        fi
    fi
done

# install_name_tool invalidates the ad-hoc code signature every dylib
# ships with; on Apple Silicon the kernel SIGKILLs any process that loads
# a Mach-O whose signature no longer matches its contents. Re-sign ad-hoc
# after rewriting load commands so the vendored copies load correctly.
echo "==> Re-signing rewritten dylibs (ad-hoc, required on Apple Silicon after install_name_tool)"
for dylib in "${VENDOR_DIR}"/*.dylib "${LIB_DARWIN}/libusd_shim.dylib"; do
    [[ -f "${dylib}" ]] || continue
    codesign --force --sign - "${dylib}"
done

echo "==> Done. libusd_shim.dylib + ${dep_count} vendored dylib(s) in ${LIB_DARWIN}"
