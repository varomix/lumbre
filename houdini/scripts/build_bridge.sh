#!/usr/bin/env bash
# Build the Houdini-safe Lumbre bridge dylib from the `houdini/lumbre_bridge`
# Odin package. It imports only the USD-free `core` renderer, so the resulting
# dylib links neither `lib/darwin/libusd_shim.dylib` nor Lumbre's vendored
# OpenUSD. That is what lets the Hydra plugin call Lumbre inside the Houdini
# process without clashing with Houdini's own USD runtime.
#
# This never invokes the CLI build (`odin build .`) and never touches the root
# executable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRIDGE_SRC="${REPO_ROOT}/houdini/lumbre_bridge"
INSTALL_LIB="${REPO_ROOT}/houdini/install/lib"
DYLIB="${INSTALL_LIB}/libLumbreBridge.dylib"

mkdir -p "${INSTALL_LIB}"

# Odin compiles to per-package objects next to the output, then links them.
# Since Odin commit 9aa84b5e3 ("Use posix_spawnp instead of system() when
# invoking the linker") the macOS shared-library link passes
# -Wl,-init,'__odin_entry_point' with no shell to strip the quotes, so ld
# looks for a symbol literally named '__odin_entry_point' and fails. When that
# happens, re-run the exact link command Odin printed through a shell, which
# removes the quotes; with a fixed Odin the first link simply succeeds.
BUILD_LOG="$(mktemp -t lumbre_bridge_build)"
if ! odin build "${BRIDGE_SRC}" -build-mode:shared -out:"${DYLIB}" -show-system-calls >"${BUILD_LOG}" 2>&1; then
    LINK_CMD="$(grep '^clang ' "${BUILD_LOG}" | tail -1)"
    if grep -q "'__odin_entry_point'" "${BUILD_LOG}" && [[ -n "${LINK_CMD}" ]]; then
        echo "Relinking through a shell to work around Odin's quoted -init flag"
        bash -c "${LINK_CMD}"
    else
        cat "${BUILD_LOG}" >&2
        rm -f "${BUILD_LOG}"
        exit 1
    fi
fi
rm -f "${BUILD_LOG}"
# The per-package objects are only link inputs.
rm -f "${INSTALL_LIB}"/libLumbreBridge-*.o

# Give the dylib an @rpath install name so a consumer (the Hydra plugin) can
# locate it via its own rpath instead of the absolute build path.
install_name_tool -id "@rpath/libLumbreBridge.dylib" "${DYLIB}"
codesign --force --sign - "${DYLIB}"

# Guardrail: fail loudly if a USD/OpenUSD dependency ever creeps in. The whole
# point of this dylib is that it does not carry one.
if otool -L "${DYLIB}" | grep -Eiq 'usd|libusd_shim'; then
    echo "ERROR: bridge dylib links a USD library; it must not." >&2
    otool -L "${DYLIB}" >&2
    exit 1
fi

# Functional smoke test: create, upload a triangle, set camera, render, and read
# the framebuffer back, asserting a non-empty image. A broken bridge fails here
# rather than silently in Houdini.
SMOKE_BIN="$(mktemp -t lumbre_bridge_smoke)"
cc -O2 -I"${BRIDGE_SRC}" "${BRIDGE_SRC}/smoke_test.c" "${DYLIB}" \
   -Wl,-rpath,"${INSTALL_LIB}" -o "${SMOKE_BIN}"
"${SMOKE_BIN}"
rm -f "${SMOKE_BIN}"

echo "Built Lumbre bridge dylib: ${DYLIB}"
