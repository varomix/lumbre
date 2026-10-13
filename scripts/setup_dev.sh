#!/usr/bin/env bash
# Checks for, and fetches, the tools needed to work on Lumbre.
#
#   scripts/setup_dev.sh           install whatever is missing
#   scripts/setup_dev.sh --check   only report; exits non-zero if anything a
#                                  plain build or test run needs is missing
#
# macOS only, like the vendored runtime under lib/darwin. Three tiers, because
# they are needed by different amounts of the work:
#
#   Build and test   Odin, SDL3, git-lfs and the LFS objects themselves (the
#                    vendored OpenUSD and Python runtime), python3 for the
#                    test scripts, and the Xcode command line tools.
#   Shaders          slangc, only when a .slang file changes: the compiled
#                    blobs are committed. There is no Homebrew formula, so a
#                    pinned release goes into .tools/, where
#                    scripts/build_shaders.sh looks when slangc is not on PATH.
#   Reported only    An OpenUSD build and a CPython 3.12 to rebuild the USD
#                    shim or re-vendor Python, OpenImageDenoise for --denoise,
#                    and Houdini for the Hydra plugin. These are large, or
#                    licensed, or tied to a local build; this script says
#                    where each is expected and leaves installing to you.
set -euo pipefail

cd "$(dirname "$0")/.."

SLANG_VERSION="2026.19"
SLANG_DIR=".tools/slang"

check_only=0
[[ "${1:-}" == "--check" ]] && check_only=1

missing=0
ok()   { echo "  ok       $1"; }
note() { echo "  optional $1"; }
bad()  { echo "  MISSING  $1"; missing=1; }

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Lumbre's vendored runtime is macOS-only (lib/darwin); nothing to set up here." >&2
  exit 1
fi

have() { command -v "$1" >/dev/null 2>&1; }

# Installs a Homebrew formula unless its command already exists.
brew_tool() {
  local cmd="$1" formula="$2"
  if have "$cmd"; then
    ok "$cmd"
  elif [[ $check_only -eq 1 ]]; then
    bad "$cmd (brew install $formula)"
  elif have brew; then
    echo "==> brew install $formula"
    brew install "$formula"
    ok "$cmd"
  else
    bad "$cmd -- Homebrew is not installed either; see https://brew.sh"
  fi
}

echo "Build and test"

if xcode-select -p >/dev/null 2>&1; then
  ok "Xcode command line tools"
else
  bad "Xcode command line tools (xcode-select --install)"
fi

brew_tool odin odin
brew_tool git-lfs git-lfs
brew_tool python3 python

# SDL3 is a library, not a command: Odin's vendor:sdl3 links Homebrew's copy.
if have brew && brew --prefix sdl3 >/dev/null 2>&1 && [[ -f "$(brew --prefix sdl3)/lib/libSDL3.dylib" ]]; then
  ok "sdl3"
elif [[ $check_only -eq 1 ]] || ! have brew; then
  bad "sdl3 (brew install sdl3)"
else
  echo "==> brew install sdl3"
  brew install sdl3
  ok "sdl3"
fi

# A checkout made without git-lfs holds pointer files where the dylibs should
# be, and fails at link time or at launch rather than here.
if have git-lfs; then
  if head -c 64 lib/darwin/libusd_shim.dylib 2>/dev/null | grep -q "git-lfs"; then
    if [[ $check_only -eq 1 ]]; then
      bad "LFS objects (git lfs install && git lfs pull)"
    else
      echo "==> git lfs pull"
      git lfs install --local
      git lfs pull
      ok "LFS objects"
    fi
  elif [[ -f lib/darwin/libusd_shim.dylib ]]; then
    ok "LFS objects"
  else
    bad "lib/darwin/libusd_shim.dylib (git lfs pull, or native/usd_shim/build.sh)"
  fi
fi

echo "Shaders"

if have slangc; then
  ok "slangc ($(slangc -v 2>&1 | head -1), on PATH)"
elif [[ -x "$SLANG_DIR/bin/slangc" ]]; then
  ok "slangc ($("$SLANG_DIR/bin/slangc" -v 2>&1 | head -1), in $SLANG_DIR)"
elif [[ $check_only -eq 1 ]]; then
  note "slangc -- needed only to rebuild shaders; run without --check to fetch $SLANG_VERSION"
else
  case "$(uname -m)" in
    arm64)  arch="aarch64" ;;
    x86_64) arch="x86_64" ;;
    *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
  esac
  url="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-macos-${arch}.tar.gz"
  echo "==> fetching Slang $SLANG_VERSION into $SLANG_DIR"
  mkdir -p "$SLANG_DIR"
  curl -fsSL "$url" | tar xz -C "$SLANG_DIR"
  ok "slangc ($("$SLANG_DIR/bin/slangc" -v 2>&1 | head -1), in $SLANG_DIR)"
fi

echo "Reported only"

if have cmake; then ok "cmake"; else note "cmake -- to rebuild the USD shim (brew install cmake)"; fi

usd_root="${USD_ROOT:-/Users/varomix/dev/OpenUSD}"
if [[ -d "$usd_root/lib" ]]; then
  ok "OpenUSD at $usd_root"
else
  note "OpenUSD build -- not at $usd_root; set USD_ROOT. Needed only by native/usd_shim/build.sh and scripts/vendor_pxr.sh"
fi

pysrc="${PYSRC:-/opt/miniconda3/lib/python3.12}"
if [[ -d "$pysrc" ]]; then
  ok "CPython 3.12 at $pysrc"
else
  note "CPython 3.12 stdlib -- not at $pysrc; set PYSRC. Needed only by scripts/vendor_python.sh"
fi

if [[ -n "${LUMBRE_OIDN_LIBRARY:-}" || -f lib/darwin/oidn/libOpenImageDenoise.dylib || -f /opt/homebrew/lib/libOpenImageDenoise.dylib ]]; then
  ok "OpenImageDenoise"
else
  note "OpenImageDenoise -- for --denoise (brew install open-image-denoise)"
fi

houdini="${HOUDINI_INSTALL:-/Applications/Houdini/Current}"
if [[ -d "$houdini" ]]; then
  ok "Houdini at $houdini"
else
  note "Houdini -- not at $houdini; set HOUDINI_INSTALL. Needed only for the Hydra plugin under houdini/"
fi

if [[ $missing -ne 0 ]]; then
  echo "Something a build needs is missing; see MISSING above." >&2
  exit 1
fi
echo "Ready: odin build . && odin build gui -out:lumbre-gui"
