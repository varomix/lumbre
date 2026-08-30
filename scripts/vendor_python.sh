#!/usr/bin/env bash
# Vendors a CPython standard library for Lumbre's embedded interpreter.
#
# Lumbre does not link its own Python: it drives the libpython that OpenUSD
# already ships (lib/darwin/usd/libpython3.12.dylib), because loading a second
# libpython into a process that already has USD's is a reliable way to crash.
# That library is the CPython *engine* only — no .py files, no extension
# modules — so it cannot start without a standard library. This script supplies
# one.
#
# The vendored stdlib MUST match the engine version exactly. OpenUSD is
# particular about its Python, and the point of vendoring is that Lumbre runs
# the same interpreter on every machine rather than whatever happens to be
# installed.
#
# Layout produced:
#   lib/darwin/python3.12/python312.zip     trimmed pure-Python stdlib
#   lib/darwin/python3.12/lib-dynload/*.so  C extension modules
#   lib/darwin/python3.12/*.dylib           libz / libffi, if needed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENGINE="${REPO_ROOT}/lib/darwin/usd/libpython3.12.dylib"
DEST="${REPO_ROOT}/lib/darwin/python3.12"

# Where to take the stdlib from. Must be the same CPython version as the engine.
PYSRC="${PYSRC:-/opt/miniconda3/lib/python3.12}"

# Extension modules may link these beyond libSystem; they are vendored next to
# the modules and repointed at @loader_path. Anything needing OpenSSL, sqlite,
# ncurses, readline, bz2, lzma, expat or tk is dropped instead — none of it is
# needed to script a renderer, and each would drag in more to vendor and sign.
ALLOWED_EXTRA=("libz.1.dylib" "libffi.8.dylib")

# Dropped wholesale: developer tooling and GUI toolkits, none of which a
# lookdev script needs, and which together are most of the stdlib's size.
TRIM=(test idlelib tkinter lib2to3 distutils ensurepip turtledemo
      site-packages lib-dynload config-3.12-darwin __pycache__ turtle.py)

die() { echo "error: $*" >&2; exit 1; }

# ── verify the source matches the engine ────────────────────────────────────
[[ -f "$ENGINE" ]] || die "engine not found: $ENGINE"
[[ -d "$PYSRC"  ]] || die "stdlib source not found: $PYSRC (set PYSRC=...)"

engine_ver="$(strings "$ENGINE" | grep -oE '^3\.12\.[0-9]+' | sort -u | head -1)"
src_ver="$(basename "$(dirname "$PYSRC")")"
src_ver="$("${PYSRC%/lib/python3.12}/bin/python3" -V 2>/dev/null | awk '{print $2}' || true)"
[[ -n "$engine_ver" ]] || die "could not read version from $ENGINE"

echo "==> engine   : $ENGINE ($engine_ver)"
echo "==> stdlib   : $PYSRC (${src_ver:-unknown})"
if [[ -n "$src_ver" && "$src_ver" != "$engine_ver" ]]; then
    die "version mismatch: engine is $engine_ver but stdlib source is $src_ver.
     Vendoring a mismatched stdlib is exactly the failure this script exists to prevent."
fi

rm -rf "$DEST"
mkdir -p "$DEST/lib-dynload"

# ── pure-Python stdlib, as a single zip on sys.path ─────────────────────────
# CPython's frozen zipimport reads this directly, which turns several thousand
# files into one.
echo "==> building python312.zip"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp -R "$PYSRC/." "$STAGE/"
for t in "${TRIM[@]}"; do rm -rf "${STAGE:?}/$t"; done
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true

( cd "$STAGE" && zip -q -r -9 "$DEST/python312.zip" . )

# ── extension modules ───────────────────────────────────────────────────────
echo "==> selecting extension modules"
accepted=0; rejected=0
declare -a needed_extra=()

for so in "$PYSRC"/lib-dynload/*.so; do
    [[ -e "$so" ]] || continue
    base="$(basename "$so")"
    # _test/_xx modules are CPython's own test scaffolding.
    case "$base" in _test*|_xx*|xx*) continue;; esac

    ok=1
    while read -r dep; do
        [[ -n "$dep" ]] || continue
        case "$dep" in
            /usr/lib/libSystem.B.dylib|/usr/lib/libc++.1.dylib) ;;
            *)
                lib="$(basename "$dep")"
                keep=0
                for a in "${ALLOWED_EXTRA[@]}"; do [[ "$lib" == "$a" ]] && keep=1; done
                if [[ $keep -eq 1 ]]; then
                    needed_extra+=("$lib")
                else
                    ok=0
                fi
                ;;
        esac
    done < <(otool -L "$so" | tail -n +2 | awk '{print $1}')

    if [[ $ok -eq 1 ]]; then
        cp "$so" "$DEST/lib-dynload/"
        accepted=$((accepted+1))
    else
        rejected=$((rejected+1))
    fi
done
echo "    accepted $accepted, skipped $rejected (external dependencies)"

# ── the handful of shared libraries those modules need ──────────────────────
if [[ ${#needed_extra[@]} -gt 0 ]]; then
    # shellcheck disable=SC2207
    uniq_extra=($(printf '%s\n' "${needed_extra[@]}" | sort -u))
    for lib in "${uniq_extra[@]}"; do
        src="$(dirname "$PYSRC")/../lib/$lib"
        [[ -f "$src" ]] || src="$(dirname "$(dirname "$PYSRC")")/$lib"
        [[ -f "$src" ]] || die "needed $lib but could not find it near $PYSRC"
        cp "$src" "$DEST/"
        echo "    vendored $lib"
    done

    # Repoint each module at the copy sitting beside it. Without this they look
    # for an @rpath that only exists inside the source installation.
    for so in "$DEST"/lib-dynload/*.so; do
        for lib in "${uniq_extra[@]}"; do
            install_name_tool -change "@rpath/$lib" "@loader_path/../$lib" "$so" 2>/dev/null || true
        done
    done
    for lib in "${uniq_extra[@]}"; do
        install_name_tool -id "@loader_path/$lib" "$DEST/$lib" 2>/dev/null || true
    done
fi

# ── re-sign ─────────────────────────────────────────────────────────────────
# Editing a Mach-O invalidates its signature, and arm64 refuses to load unsigned
# code. Ad-hoc is enough to run locally; distribution needs a real identity.
echo "==> re-signing"
find "$DEST" \( -name '*.so' -o -name '*.dylib' \) -exec codesign -f -s - {} \; 2>/dev/null

# The scripting API itself. Kept in python/ under version control and copied
# here, since this directory is rebuilt from scratch on every run.
if [[ -f "${REPO_ROOT}/python/lumbre.py" ]]; then
    cp "${REPO_ROOT}/python/lumbre.py" "$DEST/"
    echo "==> staged lumbre.py"
fi

echo "==> done"
du -sh "$DEST"
du -sh "$DEST"/* 2>/dev/null | sed 's/^/    /'
