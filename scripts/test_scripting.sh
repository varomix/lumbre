#!/usr/bin/env bash
# End-to-end tests for Python scripting: `lumbre --script`, lumbre.render,
# lumbre.randomize, and lumbre.show / lumbre.stage in the GUI.
#
#   scripts/test_scripting.sh            everything
#   scripts/test_scripting.sh --no-gui   skip the test that opens a window
#
# Needs ./lumbre and ./lumbre-gui already built. Outputs go to a temp dir that
# is kept on failure, so a mismatch can be looked at.
set -uo pipefail
export SDL_ASSERT=abort

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LUMBRE="${REPO}/lumbre"
GUI="${REPO}/lumbre-gui"
TESTS="${REPO}/python/tests"
SCENES="${REPO}/assets/usd_scene_tests"

run_gui=1
[[ "${1:-}" == "--no-gui" ]] && run_gui=0

[[ -x "${LUMBRE}" ]] || { echo "build ./lumbre first: odin build . -out:lumbre" >&2; exit 2; }

OUT="$(mktemp -d -t lumbre_scripting)"
failed=0
passed=0

section() { printf '\n== %s\n' "$1"; }
ok()   { echo "PASS $1"; passed=$((passed + 1)); }
bad()  { echo "FAIL $1"; failed=$((failed + 1)); }

# Runs a command, echoing its PASS/FAIL lines and any traceback, and counts the
# script itself as one check that must exit 0.
run_script() {
    local label="$1"; shift
    local log="${OUT}/$(echo "${label}" | tr ' /' '__').log"
    "$@" > "${log}" 2>&1
    local rc=$?
    grep -E '^(PASS|FAIL)|Traceback|Error:|^     ' "${log}" | sed 's/^/  /'
    passed=$((passed + $(grep -c '^PASS' "${log}")))
    failed=$((failed + $(grep -c '^FAIL' "${log}")))
    if [[ ${rc} -eq 0 ]] && ! grep -q 'script reported an error' "${log}"; then
        ok "${label} ran to completion"
    else
        bad "${label} exited ${rc} (log: ${log})"
    fi
}

same() {  # same <label> <file a> <file b>
    if [[ -f "$2" && -f "$3" ]] && cmp -s "$2" "$3"; then ok "$1"; else bad "$1 ($2 vs $3)"; fi
}

section "environment"
run_script "test_environment" "${LUMBRE}" --script "${TESTS}/test_environment.py"

section "render"
run_script "test_render" "${LUMBRE}" --script "${TESTS}/test_render.py" -- "${OUT}/render"
"${LUMBRE}" --scene "${SCENES}/semantic_classes.usda" --raster --labels -w 320 -h 240 \
    -o "${OUT}/render/raster/sc.png" > "${OUT}/raster_sc.log" 2>&1
for f in sc.png sc.labels.exr sc.coco.json; do
    same "lumbre.render matches --raster --labels: ${f}" "${OUT}/render/script/${f}" "${OUT}/render/raster/${f}"
done

section "dataset identity"
run_script "test_dataset_identity" "${LUMBRE}" --script "${TESTS}/test_dataset_identity.py" -- "${OUT}/identity"

section "randomize"
run_script "test_randomize seed 1 (a)" "${LUMBRE}" --script "${TESTS}/test_randomize.py" -- "${OUT}/rand_a" 1
run_script "test_randomize seed 1 (b)" "${LUMBRE}" --script "${TESTS}/test_randomize.py" -- "${OUT}/rand_b" 1
run_script "test_randomize seed 2"     "${LUMBRE}" --script "${TESTS}/test_randomize.py" -- "${OUT}/rand_c" 2
for f in ds.0000.png ds.0000.labels.exr ds.0000.coco.json ds.0001.png ds.0001.labels.exr ds.0001.coco.json; do
    same "same seed is byte-identical: ${f}" "${OUT}/rand_a/${f}" "${OUT}/rand_b/${f}"
    if cmp -s "${OUT}/rand_a/${f}" "${OUT}/rand_c/${f}"; then
        bad "a different seed changes ${f}"
    else
        ok "a different seed changes ${f}"
    fi
done
"${LUMBRE}" --scene "${OUT}/rand_a/frame1.usda" --raster --labels -w 320 -h 240 \
    -o "${OUT}/rand_a/exported/f1.png" > "${OUT}/raster_exported.log" 2>&1
for ext in png labels.exr; do
    same "exported .usda renders the same as the script: ${ext}" \
        "${OUT}/rand_a/ds.0001.${ext}" "${OUT}/rand_a/exported/f1.${ext}"
done

section "example"
run_script "random_dataset example" "${LUMBRE}" --script "${REPO}/python/examples/random_dataset.py" -- "${OUT}/example/ds" 2 3

if [[ ${run_gui} -eq 1 ]]; then
    section "gui live edit"
    if [[ -x "${GUI}" ]]; then
        mkdir -p "${OUT}/gui"
        # From a scratch directory: the test checks nothing is written to cwd.
        pushd "${OUT}/gui" >/dev/null
        run_script "test_gui_live" "${GUI}" \
            --scene "${SCENES}/semantic_classes.usda" --run-script "${TESTS}/test_gui_live.py"
        popd >/dev/null
    else
        bad "test_gui_live: build ./lumbre-gui first"
    fi
fi

printf '\n%d passed, %d failed\n' "${passed}" "${failed}"
if [[ ${failed} -eq 0 ]]; then
    rm -rf "${OUT}"
    exit 0
fi
echo "outputs kept in ${OUT}"
exit 1
