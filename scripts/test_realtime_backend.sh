#!/usr/bin/env bash
# Run on a machine with the requested driver. Fails rather than silently
# substituting Metal for a requested Vulkan device.
set -euo pipefail
cd "$(dirname "$0")/.."
export LUMBRE_GPU_DRIVER="${1:-metal}"
export LUMBRE_GPU_DEBUG=1
export SDL_ASSERT=abort
python3 scripts/check_shader_bindings.py
odin test realtime -define:GPU_TESTS=true -define:ODIN_TEST_THREADS=1 -out:/tmp/lumbre-backend-tests
bash scripts/test_scripting.sh --no-gui
