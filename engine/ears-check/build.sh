#!/bin/bash
# Builds ears-check (compares his ears in llama.cpp with the PyTorch reference) against an already-built llama.cpp tree
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LCPP=${LCPP:-$HERE/../../.run/lcpp-src}; LCPP="$(cd "$LCPP" && pwd)"
BUILD=${BUILD:-$LCPP/build-mac}
cmake --build "$BUILD" --target llama mtmd -j 10 >/dev/null
mkdir -p "$HERE/build"
${CXX:-c++} -std=c++17 -O2 -I"$LCPP/include" -I"$LCPP/ggml/include" -I"$LCPP/tools/mtmd" -I"$LCPP/src" \
    "$HERE/ears_check.cpp" -L"$BUILD/bin" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$BUILD/bin" \
    -o "$HERE/build/ears-check"
echo "built $HERE/build/ears-check"
