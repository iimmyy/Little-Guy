#!/bin/bash
# Builds llama-lg-mind (Little Guy's brain for his mind loop, see lg-mind.cpp) against an already-built llama.cpp
# tree, without adding anything to that tree (research tools live in this repo, not the fork).
#   LCPP  the llama.cpp source tree   (default: ../../.run/lcpp-src from here)
#   BUILD its build folder            (default: $LCPP/build-mac)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LCPP=${LCPP:-$HERE/../../.run/lcpp-src}; LCPP="$(cd "$LCPP" && pwd)"
BUILD=${BUILD:-$LCPP/build-mac}
cmake --build "$BUILD" --target llama llama-common mtmd -j 10 >/dev/null
mkdir -p "$HERE/build"
${CXX:-c++} -std=c++17 -O2 -I"$LCPP/include" -I"$LCPP/ggml/include" -I"$LCPP/common" -I"$LCPP/vendor" -I"$LCPP/tools/mtmd" \
    "$HERE/lg-mind.cpp" -L"$BUILD/bin" -lllama-common -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$BUILD/bin" \
    -o "$HERE/build/llama-lg-mind"
echo "built $HERE/build/llama-lg-mind"
