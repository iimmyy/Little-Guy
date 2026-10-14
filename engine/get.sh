#!/bin/bash
# Rebuild the engine source: clone upstream llama.cpp at engine/BASE and apply engine/patches.
# usage: engine/get.sh [target dir]   (default ./llama.cpp-littleguy)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:-llama.cpp-littleguy}
BASE=$(cat "$HERE/BASE")
git clone https://github.com/ggml-org/llama.cpp "$DEST"
cd "$DEST"
git checkout -q -b little-guy "$BASE"
git am -q "$HERE"/patches/*.patch
echo "engine source ready in $DEST (branch little-guy, $(git rev-list --count "$BASE"..HEAD) patches)"
