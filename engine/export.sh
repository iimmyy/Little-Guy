#!/bin/bash
# Refresh engine/patches from the engine fork (default .run/lcpp-src, branch little-guy): one patch per commit
# on top of the upstream llama.cpp base in engine/BASE. Run after committing engine changes in the fork.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FORK=${1:-$HERE/../.run/lcpp-src}
BASE=$(cat "$HERE/BASE")
rm -rf "$HERE/patches"
git -C "$FORK" format-patch --quiet --no-signature -o "$HERE/patches" "$BASE..little-guy"
echo "$(ls "$HERE/patches" | wc -l | tr -d ' ') patches on top of $BASE"
