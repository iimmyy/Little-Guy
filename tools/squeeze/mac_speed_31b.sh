#!/bin/bash
# 31B talking speed on this Mac: a few configurations, one line each (tokens/s for 32 generated tokens).
# usage: tools/squeeze/mac_speed_31b.sh <brain.gguf> "<llama-bench args>" [...more arg sets]
cd "$(dirname "$0")/../.."
B=.run/lcpp-src/build-mac/bin
f=$1; shift
for a in "$@"; do
  r=$($B/llama-bench --offline -m "$f" $a -p 0 -n 32 -r 1 2>&1 | grep -oE "tg32 *\| *[0-9.]+|failed to [a-z ]+" | tr -s " " | head -1)
  echo "$(basename "$f") [$a]: ${r:-no output}"
done
