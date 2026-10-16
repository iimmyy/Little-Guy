#!/bin/bash
# E4B community baseline on the Mac: stock llama.cpp mixes with the same importance matrix, scored on the held-out
# WikiText half vs E4B Q8_0 (KL, same top). usage: tools/squeeze/community_e4b.sh <work dir> <bf16 master> [types...]
set -uo pipefail
cd "$(dirname "$0")/../.."
D=$1; REF=$2; shift 2; TYPES=${*:-Q2_K IQ2_M IQ3_XXS Q3_K_S IQ3_M Q3_K_M Q4_K_S}
B=.run/lcpp-src/build-mac/bin
for t in $TYPES; do
  f=$D/e4b-imat-$t.gguf
  [ -f $f ] || $B/llama-quantize --imatrix $D/e4b.imatrix $REF $f $t > /dev/null 2>&1
  r=$($B/llama-perplexity -m $f -f .run/bench/wiki.eval.txt -c 512 --chunks 32 -ngl 99 --kl-divergence-base $D/e4b_q8_wiki512.kl \
      --kl-divergence 2>&1 | grep -E "^Mean    KLD|^Same top p" | tr -s ' ' | tr '\n' ' ')
  echo "$t $(stat -f %z $f) $r"
done
