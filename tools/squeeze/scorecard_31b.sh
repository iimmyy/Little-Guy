#!/bin/bash
# 31B scorecard: community mixes with the same mixed importance matrix, ours (plain and matched Q3_K), earlier brains
cd "$(dirname "$0")/../.."
W=${LITTLEGUY_WEIGHTS:-$(grep -E "^LITTLEGUY_WEIGHTS=" littleguy.env 2>/dev/null | cut -d= -f2)}
[ -n "$W" ] || { echo "set LITTLEGUY_WEIGHTS (environment or littleguy.env)"; exit 1; }
D=$W/big31b; B=.run/lcpp-src/build-mac/bin
for t in Q3_K_S IQ3_M Q3_K_M; do
  [ -f $D/31b-mixcom-$t.gguf ] || $B/llama-quantize --imatrix $D/31b-mix.imatrix $D/31b-bf16.gguf $D/31b-mixcom-$t.gguf $t 6 > /dev/null 2>&1
done
until [ -f $D/31b-mm-Q3_K.gguf ] && grep -q "built" $D/match-assemble.log 2>/dev/null; do sleep 60; done
tools/squeeze/score_31b.sh $D/31b-mixim-Q3_K.gguf $D/31b-mm-Q3_K.gguf $D/31b-mixcom-Q3_K_S.gguf $D/31b-mixcom-IQ3_M.gguf \
  $D/31b-mixcom-Q3_K_M.gguf .run/weights/big/31b-16gb.gguf .run/weights/big/31b-C-preview.gguf
CPU=1 tools/squeeze/score_31b.sh .run/weights/big/gemma-4-31B_q4_0-it.gguf
echo "== done $(date +%H:%M)"
