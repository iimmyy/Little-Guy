#!/bin/bash
# 31B 8-bit reference answers on the Mac (CPU, weights straight from the file): held-out chat ruler, then the private
# paper (stays in .run/eval-private). Same chunking as the candidates are scored with.
cd "$(dirname "$0")/../.."
B=.run/lcpp-src/build-mac/bin; M=.run/weights/big/31b-q8_0.gguf; T=${THREADS:-6}
[ -f .run/bench/31b_q8_heldout.kl ] || LG_PPL_SPECIAL=1 $B/llama-perplexity -m $M -f .run/bench/chat_heldout.txt -c 512 --chunks 32 \
    -ngl 0 -t $T --no-repack --kl-divergence-base .run/bench/31b_q8_heldout.kl 2>&1 | grep -E "Final|seconds per pass"
[ -f .run/eval-private/31b_q8_paper.kl ] || $B/llama-perplexity -m $M -f .run/eval-private/aeviant-v3.md -c 512 --chunks 32 \
    -ngl 0 -t $T --no-repack --kl-divergence-base .run/eval-private/31b_q8_paper.kl 2>&1 | grep -E "Final|seconds per pass"
echo "== done $(date +%H:%M)"
