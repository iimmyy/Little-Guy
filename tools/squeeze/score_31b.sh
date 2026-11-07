#!/bin/bash
# 31B on the Mac, KL and same-top vs the 31B Q8_0: public held-out WikiText half (2048 x 16), held-out chat (512 x 32),
# and a private unpublished paper (512 x 32, .run/eval-private, never leaves this machine). GPU when the file fits
# (smaller micro-batch keeps long rulers under the GPU limit), CPU otherwise (CPU=1).
cd "$(dirname "$0")/../.."
B=.run/lcpp-src/build-mac/bin
dev() { [ -n "${CPU:-}" ] && echo "-ngl 0 -t 10 --no-repack" || echo "-ngl 99 -ub 256"; }
kl() {  # name model text base ctx chunks env...
  local n=$1 m=$2 t=$3 b=$4 c=$5 k=$6; shift 6
  r=$(env LG_DICT_PREVIEW=0 "$@" $B/llama-perplexity -m $m -f $t -c $c --chunks $k $(dev) --kl-divergence-base $b \
      --kl-divergence 2>&1 | grep -E "^Mean    KLD|^Same top p|Insufficient|failed" | sed -E 's/Mean +KLD: +//; s/Same top p: +//' | tr '\n' ' ')
  printf "  %-8s %s\n" $n "$r"
}
for m in "$@"; do
  echo "$(basename $m) ($(stat -f %z $m) bytes)   KL ± se   same-top ± se"
  kl wiki $m .run/bench/wiki.eval.txt .run/bench/31b_q8_wiki.kl 2048 16
  kl heldout $m .run/bench/chat_heldout.txt .run/bench/31b_q8_heldout.kl 512 32 LG_PPL_SPECIAL=1
  kl paper $m .run/eval-private/paper.md .run/eval-private/31b_q8_paper.kl 512 32
done
