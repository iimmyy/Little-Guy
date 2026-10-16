#!/bin/bash
# Little Guy's brains on every E4B ruler (Mac GPU), KL and same-top vs E4B Q8_0:
#   persona (private persona chat, one window), chat, raw (squeeze-e4b evals), wiki (public held-out half)
# usage: tools/squeeze/score_e4b.sh <brain.gguf> [...]
cd "$(dirname "$0")/../.."
B=.run/lcpp-src/build-mac/bin; E=.run/weights/squeeze-e4b; S=/Volumes/Music/LittleGuy/weights/stageB
kl() {  # name model text base ctx chunks env...
  local n=$1 m=$2 t=$3 b=$4 c=$5 k=$6; shift 6
  r=$(env LG_DICT_PREVIEW=0 "$@" $B/llama-perplexity -m $m -f $t -c $c --chunks $k -ngl 99 --kl-divergence-base $b \
      --kl-divergence 2>&1 | grep -E "^Mean    KLD|^Same top p" | sed -E 's/Mean +KLD: +//; s/Same top p: +//' | tr '\n' ' ')
  printf "  %-8s %s\n" $n "$r"
}
for m in "$@"; do
  echo "$(basename $m) ($(stat -f %z $m) bytes)   KL ± se   same-top ± se"
  kl persona $m $E/persona_eval.txt $E/e4b_kl_persona.bin 3072 1 LG_PPL_SPECIAL=1 LG_PPL_ONE_WINDOW=1
  kl chat $m $E/chat_eval.txt $E/e4b_kl_chat.bin 512 4 LG_PPL_SPECIAL=1
  kl raw $m docs/2026-09-28.md $E/e4b_kl_base.bin 512 4
  kl wiki $m .run/bench/wiki.eval.txt $S/e4b_q8_wiki512.kl 512 32
done
