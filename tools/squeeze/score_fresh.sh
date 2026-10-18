#!/bin/bash
# Fresh E4B rulers (raw model, no persona), measured against Google's QAT E4B in bf16 (the yardstick):
#   wiki    WikiText-2 raw test (the standard public text), 64 windows of 512, KL + same-top vs the yardstick
#   heldout chat the yardstick wrote on held-out prompts (make_selfchat.py eval), 64 windows of 512, chat markers on
#   ppl     perplexity on the whole WikiText-2 test, the number papers quote
# usage: tools/squeeze/score_fresh.sh base               (once: writes the yardstick's logits, needs the bf16 file)
#        tools/squeeze/score_fresh.sh <brain.gguf> [...]
cd "$(dirname "$0")/../.."
# LG_BIN / LG_W / LG_WIKI point it at another machine (AWS: the CUDA build, ~/work/fresh, ~/work/bench)
B=${LG_BIN:-.run/lcpp-src/build-mac/bin}; W=${LG_W:-/Volumes/Music/LittleGuy/weights/fresh}
# LG_REF: the 16-bit yardstick (default E4B); its logits are kept next to it (ref_wiki.kl, ref_heldout.kl in LG_KLDIR)
REF=${LG_REF:-$W/e4b-qat-bf16.gguf}; K=${LG_KLDIR:-$W}; WIKI=${LG_WIKI:-.run/bench/wiki.test.raw}; CHAT=$W/chat_heldout.txt
run() {  # model text ctx chunks extra-args... (env: LG_PPL_SPECIAL)
  local m=$1 t=$2 c=$3 k=$4; shift 4
  LG_DICT_PREVIEW=0 $B/llama-perplexity -m $m -f $t -c $c --chunks $k -ngl 99 "$@" 2>&1
}
if [ "$1" = base ]; then
  run $REF $WIKI 512 64 --kl-divergence-base $K/ref_wiki.kl | grep -E "Final estimate|error" ; echo "wiki base done"
  LG_PPL_SPECIAL=1 run $REF $CHAT 512 64 --kl-divergence-base $K/ref_heldout.kl | grep -E "Final estimate|error"; echo "heldout base done"
  run $REF $WIKI 512 999999 | grep -E "Final estimate|error"
  exit
fi
for m in "$@"; do
  echo "$(basename $m) ($(wc -c < $m | tr -d ' ') bytes)"
  for r in wiki heldout; do
    [ $r = heldout ] && export LG_PPL_SPECIAL=1 || unset LG_PPL_SPECIAL
    t=$WIKI; [ $r = heldout ] && t=$CHAT
    printf "  %-8s %s\n" $r "$(run $m $t 512 64 --kl-divergence-base $K/ref_$r.kl --kl-divergence |
      grep -E "^Mean    KLD|^Same top p" | sed -E 's/Mean +KLD: +/KL /; s/Same top p: +/same-top /' | tr '\n' ' ')"
  done
  unset LG_PPL_SPECIAL
  printf "  %-8s %s\n" ppl "$(run $m $WIKI 512 999999 | grep -E 'Final estimate' | sed 's/.*Final estimate: //')"
done
