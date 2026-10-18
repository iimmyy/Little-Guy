#!/bin/bash
# Drift with its tail: mean KL plus the worst tokens (99%, 99.9%, max) and same-top, per text, full logs kept.
# Answers change at the hard tokens (where the model is torn between two words); the mean hides them, the tail shows them.
# Texts (LG_TEXTS, "name=file:chunks ..."): same windows as score_fresh.sh (512 tokens; chat markers parsed for all but
# wiki), so wiki/heldout reuse score_fresh.sh's reference logits. Ruler label checked like score_fresh.sh.
# usage: LG_REF=... LG_KLDIR=... LG_TEXTS="wiki=f:64 heldout=f:64 own=f:all" score_tail.sh base | <brain.gguf> [...]
cd "$(dirname "$0")/../.."
B=${LG_BIN:-.run/lcpp-src/build-mac/bin}; REF=$LG_REF; K=$LG_KLDIR
RULER="bin=$(cd $B && pwd) f32sums=${GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F:+1} ref=$(basename $REF)"
run() {  # model text chunks name extra...
  local m=$1 t=$2 k=$3 n=$4; shift 4; [ "$k" = all ] && k=999999
  if [ $n = wiki ]; then unset LG_PPL_SPECIAL; else export LG_PPL_SPECIAL=1; fi
  LG_DICT_PREVIEW=0 $B/llama-perplexity -m $m -f $t -c 512 --chunks $k -ngl 99 "$@" 2>&1
}
if [ "$1" = base ]; then
  [ -f $K/ruler.txt ] && [ "$(cat $K/ruler.txt)" != "$RULER" ] && { echo "RULER MISMATCH: [$(cat $K/ruler.txt)] vs [$RULER]"; exit 1; }
  echo "$RULER" > $K/ruler.txt
  for spec in $LG_TEXTS; do n=${spec%%=*}; f=${spec#*=}; k=${f##*:}; f=${f%:*}
    [ -f $K/ref_$n.kl ] && { echo "$n: reference exists"; continue; }
    run $REF $f $k $n --kl-divergence-base $K/ref_$n.kl > $K/ref_$n.log; grep -E "Final estimate" $K/ref_$n.log; echo "$n base done"
  done; exit
fi
[ "$(cat $K/ruler.txt 2>/dev/null)" != "$RULER" ] && { echo "RULER MISMATCH or missing: [$(cat $K/ruler.txt 2>/dev/null)] vs [$RULER]"; exit 1; }
mkdir -p $K/tail
for m in "$@"; do
  echo "$(basename $m)"
  for spec in $LG_TEXTS; do n=${spec%%=*}; f=${spec#*=}; k=${f##*:}; f=${f%:*}
    L=$K/tail/$(basename $m .gguf).$n.log
    run $m $f $k $n --kl-divergence-base $K/ref_$n.kl --kl-divergence > $L
    printf "  %-8s %s\n" $n "$(grep -E "^Mean +KLD|^99\.0% +KLD|^99\.9% +KLD|^Maximum KLD|^Same top p" $L |
      sed -E 's/^Mean +KLD: +/mean /; s/^99\.0% +KLD: +/p99 /; s/^99\.9% +KLD: +/p99.9 /; s/^Maximum KLD: +/max /; s/^Same top p: +/same-top /' | tr '\n' ' ')"
  done
done
