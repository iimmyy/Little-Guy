#!/bin/bash
# Is a new NPU engine bit-exact with the old one on the real brain? Both read the same text on the NPU in small
# batches (5, 6, 7 words: the kernels' groups of 4 leave 1, 2, 3 over); the new engine's word odds are compared with
# the old one's (drift = KL, same top word). Exact means drift 0 and same top 100%. Runs ON the board.
# usage: [UBS="5 6 7"] bash npu_exact.sh <old engine dir> <new engine dir> <brain> <text> <out dir>
set -u
OLD=$1 NEW=$2 BRAIN=$3 TEXT=$4 OUT=$5; mkdir -p "$OUT"
FLAGS="--device HTP0 -ngl 99 -t 4 -C 0xf0 --cpu-strict 1 -fa on -c 128 -b 128 --chunks 4"
run() { (cd $1 && env LD_LIBRARY_PATH=$1/lib ADSP_LIBRARY_PATH=$1/lib GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=4096 \
        bin/llama-perplexity -m $BRAIN -f $TEXT $FLAGS "${@:2}"); }
for ub in ${UBS:-5 6 7}; do
  run $OLD -ub $ub --kl-divergence-base $OUT/old-ub$ub.kld > $OUT/old-ub$ub.log 2>&1
  run $NEW -ub $ub --kl-divergence-base $OUT/old-ub$ub.kld --kl-divergence > $OUT/new-ub$ub.log 2>&1
  echo "batch $ub: $(grep -E '^Mean    KLD|^Same top p|^Maximum KLD' $OUT/new-ub$ub.log | tr -s ' ' | tr '\n' ' ')"
  rm -f $OUT/old-ub$ub.kld
done
echo NPU_EXACT_DONE
