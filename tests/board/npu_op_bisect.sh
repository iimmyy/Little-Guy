#!/bin/bash
# Which NPU operation spoils the brain? Perplexity on the same short text with one kind of operation left to the CPU
# (GGML_HEXAGON_OPFILTER) at a time; the 16-bit brain reads ~23 here, the board CPU 23.05. Runs ON the board.
# usage: bash npu_op_bisect.sh <engine dir> <brain> <text> <out dir> [filter ...]
set -u
ENG=$1 BRAIN=$2 TEXT=$3 OUT=$4; shift 4; mkdir -p "$OUT"
for f in "$@"; do
  name=$(echo "$f" | tr -c 'A-Za-z0-9_\n' '_')
  (cd $ENG && env LD_LIBRARY_PATH=$ENG/lib ADSP_LIBRARY_PATH=$ENG/lib GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=4096 \
     ${f:+GGML_HEXAGON_OPFILTER="$f"} bin/llama-perplexity -m $BRAIN -f $TEXT --device HTP0 -ngl 99 -t 4 -C 0xf0 --cpu-strict 1 \
     -fa on -c 128 -b 128 --chunks 4 > $OUT/filter-$name.log 2>&1)
  echo "left to the CPU: ${f:-nothing}  ->  $(grep -oE 'Final estimate: PPL = [0-9.]+' $OUT/filter-$name.log || tail -1 $OUT/filter-$name.log)"
done
