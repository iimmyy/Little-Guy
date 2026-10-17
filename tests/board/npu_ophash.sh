#!/bin/bash
# Same input N times in fresh processes on the NPU WITHOUT cutting the batch: the NPU fingerprints every op's output
# itself (diagnostic build: hash in the profile record), and the host prints it with GGML_HEXAGON_PROFILE=1. Reports, for
# each run vs run 1, the first op (in execution order) whose output differs, and the ops just before it.
# usage: bash npu_ophash.sh <engine dir> <out dir> <runs>
set -u
ENG=$1 OUT=$2 N=$3; mkdir -p "$OUT"
for i in $(seq 1 $N); do
  (cd $ENG && env LG_FP_OUT=/dev/null LG_FP_MATCH='^$' GGML_HEXAGON_PROFILE=1 LD_LIBRARY_PATH=$ENG/lib ADSP_LIBRARY_PATH=$ENG/lib \
     GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=4096 bin/llama-lg-fingerprint -v -m $HOME/models/e4b-grid4.gguf --device HTP0 -ngl 99 \
     -t 4 -C 0xf0 --cpu-strict 1 -fa on -c 512 -p "Is it okay to have grapefruit with Dayvigo?" -n 20 2>&1 \
     | grep "profile-op" | sed -E 's/.*profile-op ([^:]+): ([^:]+) : .* hash ([0-9a-f]+)$/\1 \2 \3/' > $OUT/ops$i.txt)
  echo "run $i: $(wc -l < $OUT/ops$i.txt) ops"
done
for i in $(seq 2 $N); do
  n=$(paste -d'|' $OUT/ops1.txt $OUT/ops$i.txt | awk -F'|' '$1 != $2 {print NR; exit}')
  if [ -z "$n" ]; then echo "run $i vs run 1: identical"; else
    echo "run $i vs run 1: first different op #$n"; sed -n "$((n > 3 ? n - 3 : 1)),$((n))p" $OUT/ops1.txt | sed 's/^/   run1: /'; sed -n "${n}p" $OUT/ops$i.txt | sed "s/^/   run$i: /"; fi
done
