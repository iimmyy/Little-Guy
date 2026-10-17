#!/bin/bash
# Run the same input N times in fresh processes on the NPU, fingerprinting every operation's output
# (llama-lg-fingerprint), then report for each run vs run 1 the first operation whose output differs.
# usage: [LG_FP_MATCH=regex] bash npu_fingerprint.sh <engine dir> <out dir> <runs> [extra args]
set -u
ENG=$1 OUT=$2 N=$3; shift 3; mkdir -p "$OUT"
for i in $(seq 1 $N); do
  (cd $ENG && env ${OPFILTER:+GGML_HEXAGON_OPFILTER=$OPFILTER} LG_FP_OUT=$OUT/run$i.txt LD_LIBRARY_PATH=$ENG/lib ADSP_LIBRARY_PATH=$ENG/lib GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=${OPBATCH:-4096} \
     bin/llama-lg-fingerprint -m $HOME/models/e4b-grid4.gguf --device HTP0 -ngl 99 -t 4 -C 0xf0 --cpu-strict 1 -fa on -c 512 \
     -p "Is it okay to have grapefruit with Dayvigo?" -n 20 "$@" > $OUT/run$i.log 2>&1)
  grep -c dsp-rsp $OUT/run$i.log | xargs -I{} echo "run $i: $(wc -l < $OUT/run$i.txt) lines, NPU failure reports {}, reply: $(grep -m1 'reply:' $OUT/run$i.log | cut -c1-90)"
done
for i in $(seq 2 $N); do
  first=$(paste -d'|' $OUT/run1.txt $OUT/run$i.txt | awk -F'|' '$1 != $2 {print NR": "$1"  <->  "$2; exit}')
  echo "run $i vs run 1: ${first:-identical}"
done
