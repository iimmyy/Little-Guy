#!/bin/bash
# Toward 8 tokens/s on the NPU: one-question drafter diagnosis (1 guess, drafter on the NPU with one CPU worker) under
# host-side settings that need no rebuild: batch size (smaller batches let the NPU start while the host is still
# writing the rest: pipelining) and host polling for the NPU's answer. Then the decode timing breakdown.
# usage: bash npu_8_sweep.sh <engine dir with LG_DECODE_TIMING build> <out dir>
set -u
ENG=$1 OUT=$2; mkdir -p "$OUT"
R="draft1|-ngld 99 -devd HTP0 -td 1 -Cd 0x80 --cpu-strict-draft 1"
for b in ${BATCHES:-4096 1024 256 64}; do
  echo "== batch $b";            OPBATCH=$b RUNS="$R" bash ~/spec/npu_draft_diag.sh $ENG $OUT/batch$b | grep -E "words/s|drafter time"
done
echo "== batch 4096 + host polling"; OPBATCH=4096 XENV="GGML_HEXAGON_OPPOLL=1" RUNS="$R" bash ~/spec/npu_draft_diag.sh $ENG $OUT/poll | grep -E "words/s|drafter time"
echo "== timing breakdown (batch 4096)"; OPBATCH=4096 XENV="LG_DECODE_TIMING=1" RUNS="$R" bash ~/spec/npu_draft_diag.sh $ENG $OUT/timing | grep -E "words/s"
echo SWEEP_DONE
