#!/bin/bash
# Output matching on E4B with representative calibration (self-generated chat + WikiText calibration half, chat markers
# read as control tokens): importance matrix + input rows, a Q3_K donor from that matrix, matched at two dampings.
set -uo pipefail
cd "$(dirname "$0")/../.."
S=/Volumes/Music/LittleGuy/weights/stageB; REF=/Volumes/Music/LittleGuy/weights/masters/qat-bf16.gguf; B=.run/lcpp-src/build-mac/bin
log() { echo "== $(date +%H:%M:%S) $*"; }
mkdir -p $S/act_mix
[ -f $S/e4b-mix.imatrix ] || { log record; LG_ACT_DUMP=$S/act_mix LG_ACT_MAX=16384 $B/llama-imatrix -m $S/e4b-q8_0.gguf \
    -f .run/bench/calib_mix.txt --parse-special -o $S/e4b-mix.imatrix -c 512 --chunks 80 -ngl 99 > $S/imatrix-mix.log 2>&1; }
[ -f $S/e4b-mixim-Q3_K.gguf ] || { log donor; $B/llama-quantize --pure --imatrix $S/e4b-mix.imatrix $REF $S/e4b-mixim-Q3_K.gguf Q3_K > /dev/null 2>&1; }
for dp in 0.01 0.1; do
  [ -f $S/e4b-mixmatch-d$dp-Q3_K.gguf.json ] || { log "match damp $dp"; LG_MATCH_DAMP=$dp python3 tools/squeeze/match.py build $REF \
      $S/e4b-mixim-Q3_K.gguf $S/act_mix $S/e4b-mixmatch-d$dp-Q3_K.gguf > $S/match-mix-d$dp.log 2>&1 & }
done
wait; log scoring
tools/squeeze/score_e4b.sh $S/e4b-mixim-Q3_K.gguf $S/e4b-mixmatch-d0.01-Q3_K.gguf $S/e4b-mixmatch-d0.1-Q3_K.gguf
log done
