#!/bin/bash
# E4B scorecard: community mixes made with the same (mixed) importance matrix as ours, plus our brains, on every ruler
cd "$(dirname "$0")/../.."
S=/Volumes/Music/LittleGuy/weights/stageB; REF=/Volumes/Music/LittleGuy/weights/masters/qat-bf16.gguf; B=.run/lcpp-src/build-mac/bin
for t in Q2_K IQ3_XXS Q3_K_S IQ3_M Q3_K_M Q4_K_S; do
  [ -f $S/e4b-mixcom-$t.gguf ] || $B/llama-quantize --imatrix $S/e4b-mix.imatrix $REF $S/e4b-mixcom-$t.gguf $t 4 > /dev/null 2>&1
done
python3 tools/squeeze/bytes_per_word.py /Volumes/Music/LittleGuy/weights/experiments/down-mix/mix-21-preview.gguf $S/e4b-mixim-Q3_K.gguf \
  $S/e4b-mixmatch-d0.1-Q3_K.gguf $S/e4b-mixcom-*.gguf /Volumes/Music/LittleGuy/weights/experiments/qat-Q4_0.gguf
tools/squeeze/score_e4b.sh /Volumes/Music/LittleGuy/weights/experiments/down-mix/mix-21-preview.gguf $S/e4b-mixim-Q3_K.gguf \
  $S/e4b-mixmatch-d0.1-Q3_K.gguf $S/e4b-mixcom-*.gguf /Volumes/Music/LittleGuy/weights/experiments/qat-Q4_0.gguf
echo "== done $(date +%H:%M)"
