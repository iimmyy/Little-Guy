#!/bin/bash
# E4B board levels with the mixed calibration (damping 0.1): Q2_K from the mixed importance matrix, Q4_0 = Google's
# official QAT file (its codes re-picked by matching too); Q3_K already done (e4b-mixmatch-d0.1-Q3_K)
cd "$(dirname "$0")/../.."
[ -f /Volumes/Music/LittleGuy/weights/stageB/e4b-mixim-Q2_K.gguf ] || .run/lcpp-src/build-mac/bin/llama-quantize --pure --imatrix /Volumes/Music/LittleGuy/weights/stageB/e4b-mix.imatrix /Volumes/Music/LittleGuy/weights/masters/qat-bf16.gguf /Volumes/Music/LittleGuy/weights/stageB/e4b-mixim-Q2_K.gguf Q2_K 4 > /dev/null 2>&1
export LG_MATCH_DAMP=0.1
python3 tools/squeeze/match.py build /Volumes/Music/LittleGuy/weights/masters/qat-bf16.gguf /Volumes/Music/LittleGuy/weights/stageB/e4b-mixim-Q2_K.gguf /Volumes/Music/LittleGuy/weights/stageB/act_mix /Volumes/Music/LittleGuy/weights/stageB/e4b-mixmatch-d0.1-Q2_K.gguf > /Volumes/Music/LittleGuy/weights/stageB/match-mix-q2k.log 2>&1 &
python3 tools/squeeze/match.py build /Volumes/Music/LittleGuy/weights/masters/qat-bf16.gguf /Users/immyyyyy/Documents/LittleGuy/.run/weights/brains/gemma-4-E4B_q4_0-it.gguf /Volumes/Music/LittleGuy/weights/stageB/act_mix /Volumes/Music/LittleGuy/weights/stageB/e4b-mixmatch-d0.1-Q4_0.gguf > /Volumes/Music/LittleGuy/weights/stageB/match-mix-q40.log 2>&1 &
wait; echo "== done $(date +%H:%M)"
