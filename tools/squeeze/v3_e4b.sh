#!/bin/bash
# v3 on E4B for the board: wait for the matched Q2_K/Q3_K/Q4_0 donors (mixed calibration), harm = error left after
# matching -> paired calibration -> allocation curve. Sizes: 7-group budget + 2.951 GB frame (per word +0.308 GB).
cd "$(dirname "$0")/../.."
CFG=tools/squeeze/gemma4-e4b-v3.json; Q="python3 tools/squeeze/squeeze2.py $CFG"; S=/Volumes/Music/LittleGuy/weights/stageB
until [ -f $S/e4b-mixmatch-d0.1-Q2_K.gguf.json ] && [ -f $S/e4b-mixmatch-d0.1-Q3_K.gguf.json ] && [ -f $S/e4b-mixmatch-d0.1-Q4_0.gguf.json ]; do sleep 30; done
echo "== $(date +%H:%M:%S) discover"; $Q discover
echo "== $(date +%H:%M:%S) harm"; $Q harm_from
echo "== $(date +%H:%M:%S) calibrate"; $Q calibrate
echo "== $(date +%H:%M:%S) curve"; $Q curve ${SIZES:-4.42 4.54 4.68 4.84 5.04}
echo "== $(date +%H:%M:%S) done"
