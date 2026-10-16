#!/bin/bash
# v3 on E4B (Mac): wait for the matched donors, then harm from matching -> paired calibration -> quality-vs-size curve
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG=tools/squeeze/gemma4-e4b-v3.json; Q="python3 tools/squeeze/squeeze2.py $CFG"; S=/Volumes/Music/LittleGuy/weights/stageB
until [ -f $S/e4b-match-Q2_K.gguf.json ] && [ -f $S/e4b-match-Q3_K.gguf.json ] && [ -f $S/e4b-match-Q4_0.gguf.json ]; do sleep 30; done
echo "== $(date +%H:%M:%S) discover"; $Q discover
echo "== $(date +%H:%M:%S) harm"; $Q harm_from
echo "== $(date +%H:%M:%S) calibrate"; $Q calibrate
echo "== $(date +%H:%M:%S) curve"; $Q curve ${SIZES:-2.8 3.0 3.27 3.5 3.8}
echo "== $(date +%H:%M:%S) done"
