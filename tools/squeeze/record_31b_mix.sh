#!/bin/bash
# 31B input signals for output matching, Mac CPU (the 8-bit does not fit the GPU): weights read straight from the file
# (--no-repack, no swap), 24 x 512 tokens of the calibration text, one file per distinct input on the external drive
cd "$(dirname "$0")/../.."
A=${1:-/Volumes/Music/LittleGuy/weights/big31b/act_mix}; mkdir -p $A
LG_ACT_DUMP=$A LG_ACT_MAX=12288 .run/lcpp-src/build-mac/bin/llama-imatrix -m .run/weights/big/31b-q8_0.gguf \
  -f .run/bench/calib_mix.txt --parse-special -o $A/../31b-mix.imatrix -c 512 --chunks 24 -ngl 0 -t 10 --no-repack
