#!/bin/bash
# Does the drafter speed up his current brains? Each brain with no drafter and with the masked drafter
# guessing 3 words ahead, on the board's CPU, every run starting with the hottest core at or below 45 C.
cd ~/spec && export RUN=$HOME/spec SERVER=$HOME/llama/bin/llama-server
export LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib GGML_HEXAGON_NDEV=0
export NGL=0 EXTRA="--device none -t 4 -td 4" N_PROMPTS=4 MAX_TOKENS=96
export DRAFT=$HOME/models/gemma-4-E4B-assistant-masked-q8_0.gguf
hottest() { m=0; for z in 1 2 3 4 7 8 9 10 11 12 13 14; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; echo $(( m / 1000 )); }
cool() { while [ $(hottest) -gt 45 ]; do sleep 5; done; }
for m in ${MODELS:-qat-Q4_0-f16proj qat-Q2_K-vip4-f16proj}; do
  export MAIN=$HOME/models/$m.gguf
  cool; echo "== $m, no drafter";            python3 run_spec.py 0 8095
  cool; echo "== $m, drafter 3 words ahead"; python3 run_spec.py 3 8095
  echo "   peak $(hottest) C"
done
