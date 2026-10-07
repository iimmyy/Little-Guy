#!/bin/bash
# Board drafter comparison: no drafter, old (full vocab) and clustered drafter, at 3 and 6 guesses.
cd ~/spec && export RUN=$HOME/spec SERVER=$HOME/llama/bin/llama-server MAIN=$HOME/models/gemma-4-E4B_q4_0-it.gguf
export LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib NGL=0 EXTRA="--device none -t 4 -td 4" N_PROMPTS=4 MAX_TOKENS=96
cool() { while [ $(( $(cat /sys/class/thermal/thermal_zone1/temp) / 1000 )) -gt 50 ]; do sleep 5; done; }
cool; echo "== no drafter"; python3 run_spec.py 0 8095
for n in 3 6; do
  for d in old masked; do
    cool; echo "== drafter $d, $n guesses"
    if [ $d = old ]; then export DRAFT=$HOME/models/gemma-4-E4B-assistant-q8_0.gguf; else export DRAFT=$HOME/models/gemma-4-E4B-assistant-masked-q8_0.gguf; fi
    python3 run_spec.py $n 8095
  done
done
