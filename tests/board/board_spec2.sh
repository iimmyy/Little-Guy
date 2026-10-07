#!/bin/bash
# Fair old vs clustered drafter comparison (3 guesses), each run starting at <= 45 C.
cd ~/spec && export RUN=$HOME/spec SERVER=$HOME/llama/bin/llama-server MAIN=$HOME/models/gemma-4-E4B_q4_0-it.gguf
export LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib NGL=0 EXTRA="--device none -t 4 -td 4" N_PROMPTS=4 MAX_TOKENS=96
cool() { while [ $(( $(cat /sys/class/thermal/thermal_zone1/temp) / 1000 )) -gt 45 ]; do sleep 5; done; }
for d in masked old masked old; do
  cool; echo "== drafter $d, 3 guesses, start $(( $(cat /sys/class/thermal/thermal_zone1/temp) / 1000 ))C"
  if [ $d = old ]; then export DRAFT=$HOME/models/gemma-4-E4B-assistant-q8_0.gguf; else export DRAFT=$HOME/models/gemma-4-E4B-assistant-masked-q8_0.gguf; fi
  python3 run_spec.py 3 8096
  echo "   guard steps this run: $(journalctl -t thermal-guard --since "-6 min" --no-pager -o cat | grep -c slowing)"
done
