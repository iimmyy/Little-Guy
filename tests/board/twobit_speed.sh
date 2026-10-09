#!/bin/bash
# Is 2-bit actually faster on the board? Writing speed (decode) and reading speed (prefill), CPU only,
# each model starting with the hottest core at or below 45 C. Lines also go to Little Guy's feed.
cd ~/llama
export LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib GGML_HEXAGON_NDEV=0
FEED=~/spec/live.log
hottest() { m=0; for z in 1 2 3 4 7 8 9 10 11 12 13 14; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; echo $(( m / 1000 )); }
say() { echo "$(date +%H:%M:%S)  $*" | tee -a $FEED; }
say "=== 2-bit speed test (CPU, 4 threads) ==="
for m in ${MODELS:-gemma-4-E4B_q4_0-it.gguf f16-Q2_K.gguf f16-Q2_K-dict4.gguf}; do
  while [ $(hottest) -gt 45 ]; do sleep 10; done
  out=$(bin/llama-bench -m ~/models/$m -t 4 -p 128 -n 64 -r 2 --device none -ngl 0 2>/dev/null | grep -E "pp128|tg64")
  pp=$(echo "$out" | awk -F'|' '/pp128/ {print $(NF-1)}' | xargs)
  tg=$(echo "$out" | awk -F'|' '/tg64/ {print $(NF-1)}' | xargs)
  say "$(printf '%-28s writing %-14s reading %-14s (peak %s C)' $m "$tg tok/s" "$pp tok/s" $(hottest))"
done
