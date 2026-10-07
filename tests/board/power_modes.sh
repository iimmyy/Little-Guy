#!/bin/bash
# Power test: 4 load modes, 45 s each, cooled to <=45 C between them.
# Each mode prints "READ THE METER NOW" to the live page once the load has settled.
LIVE=~/spec/live.log
LL="env LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib GGML_HEXAGON_NDEV=0"
MODEL=~/models/gemma-4-E4B_q4_0-it.gguf
temp() { echo $(( $(cat /sys/class/thermal/thermal_zone1/temp) / 1000 )); }
say() { echo "$(date +%H:%M:%S)  $*" | tee -a $LIVE; }
cool() { while [ $(temp) -gt 45 ]; do say "cooling... $(temp) C"; sleep 10; done; }
npuloop() { while [ ! -f /tmp/power_stop ]; do ~/npu/lmhead/bench.sh >/dev/null 2>&1; done; }

mode() {  # mode <number> <name> <commands...>
  local n=$1 name=$2; shift 2
  cool
  rm -f /tmp/power_stop
  say "=================  MODE $n: $name  (starting, wait)  ================="
  ( eval "$@" ) >/dev/null 2>&1 &
  local pid=$!
  sleep 10
  say ">>>>>>>>>>>>>>>>>  MODE $n: READ THE METER NOW (35 s)  <<<<<<<<<<<<<<<<<"
  sleep 35
  touch /tmp/power_stop; pkill -x llama-bench; pkill -x qnn-net-run
  wait $pid 2>/dev/null
  say "MODE $n done, peak temp $(temp) C"
}

echo "=== POWER TEST ===" > $LIVE
mode 2 "brain TALKING (writing words)" "$LL ~/llama/bin/llama-bench -m $MODEL -p 0 -n 4096 -t 4 -r 1"
mode 3 "brain READING (long prompt)"   "$LL ~/llama/bin/llama-bench -m $MODEL -p 2048 -n 0 -t 4 -r 20"
mode 4 "NPU working"                   "npuloop"
mode 5 "EVERYTHING (reading + NPU)"    "npuloop & $LL ~/llama/bin/llama-bench -m $MODEL -p 2048 -n 0 -t 4 -r 20"
say "=== POWER TEST DONE ==="
