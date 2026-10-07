#!/bin/bash
# Power test, one stage at a time: power_stage.sh <2|3|4|5>
# Holds the load until ~/spec/power_stage.sh stop (or 150 s, a heat safety cap).
LIVE=~/spec/live.log
LL="env LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib GGML_HEXAGON_NDEV=0"
MODEL=~/models/gemma-4-E4B_q4_0-it.gguf
# hottest CPU core (zones 1-4, 7-14); one calm sensor can read 15 C under the hottest
temp() { m=0; for z in 1 2 3 4 7 8 9 10 11 12 13 14; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; echo $(( m / 1000 )); }
say() { echo "$(date +%H:%M:%S)  $*" | tee -a $LIVE; }
npuloop() { while [ ! -f /tmp/power_stop ]; do ~/npu/lmhead/bench.sh >/dev/null 2>&1; done; }

if [ "$1" = stop ]; then touch /tmp/power_stop; exit; fi

case $1 in
  2) name="brain TALKING (writing words)"; cmd="$LL ~/llama/bin/llama-bench -m $MODEL -p 0 -n 8192 -t 4 -r 1" ;;
  3) name="brain READING (long prompt)";   cmd="$LL ~/llama/bin/llama-bench -m $MODEL -p 2048 -n 0 -t 4 -r 50" ;;
  4) name="NPU working";                   cmd="npuloop" ;;
  5) name="EVERYTHING (reading + NPU)";    cmd="npuloop & $LL ~/llama/bin/llama-bench -m $MODEL -p 2048 -n 0 -t 4 -r 50" ;;
  *) echo "usage: $0 <2|3|4|5|stop>"; exit 1 ;;
esac

rm -f /tmp/power_stop
while [ $(temp) -gt 45 ]; do
  [ -f /tmp/power_stop ] && { say "mode $1 cancelled while cooling"; exit; }
  say "cooling before mode $1... hottest core $(temp) C"; sleep 10
done
say "=================  MODE $1: $name  (starting, wait)  ================="
( eval "$cmd" ) >/dev/null 2>&1 &
pid=$!
sleep 10
say ">>>>>>>>  MODE $1: READ THE METER NOW (holding until you're done)  <<<<<<<<"
start=$(date +%s)
while [ ! -f /tmp/power_stop ] && [ $(( $(date +%s) - start )) -lt 150 ]; do
  sleep 15; say "   mode $1 still running, $(( $(date +%s) - start )) s, hottest core $(temp) C"
done
touch /tmp/power_stop; pkill -x llama-bench; pkill -x qnn-net-run
wait $pid 2>/dev/null
say "MODE $1 done, temp $(temp) C"
