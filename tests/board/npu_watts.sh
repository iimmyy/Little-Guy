#!/bin/bash
# Watts per GB/s: is the NPU a cheaper way to haul his brain than the CPU? One stage at a time, read off the
# USB-C meter: npu_watts.sh <rest|cpu|npu|stop>
#   rest  nothing running (the baseline to subtract)
#   cpu   today's brain talking on the 4 big cores (fork engine in ~/preview, brain mix-21 + dictionary preview)
#   npu   our 2-bit HVX gemv loop (research/npu-hvx, gemv2_loop.so) streaming weights nonstop on the NPU
# The October test's NPU stage (qnn-net-run lm head) sat idle ~80% of the time, so its "under 1 W" wasn't a fair
# match. Both loads here haul weights continuously and print their own GB/s, so watts can be compared per GB/s.
# Holds each stage until "npu_watts.sh stop" or 150 s (heat cap), after cooling to 45 C.
set -u
LIVE=~/spec/live.log
NPU=$HOME/npu/hvx
DIR=${LG_DIR:-$HOME/preview}
BRAIN=${LG_BRAIN:-$HOME/models/mix-21-preview.gguf}
temp() { m=0; for z in 1 2 3 4 7 8 9 10 11 12 13 14; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; echo $(( m / 1000 )); }
say() { echo "$(date +%H:%M:%S)  $*" | tee -a $LIVE; }

if [ "${1:-}" = stop ]; then touch /tmp/watts_stop; exit; fi
case ${1:-} in
  rest) cmd="while [ ! -f /tmp/watts_stop ]; do sleep 1; done" ;;
  cpu)  cmd="cd $DIR && LD_LIBRARY_PATH=$DIR/lib GGML_HEXAGON_NDEV=0 bin/llama-bench -m $BRAIN -ngl 0 -dev none \
               -t 4 -C 0xf0 --cpu-strict 1 -fa 1 -p 0 -n 1300 -r 1 2>/dev/null | grep tg1300 > /tmp/watts_cpu.txt" ;;
  npu)  cmd="cd $NPU && LD_LIBRARY_PATH=$HOME/npu/lmhead/lib ADSP_LIBRARY_PATH='$NPU;$HOME/npu/lmhead/lib' \
               ./run_main_on_hexagon 3 gemv2_loop.so 150 > /dev/null 2>&1" ;;
  *) echo "usage: $0 <rest|cpu|npu|stop>"; exit 1 ;;
esac
if pgrep -x llama-server > /dev/null; then echo "the chat page is running; stop it first"; exit 1; fi

rm -f /tmp/watts_stop /tmp/watts_cpu.txt
while [ "$(temp)" -gt 45 ]; do say "cooling before $1... hottest core $(temp) C"; sleep 10; done
say "=================  $1: starting, wait  ================="
( eval "$cmd" ) &
pid=$!
sleep 10
say ">>>>>>>>  $1: READ THE METER NOW (volts x amps), holding until you're done  <<<<<<<<"
start=$(date +%s)
while [ ! -f /tmp/watts_stop ] && kill -0 $pid 2>/dev/null && [ $(( $(date +%s) - start )) -lt 150 ]; do
  sleep 15; say "   $1 running, $(( $(date +%s) - start )) s, hottest core $(temp) C"
done
touch /tmp/watts_stop; pkill -x llama-bench; pkill -x run_main_on_hexagon
wait $pid 2>/dev/null
case $1 in
  cpu) say "CPU talking: $(awk -F'|' '{print $(NF-1)}' /tmp/watts_cpu.txt 2>/dev/null) words/s (x ~1.8 GB per word = GB/s; empty if stopped early)" ;;
  npu) say "NPU speed every 2 s:"; grep -E "passes|t=" $NPU/gemv2_loop_result.txt | tail -5 | sed 's/^/     /' | tee -a $LIVE ;;
esac
say "$1 done, hottest core $(temp) C"
