#!/bin/bash
# Heat per GB/s, no meter needed: the same stretch of nonstop weight hauling on the CPU (today's brain talking on the
# 4 big cores) and on the NPU (our 2-bit HVX gemv loop), each after cooling to 45 C. Samples the hottest CPU/NPU
# sensor every 5 s and counts thermal-guard slowdowns, so "same work, how hot" can be compared per GB/s.
# The chat page must be stopped first. Settings: SECS (default 135), LG_DIR, LG_BRAIN.
set -u
SECS=${SECS:-135}
DIR=${LG_DIR:-$HOME/preview}
BRAIN=${LG_BRAIN:-$HOME/models/mix-21-preview.gguf}
NPU=$HOME/npu/hvx
hot() { m=0; for z in 1 2 3 4 7 8 9 10 11 12 13 14; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; echo $(( m / 1000 )); }
nsp() { for z in /sys/class/thermal/thermal_zone*; do [[ $(cat $z/type) == nspss* ]] && echo $(( $(cat $z/temp) / 1000 )); done | sort -n | tail -1; }
if pgrep -x llama-server > /dev/null; then echo "the chat page is running; stop it first"; exit 1; fi

watch() {   # name, pid: sample until the pid exits
  local peak=0 sum=0 n=0 npeak=0 t
  while kill -0 $2 2>/dev/null; do
    t=$(hot); sum=$((sum + t)); n=$((n + 1)); [ $t -gt $peak ] && peak=$t
    t=$(nsp); [ -n "$t" ] && [ $t -gt $npeak ] && npeak=$t
    sleep 5
  done
  echo "$1: hottest sensor peak ${peak} C, average $((sum / (n > 0 ? n : 1))) C, NPU sensor peak ${npeak} C"
}

for job in cpu npu; do
  while [ "$(hot)" -gt 45 ]; do sleep 5; done
  start=$(date "+%Y-%m-%d %H:%M:%S")
  if [ $job = cpu ]; then
    (cd $DIR && LD_LIBRARY_PATH=$DIR/lib GGML_HEXAGON_NDEV=0 timeout $((SECS + 30)) bin/llama-bench -m $BRAIN -ngl 0 -dev none \
       -t 4 -C 0xf0 --cpu-strict 1 -fa 1 -p 0 -n $((SECS * 9)) -r 1 2>/dev/null | grep tg > /tmp/heat_cpu.txt) &
  else
    (cd $NPU && LD_LIBRARY_PATH=$HOME/npu/lmhead/lib ADSP_LIBRARY_PATH="$NPU;$HOME/npu/lmhead/lib" \
       ./run_main_on_hexagon 3 gemv2_loop.so $SECS > /dev/null 2>&1) &
  fi
  watch $job $!
  echo "  guard slowdowns: $(journalctl -t thermal-guard --since "$start" --no-pager | grep -c slowing)"
  if [ $job = cpu ]; then
    w=$(awk -F'|' '{print $(NF-1)}' /tmp/heat_cpu.txt | awk '{print $1}')
    echo "  CPU: $w words/s = $(awk -v w="$w" 'BEGIN {printf "%.1f", w * 1.783}') GB/s"
  else
    echo "  NPU: $(grep passes $NPU/gemv2_loop_result.txt)"
  fi
done
