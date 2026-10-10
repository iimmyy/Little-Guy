#!/bin/bash
# Reading (prefill) speed with the NPU helping: CPU alone, then 10, 21 and 42 layers' multiplications on
# the NPU. Only MUL_MAT goes to the NPU (the one op it does correctly); everything else stays on the CPU.
# Every run starts with the hottest core at or below 42 C and reports whether the CPU throttled.
cd ~/llama
export LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib
MODEL=$HOME/models/${MODEL:-qat-Q4_0-f16proj}.gguf
hottest() { m=0; for z in 1 2 3 4 7 8 9 10 11 12 13 14; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; echo $(( m / 1000 )); }
cool() { while [ $(hottest) -gt 42 ]; do sleep 5; done; }
throttle() { for p in 0 4 7; do c=$(cat /sys/devices/system/cpu/cpufreq/policy$p/scaling_max_freq); h=$(cat /sys/devices/system/cpu/cpufreq/policy$p/cpuinfo_max_freq); [ $c -lt $h ] && { echo "   THROTTLED (cluster $p at $((c/1000)) of $((h/1000)) MHz), peak $(hottest) C"; return; }; done; echo "   clean (no throttling), peak $(hottest) C"; }
bench() { bin/llama-bench -m $MODEL -t 4 -p 64,128 -n 0 -r 1 "$@" 2>&1 | grep -E "pp64|pp128|error|failed" | awk -F'|' '{print "   " $(NF-2) ":" $(NF-1) " tok/s"}'; }

cool; echo "== CPU alone"
GGML_HEXAGON_NDEV=0 bench -dev none -ngl 0; throttle
for n in 10 21 42; do
  cool; echo "== NPU doing the multiplications for $n of 42 layers"
  GGML_HEXAGON_OPFILTER='(?!(MUL_MAT|NONE|VIEW|RESHAPE|PERMUTE|TRANSPOSE)$).*' bench -dev HTP0 -ngl $n -nkvo 1; throttle
done
