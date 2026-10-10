#!/bin/bash
# Both drivers at once: does the NPU's 2-bit gemv and the CPU's 2-bit talking each keep their solo speed
# when they run together (separate roads to memory) or slow each other down (one shared road)?
#   A: NPU alone (gemv2_loop.so)   B: CPU alone (llama-bench writing)   C: both together
# In quiet mode it lifts only the CPU clock cap for the test (lights stay off) and puts it back at the end.
NPU=$HOME/npu/hvx; MODEL=$HOME/models/${MODEL:-qat-Q2_K-vip4-f16proj}.gguf
export ADSP_LIBRARY_PATH="$NPU;$HOME/npu/lmhead/lib"
hottest() { m=0; for z in 1 2 3 4 7 8 9 10 11 12 13 14; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; echo $(( m / 1000 )); }
cool() { while [ $(hottest) -gt 50 ]; do sleep 5; done; }
npu() { (cd $NPU && LD_LIBRARY_PATH=$HOME/npu/lmhead/lib ./run_main_on_hexagon 3 gemv2_loop.so $1 >/dev/null 2>&1; grep -E "passes|t=" gemv2_loop_result.txt); }
cpu() { (cd ~/llama && LD_LIBRARY_PATH=lib GGML_HEXAGON_NDEV=0 bin/llama-bench -m $MODEL -t 4 -p 0 -n 128 -r 1 --device none -ngl 0 2>/dev/null | grep tg128 | awk -F'|' '{print $(NF-1) " tok/s"}'); }

caps=(); for p in 0 4 7; do f=/sys/devices/system/cpu/cpufreq/policy$p; caps+=($(cat $f/scaling_max_freq)); cat $f/cpuinfo_max_freq | sudo -n tee $f/scaling_max_freq >/dev/null; done
echo "CPU caps lifted for the test (were ${caps[*]} kHz)"

cool; echo "== A: NPU alone (12 s)";  echo "   $(npu 12)"
cool; echo "== B: CPU alone (his 2-bit brain writing 128 words)"; echo "   $(cpu)"
cool; echo "== C: both together"
start=$(date +%s); npu 36 > /tmp/npu_both.txt &
sleep 6                      # let the NPU program pack its weights and settle before the CPU starts
c0=$(( $(date +%s) - start )); r=$(cpu); c1=$(( $(date +%s) - start ))
echo "   CPU while NPU runs: $r   (CPU ran from about t=${c0}s to t=${c1}s; it loads the brain for ~3 s first)"
wait; echo "   NPU speed every 2 s (read the stretch while the CPU ran):"; sed 's/^/     /' /tmp/npu_both.txt
echo "   peak $(hottest) C"

i=0; for p in 0 4 7; do echo ${caps[$i]} | sudo -n tee /sys/devices/system/cpu/cpufreq/policy$p/scaling_max_freq >/dev/null; i=$((i+1)); done
echo "CPU caps restored"
