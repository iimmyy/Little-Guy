#!/bin/bash
# Why does llama.cpp's NPU backend crash on v68 (dspqueue_read 0x2e = AEE_EINTERRUPTED = the NPU side died)?
# Smallest tests first. Each line: what we change -> does a single op survive and match the CPU?
cd ~/llama
export LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib
FEED=~/spec/live.log
run() {  # run <label> <env...> -- <test-backend-ops args>
  local label=$1; shift
  echo "=== $label" | tee -a $FEED
  timeout 120 env "$@" > /tmp/npu_debug.out 2>&1; local code=$?
  grep -E "OK|FAIL|passed|failed|dspqueue|abort|error|Backend|HMX|vmem" /tmp/npu_debug.out | tail -8 | tee -a $FEED
  echo "    exit $code" | tee -a $FEED
}
run "1. simple ADD, defaults"                       bin/test-backend-ops -b HTP0 -o ADD
run "2. simple ADD, HMX off"     GGML_HEXAGON_USE_HMX=0 bin/test-backend-ops -b HTP0 -o ADD
run "3. MUL_MAT, defaults"                          bin/test-backend-ops -b HTP0 -o MUL_MAT
run "4. MUL_MAT, HMX off"        GGML_HEXAGON_USE_HMX=0 bin/test-backend-ops -b HTP0 -o MUL_MAT
run "5. MUL_MAT, HMX off, measured vmem" GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_VMEM=0 bin/test-backend-ops -b HTP0 -o MUL_MAT
