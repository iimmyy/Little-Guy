#!/bin/bash
# Handoff cost: per batch of NPU work, total round trip vs NPU compute time (profiler mode 1).
cd ~/llama
export LD_LIBRARY_PATH=$HOME/llama/lib ADSP_LIBRARY_PATH=$HOME/llama/lib
export GGML_HEXAGON_OPFILTER='(?!(MUL_MAT|NONE|VIEW|RESHAPE|PERMUTE|TRANSPOSE)$).*' GGML_HEXAGON_PROFILE=1
LOG=~/spec/npu_handoff.log
bin/llama-server -m ~/models/gemma-4-E4B_q4_0-it.gguf -t 4 -c 1024 --port 8091 --no-webui \
    --device HTP0 -ngl 4 -nkvo --verbose > $LOG 2>&1 &
pid=$!
for i in $(seq 120); do curl -s localhost:8091/health | grep -q ok && break; sleep 1; done
curl -s localhost:8091/completion -d '{"prompt":"<|turn>user\nWhy is the sky blue?<turn|>\n<|turn>model\n","n_predict":16,"temperature":0}' > /dev/null
kill $pid; wait $pid 2>/dev/null
grep "profile-batch" $LOG | tail -40 | awk '{
  for (i=1;i<=NF;i++) { if ($i=="n-ops") n=$(i+1); if ($i=="batch-dur-usec") b=$(i+1); if ($i=="htp-ops-usec") h=$(i+1) }
  cnt++; tb+=b; th+=h; tn+=n
} END { printf "last %d batches: avg %.1f ops per batch, round trip %.0f us, NPU compute %.0f us, handoff %.0f us per batch\n", cnt, tn/cnt, tb/cnt, th/cnt, (tb-th)/cnt }'
grep "profile-batch" $LOG | tail -3
