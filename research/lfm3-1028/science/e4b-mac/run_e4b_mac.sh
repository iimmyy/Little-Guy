#!/bin/sh
# E4B untouched (full precision) on the Mac, science build half, bare question, greedy. 4 slots.
O=$LITTLEGUY_ROOT/.run/results/mind/lfm3-1028/science/e4b-mac; P=8301
~/.littleguy/src/llama.cpp-lfm3/build/bin/llama-server -m $LITTLEGUY_WEIGHTS/fresh/e4b-full-bf16.gguf -ngl 99 -c 32768 --parallel 4 --port $P --host 127.0.0.1 --jinja --cache-ram 0 --reasoning-format none > $O/server.log 2>&1 &
SV=$!
echo $SV > $O/server.pid
i=0; until curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok; do i=$((i+1)); [ $i -gt 120 ] && { echo SERVER_NEVER_UP; kill $SV; exit 1; }; kill -0 $SV 2>/dev/null || { echo SERVER_DIED; exit 1; }; sleep 5; done
echo up $(date)
$LITTLEGUY_ROOT/.run/mind-venv/bin/python $LITTLEGUY_ROOT/tools/mind/census/engine_run.py http://127.0.0.1:$P $LITTLEGUY_ROOT/.run/results/mind/lfm3-1028/science/science-build-all-v1.jsonl $O/e4b-g.jsonl none chat=1 par=4 tail=0 > $O/run-g.log 2>&1
kill $SV
touch $O/e4b-g.done
