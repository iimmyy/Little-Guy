#!/bin/sh
# Untouched LFM3 with room to finish. GOAL-reach-v1.md. usage: reach.sh
W=$HOME/work/ligands; O=$W/science/room/reach; HF=$HOME/.littleguy/hf/lfm3-1.7b; P=8210; SET=$W/science/room/mix/sci-held-81.jsonl
mkdir -p $O; cd $O
$HOME/work/llama.cpp-lfm3/build-a10/bin/llama-server -m $W/night/plain-BF16.gguf -ngl 99 -c 294912 --parallel 12 --port $P --host 127.0.0.1 --jinja --cache-ram 0 > server-reach.log 2>&1 &
SV=$!
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok && break; sleep 5; done
curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok || { echo SERVER_FAILED; exit 1; }
echo "$(date -u +%H:%M) server up pid $SV"
PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python $W/tools/mind/census/engine_run.py"; U=http://127.0.0.1:$P
$PY $U $SET reach-g.jsonl $HF chat=1 par=6 tail=0 maxnew=24000 > run-reach-g.log 2>&1 &
A=$!
$PY $U $SET reach-s1.jsonl $HF chat=1 par=6 tail=0 maxnew=24000 sample=1.0,0.95,20 seed=1 > run-reach-s1.log 2>&1 &
B=$!
wait $A $B
echo "$(date -u +%H:%M) two runs done"
kill $SV
echo "REACH_DONE"
