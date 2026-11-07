#!/bin/sh
# eval_box1.sh <name> <gguf> <port> : bare eval-107 greedy + s1 on box1, 16 slots
W=$HOME/work/ligands; S=$W/brake/sweep; HF=$HOME/.littleguy/hf/lfm3-1.7b; n=$1; g=$2; P=$3
$HOME/work/llama.cpp-lfm3/build/bin/llama-server -m $g -ngl 99 -c 196608 --parallel 24 --port $P --host 127.0.0.1 --jinja --cache-ram 0 > $S/server-$n.log 2>&1 &
SP=$!
until curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok; do sleep 5; done
PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python $W/tools/mind/census/engine_run.py"
$PY http://127.0.0.1:$P $W/brake/eval-107.jsonl $S/eval-$n.jsonl $HF chat=1 par=12 tail=0 > $S/run-$n.log 2>&1 &
A=$!
$PY http://127.0.0.1:$P $W/brake/eval-107.jsonl $S/eval-$n-s1.jsonl $HF chat=1 par=12 tail=0 sample=1.0,0.95,20 seed=1 > $S/run-$n-s1.log 2>&1 &
B=$!
wait $A; wait $B
kill $SP
touch $S/eval-$n.done
