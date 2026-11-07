#!/bin/sh
# full_test.sh <name> <gguf> <port>: trap sets (thinking on) and the night test (greedy + s1-3), bare question, one server, 48 slots
W=$HOME/work/ligands; N=$W/night; B=$W/brake; HF=$HOME/.littleguy/hf/lfm3-1.7b; n=$1; g=$2; P=$3
$HOME/work/llama.cpp-lfm3/build/bin/llama-server -m $g -ngl 99 -c 393216 --parallel 48 --port $P --host 127.0.0.1 --jinja --cache-ram 0 > $B/server-full-$n.log 2>&1 &
SV=$!
until curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok; do sleep 5; done
PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python $W/tools/mind/census/engine_run.py"; U=http://127.0.0.1:$P; NP=$W/data/math500-night-test100.jsonl
PIDS=""
for set in tricks herring walls; do
  $PY $U $B/$set.jsonl $B/brake-$set-$n-bare-think.jsonl $HF chat=1 par=8 tail=0 > $B/run-$set-$n-bare-think.log 2>&1 &
  PIDS="$PIDS $!"
done
$PY $U $NP $N/bare-$n.jsonl $HF chat=1 par=6 tail=0 > $N/run-bare-$n.log 2>&1 &
PIDS="$PIDS $!"
for s in 1 2 3; do
  $PY $U $NP $N/bare-$n-s$s.jsonl $HF chat=1 par=6 tail=0 sample=1.0,0.95,20 seed=$s > $N/run-bare-$n-s$s.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
kill $SV
touch $B/full-$n.done
