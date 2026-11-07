#!/bin/sh
# Song on f5, science. Same settings as science_touched.sh. GOAL-f5song-v1.md. usage: f5song.sh <name> <gguf> <extra server args...>
N=$1; G=$2; shift 2
W=$HOME/work/ligands; O=$W/science/room/mix; HF=$HOME/.littleguy/hf/lfm3-1.7b; P=8200; SET=$O/sci-held-81.jsonl
cd $O
$HOME/work/llama.cpp-lfm3/build-a10/bin/llama-server -m $G -ngl 99 -c 196608 --parallel 24 --port $P --host 127.0.0.1 --jinja --cache-ram 0 "$@" > server-$N.log 2>&1 &
SV=$!
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok && break; sleep 5; done
curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok || { echo SERVER_FAILED; exit 1; }
PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python $W/tools/mind/census/engine_run.py"; U=http://127.0.0.1:$P
$PY $U $SET $O/$N-g.jsonl $HF chat=1 par=6 tail=0 > $O/run-$N-g.log 2>&1 &
PIDS=$!
for s in 1 2 3; do
  $PY $U $SET $O/$N-s$s.jsonl $HF chat=1 par=6 tail=0 sample=1.0,0.95,20 seed=$s > $O/run-$N-s$s.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
kill $SV
echo "F5SONG_DONE $N"
