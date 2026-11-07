#!/bin/sh
# f5 and c5 with room, then untouched seeds 2 and 3. GOAL-f5room-v1.md. usage: room.sh
W=$HOME/work/ligands; O=$W/science/room/f5room; HF=$HOME/.littleguy/hf/lfm3-1.7b; SET=$W/science/room/mix/sci-held-81.jsonl
mkdir -p $O; cd $O
B=$HOME/work/llama.cpp-lfm3/build-a10/bin/llama-server
PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python $W/tools/mind/census/engine_run.py"
up() { # name gguf port ctx slots
  $B -m $2 -ngl 99 -c $4 --parallel $5 --port $3 --host 127.0.0.1 --jinja --cache-ram 0 > server-$1.log 2>&1 &
  eval SV_$1=$!
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:$3/health | grep -q ok && break; sleep 5; done
  curl -s -m 3 http://127.0.0.1:$3/health | grep -q ok || { echo SERVER_FAILED $1; exit 1; }
  echo "$(date -u +%H:%M) server up $1"
}
up f5 $W/science/touched/f5-BF16.gguf 8220 589824 24
up c5 $W/night/c5-BF16.gguf 8221 294912 12
$PY http://127.0.0.1:8220 $SET f5room-g.jsonl $HF chat=1 par=6 tail=0 maxnew=24000 > run-f5room-g.log 2>&1 &
F="$!"
for s in 1 2 3; do
  $PY http://127.0.0.1:8220 $SET f5room-s$s.jsonl $HF chat=1 par=6 tail=0 maxnew=24000 sample=1.0,0.95,20 seed=$s > run-f5room-s$s.log 2>&1 &
  F="$F $!"
done
$PY http://127.0.0.1:8221 $SET c5room-g.jsonl $HF chat=1 par=6 tail=0 maxnew=24000 > run-c5room-g.log 2>&1 &
C="$!"
$PY http://127.0.0.1:8221 $SET c5room-s1.jsonl $HF chat=1 par=6 tail=0 maxnew=24000 sample=1.0,0.95,20 seed=1 > run-c5room-s1.log 2>&1 &
C="$C $!"
wait $F; echo "$(date -u +%H:%M) f5 done"; kill $SV_f5
echo F5ROOM_DONE
up plain $W/night/plain-BF16.gguf 8222 294912 12
$PY http://127.0.0.1:8222 $SET reach-s2.jsonl $HF chat=1 par=6 tail=0 maxnew=24000 sample=1.0,0.95,20 seed=2 > run-reach-s2.log 2>&1 &
P="$!"
$PY http://127.0.0.1:8222 $SET reach-s3.jsonl $HF chat=1 par=6 tail=0 maxnew=24000 sample=1.0,0.95,20 seed=3 > run-reach-s3.log 2>&1 &
P="$P $!"
wait $C; echo "$(date -u +%H:%M) c5 done"; kill $SV_c5
echo C5ROOM_DONE
wait $P; echo "$(date -u +%H:%M) untouched seeds done"; kill $SV_plain
echo ROOM_ALL_DONE
