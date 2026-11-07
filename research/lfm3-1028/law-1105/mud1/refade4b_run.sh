#!/bin/sh
# Pass 4 references, GOAL-refade-v4.md. Starts once pass 4 prints PASS4_DONE, gives up after 40 minutes.
W=$HOME/work/ligands; O=$W/science/room/mud; SRV=$HOME/work/llama.cpp-lfm3/build-g5/bin/llama-server
PYB="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"; RUN="$PYB $W/data/run_baseline.py"
cd $O
i=0; until grep -q PASS4_DONE refade4.log; do i=$((i+1)); [ $i -gt 160 ] && { echo PASS4_NEVER_FINISHED; exit 1; }; sleep 15; done
up() { # name gguf port
  $SRV -m $2 -ngl 99 -c 120000 --parallel 10 --port $3 --host 127.0.0.1 --jinja --cache-ram 0 > server-$1.log 2>&1 &
  LAST=$!
  for i in $(seq 1 400); do curl -s -m 2 127.0.0.1:$3/health | grep -q ok && return 0; kill -0 $LAST 2>/dev/null || break; sleep 1; done
  echo "SERVER_FAILED $1"; return 1
}
arm() { # name port
  env BARE=1 MAXNEW=6000 timeout 7200 $RUN screen-sci-40.jsonl screen-$1-sci.jsonl $2 5 > run-screen-$1-sci.log 2>&1 &
  x=$!
  env BARE=1 MAXNEW=6000 timeout 7200 $RUN screen-math-40.jsonl screen-$1-math.jsonl $2 5 > run-screen-$1-math.log 2>&1 &
  y=$!
  wait $x $y; echo "$(date -u +%H:%M) done $1"
}
up plain $W/night/plain-BF16.gguf 8230 || exit 1; S=$LAST
up f5 $W/science/touched/f5-BF16.gguf 8231 || exit 1; T=$LAST
arm plain 8230 & A=$!
arm f5 8231 & B=$!
wait $A $B
kill $S $T; sleep 5
echo PASS4B_DONE
