#!/bin/sh
# Runs up to two arms side by side on the 80 screen problems, same harness and cap as the refade screens. usage: fix_arm.sh <tag> <name1> <gguf1> [<name2> <gguf2>]
W=$HOME/work/ligands; O=$W/science/room/fix; SRV=$HOME/work/llama.cpp-lfm3/build-g5/bin/llama-server
PYB="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"; RUN="$PYB $W/data/run_baseline.py"
cd $O
tag=$1; shift
up() { # name gguf port
  $SRV -m $2 -ngl 99 -c 32768 --parallel 4 --port $3 --host 127.0.0.1 --jinja --cache-ram 0 > server-$1.log 2>&1 &
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
up $1 $2 8240 || exit 1; S=$LAST
arm $1 8240 & A=$!
if [ -n "$3" ]; then
  up $3 $4 8241 || exit 1; T=$LAST
  arm $3 8241 & B=$!
  wait $A $B; kill $S $T
else
  wait $A; kill $S
fi
sleep 5
echo "${tag}_DONE"
