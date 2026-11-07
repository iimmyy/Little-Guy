#!/bin/sh
# First refade screen. GOAL-refade-v1.md. Waits for the mud reading, checks the engine, then runs arms two at a time.
W=$HOME/work/ligands; O=$W/science/room/mud; F=/opt/dlami/nvme/fade; C5=$W/night/c5-BF16.gguf; SRV=$HOME/work/llama.cpp-lfm3/build-g5/bin/llama-server
PYB="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"; RUN="$PYB $W/data/run_baseline.py"
cd $O
while ! grep -q MUD_RUN_DONE mud.log; do sleep 30; done
$HOME/venv/bin/python gguf_fade.py $C5 $F/conv05.gguf all:conv=0.5 || exit 1
$HOME/venv/bin/python gguf_fade.py $C5 $F/conv15.gguf all:conv=1.5 || exit 1
$HOME/venv/bin/python gguf_fade.py $C5 $F/attn05.gguf all:attn=0.5 || exit 1
$HOME/venv/bin/python gguf_fade.py $C5 $F/attn15.gguf all:attn=1.5 || exit 1
up() { # name gguf port
  $SRV -m $2 -ngl 99 -c 120000 --parallel 10 --port $3 --host 127.0.0.1 --jinja --cache-ram 0 > server-$1.log 2>&1 &
  LAST=$!
  for i in $(seq 1 400); do curl -s -m 2 127.0.0.1:$3/health | grep -q ok && return 0; kill -0 $LAST 2>/dev/null || break; sleep 1; done
  echo "SERVER_FAILED $1"; return 1
}
up conv05 $F/conv05.gguf 8230 || exit 1; S=$LAST
up attn15 $F/attn15.gguf 8231 || exit 1; T=$LAST
$PYB -u fade_check.py $W/pam/c5 votes-steps-v1.jsonl 8230 conv 0.5 > check-conv05.txt 2>&1; a=$?
$PYB -u fade_check.py $W/pam/c5 votes-steps-v1.jsonl 8231 attn 1.5 > check-attn15.txt 2>&1; b=$?
tail -n 1 check-conv05.txt; tail -n 1 check-attn15.txt
[ $a = 0 ] && [ $b = 0 ] || { echo CHECK_FAILED; kill $S $T; exit 1; }
arm() { # name port
  env BARE=1 MAXNEW=6000 timeout 14400 $RUN screen-sci-40.jsonl screen-$1-sci.jsonl $2 5 > run-screen-$1-sci.log 2>&1 &
  x=$!
  env BARE=1 MAXNEW=6000 timeout 14400 $RUN screen-math-40.jsonl screen-$1-math.jsonl $2 5 > run-screen-$1-math.log 2>&1 &
  y=$!
  wait $x $y; echo "$(date -u +%H:%M) done $1"
}
arm conv05 8230 & A=$!
arm attn15 8231 & B=$!
wait $A $B; kill $S $T; sleep 5
up base $C5 8230 || exit 1; S=$LAST
up attn05 $F/attn05.gguf 8231 || exit 1; T=$LAST
arm base 8230 & A=$!
arm attn05 8231 & B=$!
wait $A $B; kill $S $T; sleep 5
up conv15 $F/conv15.gguf 8230 || exit 1; S=$LAST
arm conv15 8230
kill $S
echo SCREEN_DONE
