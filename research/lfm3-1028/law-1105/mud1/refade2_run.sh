#!/bin/sh
# Pass 2. GOAL-refade-v2.md. Builds each file when its turn comes, deletes it after, two arms at a time.
W=$HOME/work/ligands; O=$W/science/room/mud; F=/opt/dlami/nvme/fade; C5=$W/night/c5-BF16.gguf; SRV=$HOME/work/llama.cpp-lfm3/build-g5/bin/llama-server
PYB="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"; RUN="$PYB $W/data/run_baseline.py"
cd $O
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
one() { # name rule port; sets LAST
  if [ "$2" = base ]; then G=$C5; else G=$F/$1.gguf; $HOME/venv/bin/python gguf_fade.py $C5 $G "$2" || return 1; fi
  up $1 $G $3
}
pair() { # n1 r1 n2 r2
  one $1 "$2" 8230 || exit 1; S=$LAST
  if [ -n "$3" ]; then one $3 "$4" 8231 || exit 1; T=$LAST; else T=; fi
  arm $1 8230 & A=$!
  if [ -n "$3" ]; then arm $3 8231 & B=$!; wait $A $B; else wait $A; fi
  kill $S $T; sleep 5
  rm -f $F/$1.gguf; [ -n "$3" ] && rm -f $F/$3.gguf
}
pair base2 base        e-c08 0-7:conv=0.8
pair e-c12 0-7:conv=1.25 e-a08 0-7:attn=0.8
pair e-a12 0-7:attn=1.25 m-c08 8-15:conv=0.8
pair m-c12 8-15:conv=1.25 m-a08 8-15:attn=0.8
pair m-a12 8-15:attn=1.25 l-c08 16-23:conv=0.8
pair l-c12 16-23:conv=1.25 l-a08 16-23:attn=0.8
pair l-a12 16-23:attn=1.25
echo PASS2_DONE
