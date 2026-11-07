#!/bin/sh
# Song + compressor, in full. GOAL-mix-v1.md. Arms run one after another on one server at a time.
W=$HOME/work/ligands; R=$W/science/room; S=$R/mix; SRV=$HOME/work/llama.cpp-lfm3/build-g5/bin/llama-server; PL=$HOME/.littleguy/hf/lfm3-1.7b; P=8094
PYB="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"
RUN="$PYB $W/data/run_baseline.py"
cd $S
up() { # name gguf extra...
  n=$1; g=$2; shift 2
  $SRV -m $g -ngl 99 -c 120000 --parallel 10 --port $P --host 127.0.0.1 --jinja --cache-ram 0 "$@" > server-$n.log 2>&1 &
  s=$!
  for i in $(seq 1 400); do curl -s -m 2 127.0.0.1:$P/health | grep -q ok && return 0; kill -0 $s 2>/dev/null || break; sleep 1; done
  echo "SERVER_FAILED $n"; return 1
}
arm() { # name
  env BARE=1 MAXNEW=6000 timeout 21600 $RUN $W/brake/eval-107.jsonl eval-$1.jsonl $P 5 > run-$1.log 2>&1 &
  a=$!
  env BARE=1 MAXNEW=6000 SAMPLE=1.0,0.95,20 SEED=1 timeout 21600 $RUN $W/brake/eval-107.jsonl eval-$1-s1.jsonl $P 5 > run-$1-s1.log 2>&1 &
  b=$!
  wait $a $b
  echo "$(date +%H:%M) done $1"
}
down() { kill $s; sleep 5; }
env PYTHONPATH=$HOME/work/llama.cpp-lfm3/gguf-py:$HOME/tf521 $HOME/venv/bin/python -u make_cvec.py $R/turnup-dirs-v1.pt wait 21 wait-L21.gguf || exit 1
# engine against Python, plain and knob
up check $W/night/plain-BF16.gguf --control-vector-scaled wait-L21.gguf:1.0 || exit 1
$PYB -u engine_check.py $PL $W/science/broken/broken-steps-v2.jsonl $R/turnup-held-wait-all21-v1.jsonl knob:1 $P 15 > check-knob.txt 2>&1; c=$?
tail -n 1 check-knob.txt; down
[ $c = 0 ] || { echo CHECK_FAILED; exit 1; }
up plainhead $W/night/plain-BF16.gguf || exit 1
head -n 10 $W/brake/eval-107.jsonl > eval-head10.jsonl
env BARE=1 MAXNEW=6000 timeout 3600 $RUN eval-head10.jsonl eval-plainhead.jsonl $P 5 > run-plainhead.log 2>&1
down
echo "$(date +%H:%M) checks done"
up mix $W/night/c5-BF16.gguf --control-vector-scaled wait-L21.gguf:1.0 || exit 1
arm mix; down
up c5 $W/night/c5-BF16.gguf || exit 1
arm c5; down
up song $W/night/plain-BF16.gguf --control-vector-scaled wait-L21.gguf:1.0 || exit 1
arm song; down
echo MIX_ALL_DONE
