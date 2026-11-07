#!/bin/sh
# Science, after the math mix and c5 arms. GOAL-sci-v1.md. Takes over from mix_all.sh before its song-alone arm.
W=$HOME/work/ligands; R=$W/science/room; S=$R/mix; SRV=$HOME/work/llama.cpp-lfm3/build-g5/bin/llama-server; P=8094
RUN="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python $W/data/run_baseline.py"
cd $S
until grep -q "done c5" mix-all.log; do sleep 10; done
for p in $(pgrep -f "sh mix_al[l].sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "llama-serve[r].*port $P"); do kill $p; done
sleep 8
up() { n=$1; g=$2; shift 2
  $SRV -m $g -ngl 99 -c 120000 --parallel 10 --port $P --host 127.0.0.1 --jinja --cache-ram 0 "$@" > server-$n.log 2>&1 &
  s=$!
  for i in $(seq 1 400); do curl -s -m 2 127.0.0.1:$P/health | grep -q ok && return 0; kill -0 $s 2>/dev/null || break; sleep 1; done
  echo "SERVER_FAILED $n"; return 1
}
arm() {
  env BARE=1 MAXNEW=6000 timeout 28800 $RUN sci-held-81.jsonl eval-$1.jsonl $P 5 > run-$1.log 2>&1 &
  a=$!
  env BARE=1 MAXNEW=6000 SAMPLE=1.0,0.95,20 SEED=1 timeout 28800 $RUN sci-held-81.jsonl eval-$1-s1.jsonl $P 5 > run-$1-s1.log 2>&1 &
  b=$!
  wait $a $b
  echo "$(date +%H:%M) done $1"
}
down() { kill $s; sleep 5; }
up scimix $W/night/c5-BF16.gguf --control-vector-scaled wait-L21.gguf:1.0 || exit 1
arm scimix; down
up scic5 $W/night/c5-BF16.gguf || exit 1
arm scic5; down
echo SCI_ALL_DONE
