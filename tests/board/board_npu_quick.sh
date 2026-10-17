#!/bin/bash
# QUICK NPU check: end-to-end talking speed on the board's NPU (raw_behavior.py's NPU settings), raw questions
# (no persona), Gemma's sampling, every run starting with the hottest zone at or below 45 C. Brains with and without
# the drafter (the brain checks every guess, so the words are its own). Runs ON the board.
# usage: bash board_npu_quick.sh <out dir> [new engine dir, default ~/preview5]
# The engine runs alone and with the masked drafter (on the NPU, one CPU worker) at 1, 2 and 3 guesses.
set -u
OUT=${1:-$HOME/spec/speed}; mkdir -p "$OUT"
ENG=${2:-$HOME/preview5}
BRAIN=${BRAIN:-$HOME/models/e4b-grid4.gguf}   # e.g. Google's release: BRAIN=~/models/gemma-4-E4B_q4_0-it.gguf TAG=google
TAG=${TAG:-grid4}
M=$HOME/models
CPU="--device HTP0 -ngl 99 -t 4 -C 0xf0 --cpu-strict 1 -c 4096 -fa on --reasoning off --no-webui"
# processor, NPU and memory sensors only (the storage chip's ufs-thermal idles near 55 C and would never cool)
hottest() { m=0; for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in cpu*|cpuss*|nspss*|ddr*) v=$(cat $z/temp); [ $v -gt $m ] && m=$v;; esac; done; echo $(( m / 1000 )); }
cool() { while [ $(hottest) -gt 45 ]; do sleep 5; done; }

run() {   # name engine-dir brain extra-env extra-args
  local name=$1 dir=$2 brain=$3 env=$4; shift 4
  # a server still holding the port would silently answer this run's questions instead (2026-10-17: every drafter
  # run that day talked to an earlier server), so refuse to start until the port is free
  if curl -s -m 2 127.0.0.1:8096/health >/dev/null; then echo "   $name: port 8096 is busy, NOT run"; return; fi
  cool
  local t0=$(hottest)
  # exec: $! is then the server itself, so the kill below stops it (not only the subshell around it)
  (cd $dir && exec env $env LD_LIBRARY_PATH=$dir/lib ADSP_LIBRARY_PATH=$dir/lib GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=4096 bin/llama-server -m $brain $CPU --port 8096 "$@" \
     > "$OUT/$name.server.log" 2>&1) &
  local pid=$!
  local w=0
  until curl -s 127.0.0.1:8096/health | grep -q ok; do
    sleep 1; w=$((w + 1))
    if [ $w -gt 300 ] || ! kill -0 $pid 2>/dev/null; then echo "   $name: server did not start, see $name.server.log"; kill $pid 2>/dev/null; return; fi
  done
  local drafting=0; case " $* " in *" -md "*) drafting=1;; esac
  python3 - "$OUT/$name.json" $drafting <<'EOF'
import json, sys, urllib.request
Q = ["Is it okay to have grapefruit with Dayvigo?",
     "Hey, I just got home from a long shift. How was your day?",
     "Can I leave my dog in the car for ten minutes? It's about 30°C out.",
     "My friend failed her driving test and feels terrible. Can you write her a short message to cheer her up?"]
rows = []
for seed in (1000, 1001, 1002):
    for q in Q:
        body = {"messages": [{"role": "user", "content": q}], "max_tokens": 300, "seed": seed,
                "temperature": 1.0, "top_k": 64, "top_p": 0.95}
        req = urllib.request.Request("http://127.0.0.1:8096/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        t = r["timings"]
        rows.append({"q": q, "seed": seed, "n": t["predicted_n"], "ms": t["predicted_ms"],
                     "drafted": t.get("draft_n", 0), "kept": t.get("draft_n_accepted", 0),
                     "reply": r["choices"][0]["message"]["content"]})
        json.dump(rows, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
n = sum(r["n"] for r in rows); ms = sum(r["ms"] for r in rows); d = sum(r["drafted"] for r in rows); a = sum(r["kept"] for r in rows)
print(f"{n} words in {ms / 1000:.1f} s = {1000 * n / ms:.2f} words/s" + (f", guesses kept {a}/{d} = {100 * a / d:.0f}%" if d else ""))
if sys.argv[2] == "1" and not d: print("   INVALID: drafter run with no guesses")
EOF
  kill $pid; wait $pid 2>/dev/null
  local w2=0; while curl -s -m 2 127.0.0.1:8096/health >/dev/null && [ $w2 -lt 60 ]; do sleep 1; w2=$((w2 + 1)); done
  echo "   ($name started at $t0 C, ended at $(hottest) C)"
}

D=$M/gemma-4-E4B-assistant-masked-q8_0.gguf
# the drafter on the NPU with ONE CPU worker (2026-10-17: the CPU-side steps are tiny, and with 4 workers syncing
# them the drafter cost 57 ms per guess, with 1 worker 8.9 ms; on the CPU (-devd none) it cost far more)
DRAFT="--spec-type draft-mtp -ngld 99 -devd HTP0 -td 1 -Cd 0x80 --cpu-strict-draft 1"
E=$(basename $ENG)
echo "== NPU $E: $TAG";                                 run npu-$E-$TAG          $ENG $BRAIN ""
for n in 1 2 3; do
echo "== NPU $E: $TAG + masked drafter, $n guesses";      run npu-$E-$TAG-masked$n  $ENG $BRAIN "" -md $D $DRAFT --spec-draft-n-max $n
done
echo BOARD_SPEED_DONE
