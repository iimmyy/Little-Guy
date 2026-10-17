#!/bin/bash
# QUICK NPU check: end-to-end talking speed on the board's NPU (raw_behavior.py's NPU settings), raw questions
# (no persona), Gemma's sampling, every run starting with the hottest zone at or below 45 C. Brains with and without
# the drafter (the brain checks every guess, so the words are its own). Runs ON the board.
# usage: bash board_speed.sh <out dir>
set -u
OUT=${1:-$HOME/spec/speed}; mkdir -p "$OUT"
M=$HOME/models
CPU="--device HTP0 -ngl 99 -t 4 -C 0xf0 --cpu-strict 1 -c 4096 -fa on --reasoning off --no-webui"
# processor, NPU and memory sensors only (the storage chip's ufs-thermal idles near 55 C and would never cool)
hottest() { m=0; for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in cpu*|cpuss*|nspss*|ddr*) v=$(cat $z/temp); [ $v -gt $m ] && m=$v;; esac; done; echo $(( m / 1000 )); }
cool() { while [ $(hottest) -gt 45 ]; do sleep 5; done; }

run() {   # name engine-dir brain extra-env extra-args
  local name=$1 dir=$2 brain=$3 env=$4; shift 4
  cool
  local t0=$(hottest)
  (cd $dir && env $env LD_LIBRARY_PATH=$dir/lib ADSP_LIBRARY_PATH=$dir/lib GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=4096 bin/llama-server -m $brain $CPU --port 8096 "$@" \
     > "$OUT/$name.server.log" 2>&1) &
  local pid=$!
  local w=0
  until curl -s 127.0.0.1:8096/health | grep -q ok; do
    sleep 1; w=$((w + 1))
    if [ $w -gt 300 ] || ! kill -0 $pid 2>/dev/null; then echo "   $name: server did not start, see $name.server.log"; kill $pid 2>/dev/null; return; fi
  done
  python3 - "$OUT/$name.json" <<'EOF'
import json, sys, urllib.request
Q = ["Is it okay to have grapefruit with Dayvigo?",
     "Hey, I just got home from a long shift. How was your day?",
     "Can I leave my dog in the car for ten minutes? It's about 30°C out.",
     "My friend failed her driving test and feels terrible. Can you write her a short message to cheer her up?"]
rows = []
for seed in (1000,):
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
EOF
  kill $pid; wait $pid 2>/dev/null
  echo "   ($name started at $t0 C, ended at $(hottest) C)"
}

D=$M/gemma-4-E4B-assistant-masked-q8_0.gguf
DRAFT="--spec-type draft-mtp -ngld 0 -devd none -td 4"
echo "== NPU: new brain";                            run npu-grid4          $HOME/preview5 $M/e4b-grid4.gguf ""
echo "== NPU: new brain + masked drafter, 3 guesses"; run npu-grid4-masked3  $HOME/preview5 $M/e4b-grid4.gguf "" -md $D $DRAFT --spec-draft-n-max 3
echo BOARD_SPEED_DONE
