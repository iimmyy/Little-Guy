#!/bin/bash
# Why is the drafter slow on the NPU? One raw question, 120 words, greedy, per setup: words/s, guesses kept, and the
# server's own per-step timing. Short on purpose (minutes, not hours). Runs ON the board.
# usage: bash npu_draft_diag.sh <engine dir> <out dir>
set -u
ENG=$1 OUT=$2; mkdir -p "$OUT"
M=$HOME/models; D=$M/gemma-4-E4B-assistant-masked-q8_0.gguf
NPU="--device HTP0 -ngl 99 -t 4 -C 0xf0 --cpu-strict 1 -c 4096 -fa on --reasoning off --no-webui --port 8096"
run() {   # name extra-args
  local name=$1; shift
  if curl -s -m 2 127.0.0.1:8096/health >/dev/null; then echo "$name: port busy, NOT run"; return; fi
  (cd $ENG && exec env LD_LIBRARY_PATH=$ENG/lib ADSP_LIBRARY_PATH=$ENG/lib GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=4096 \
     bin/llama-server -m $M/e4b-grid4.gguf $NPU "$@" > "$OUT/$name.server.log" 2>&1) &
  local pid=$! w=0
  until curl -s 127.0.0.1:8096/health | grep -q ok; do
    sleep 1; w=$((w + 1)); if [ $w -gt 300 ] || ! kill -0 $pid 2>/dev/null; then echo "$name: did not start"; kill $pid 2>/dev/null; return; fi
  done
  python3 - "$name" "$OUT/$name.json" "$OUT/none.json" <<'PY'
import json, sys, urllib.request
body = {"messages": [{"role": "user", "content": "Is it okay to have grapefruit with Dayvigo?"}], "max_tokens": 120,
        "temperature": 0, "seed": 1000}
req = urllib.request.Request("http://127.0.0.1:8096/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1200)); t = r["timings"]
json.dump(r, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
d, a = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
import os
same = ""
if sys.argv[1] != "none" and os.path.exists(sys.argv[3]):
    same = ", reply " + ("IDENTICAL to no drafter" if json.load(open(sys.argv[3]))["choices"][0]["message"]["content"] == r["choices"][0]["message"]["content"] else "DIFFERS from no drafter")
print(f"{sys.argv[1]:28} {t['predicted_n']} words, {1000 * t['predicted_n'] / t['predicted_ms']:.2f} words/s, guesses kept {a}/{d}{same}")
PY
  grep -q dsp-rsp "$OUT/$name.server.log" && echo "   $name: NPU reported failed ops: $(grep -c dsp-rsp "$OUT/$name.server.log")"
  kill $pid; wait $pid 2>/dev/null
  w=0; while curl -s -m 2 127.0.0.1:8096/health >/dev/null && [ $w -lt 60 ]; do sleep 1; w=$((w + 1)); done
}
DR="-md $D --spec-type draft-mtp --spec-draft-n-max 1"
run none
if [ -n "${ONLY_REPEAT:-}" ]; then run none-again; run none-again2; echo DIAG_DONE; exit; fi
run draft1-cpu                $DR -ngld 0 -devd none -td 4
run draft1-cpu-nobackendsamp  $DR -ngld 0 -devd none -td 4 --no-spec-draft-backend-sampling
run draft1-npu                $DR -ngld 99 -devd HTP0
run draft1-npu-nobackendsamp  $DR -ngld 99 -devd HTP0 --no-spec-draft-backend-sampling
echo DIAG_DONE
