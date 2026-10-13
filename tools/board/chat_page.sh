#!/bin/bash
# Little Guy's chat page on the board, with his persona already read when the first message arrives.
#
# Reading the persona (~700 word-pieces) takes ~22 s on the Q6A, so a fresh server made the first reply slow.
# On the first start this reads the persona once and saves that state (llama-server slot save); every later start
# loads the saved state in about a second. The saved file is tied to the brain and the persona: change either and
# it is rebuilt automatically.
#
# Settings (environment): LG_DIR (engine folder with bin/ and lib/, default ~/preview), LG_BRAIN (brain .gguf),
# LG_PERSONA (persona text, '#' lines ignored), LG_PORT (default 8090).
set -u
DIR=${LG_DIR:-$HOME/preview}
BRAIN=${LG_BRAIN:-$HOME/models/clean-Q2_K-C-preview.gguf}
PERSONA=${LG_PERSONA:-$HOME/spec/little_guy_persona.txt}
PORT=${LG_PORT:-8090}
SLOTS=$DIR/slots
mkdir -p "$SLOTS"
cd "$DIR"

# default UI settings: persona as the system message, speed shown under replies
python3 - "$PERSONA" > ui.json <<'EOF'
import json, sys
p = "\n".join(l for l in open(sys.argv[1]).read().splitlines() if l.strip() and not l.startswith("#"))
print(json.dumps({"systemMessage": p, "showTokensPerSecond": True}))
EOF

LD_LIBRARY_PATH=$DIR/lib GGML_HEXAGON_NDEV=0 bin/llama-server -m "$BRAIN" -ngl 0 --device none \
  -t 4 -C 0xf0 --cpu-strict 1 -c 4096 --reasoning off --host 0.0.0.0 --port "$PORT" \
  --ui-config-file ui.json --slot-save-path "$SLOTS" --swa-full &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until curl -s "127.0.0.1:$PORT/health" | grep -q ok; do sleep 1; done

# the saved state is named after the brain and persona, so a change to either makes a new one
KEY=$(cat "$PERSONA" <(stat -c %s-%Y "$BRAIN") | sha256sum | cut -c1-16)
FILE=persona-$KEY.bin
t0=$(date +%s.%N)
if [ -f "$SLOTS/$FILE" ] && curl -sf -X POST "127.0.0.1:$PORT/slots/0?action=restore" -H 'Content-Type: application/json' \
     -d "{\"filename\":\"$FILE\"}" > /dev/null; then
  echo "persona loaded from $FILE in $(echo "$(date +%s.%N) - $t0" | bc) s"
else
  # read it once: the chat template's system turn is the prefix of every conversation the page sends
  PROMPT=$(curl -s "127.0.0.1:$PORT/apply-template" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"system","content":json.load(open("ui.json"))["systemMessage"]}],"add_generation_prompt":False}))')" \
    | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["prompt"]))')
  curl -s "127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":$PROMPT,\"n_predict\":0,\"cache_prompt\":true,\"id_slot\":0}" > /dev/null
  curl -sf -X POST "127.0.0.1:$PORT/slots/0?action=save" -H 'Content-Type: application/json' -d "{\"filename\":\"$FILE\"}" > /dev/null
  echo "persona read and saved as $FILE in $(echo "$(date +%s.%N) - $t0" | bc) s"
fi
wait $SERVER
