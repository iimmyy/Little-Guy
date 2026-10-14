#!/bin/bash
# Gemma 4 31B through the squeeze pipeline, end to end, after research/big-model/build31b.sh.
set -uo pipefail
S=${S:-/scratch}; B=$S/src/build/bin; . $S/venv/bin/activate
CFG=$S/squeeze/gemma4-31b.json; Q="python $S/squeeze/squeeze.py $CFG"
log() { echo "== $(date -u +%H:%M:%S) $*"; }
while pgrep -f solo.sh > /dev/null; do sleep 30; done
log "engine: scoring tool with chat markers"
(cd $S/src && cmake --build build --target llama-perplexity -j 16 > /dev/null 2>&1) || { log "build failed"; exit 1; }
[ -f $S/31b_kl_chat.bin ] || { log "chat reference (q8_0)"; LG_PPL_SPECIAL=1 LG_DICT_PREVIEW=0 $B/llama-perplexity -m $S/31b-q8_0.gguf -f $S/chat_eval.txt -c 512 --chunks 4 -t 16 --kl-divergence-base $S/31b_kl_chat.bin > $S/kl_chat_base.log 2>&1; }
log donors; $Q donors
log probe; $Q probe
for b in 13 14 15 16; do
  log "allocate $b GB"; python - "$CFG" "$b" <<'PY'
import json, sys; c = json.load(open(sys.argv[1])); c["budget_gb"] = float(sys.argv[2]); c["name"] = f"31b-{sys.argv[2]}gb"
json.dump(c, open(sys.argv[1] + f".{sys.argv[2]}", "w"), indent=1)
PY
  python $S/squeeze/squeeze.py $CFG.$b allocate && cp /mnt/squeeze/recipe.json /mnt/squeeze/recipe-$b.json
done
for b in 14 15; do
  cp /mnt/squeeze/recipe-$b.json /mnt/squeeze/recipe.json
  log "build $b GB"; python $S/squeeze/squeeze.py $CFG.$b build
  log "score $b GB"; python $S/squeeze/squeeze.py $CFG score /mnt/squeeze/31b-${b}gb.gguf
done
log "score google q4"; python $S/squeeze/squeeze.py $CFG score $S/google/gemma-4-31B_q4_0-it.gguf
log "score first try (brain C recipe)"; python $S/squeeze/squeeze.py $CFG score $S/31b-C-preview.gguf
log done
