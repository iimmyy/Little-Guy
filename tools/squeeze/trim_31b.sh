#!/bin/bash
# 31B top-down: from Google's 4-bit, downgrade tracks while drift stays within 10% of the official 4-bit's own
# drift on the chat test (measured at the same settings), then build and score on both texts.
set -uo pipefail
S=${S:-/scratch}; . $S/venv/bin/activate
CFG=$S/squeeze/gemma4-31b.json; Q="python $S/squeeze/squeeze.py"
log() { echo "== $(date -u +%H:%M:%S) $*"; }
python - "$CFG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(name="31b-trim", target_kl=round(0.06362 * 1.10, 5), probe_chunks=c["evals"][0]["chunks"], refine_probe=8, refine_keep=3)
json.dump(c, open(sys.argv[1] + ".trim", "w"), indent=1)
PY
log trim; $Q $CFG.trim trim
log build; $Q $CFG.trim build
log "score ours"; $Q $CFG score /mnt/squeeze/31b-trim.gguf
log done
