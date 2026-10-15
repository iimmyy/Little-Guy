#!/bin/bash
# 31B for a 24 GB Mac: from Google's 4-bit, trim the least harmful tracks until the file is <= 16 GB (the Mac's
# GPU limit is 19.07 GB for model + chat memory + buffers), then build and score on both texts.
set -uo pipefail
S=${S:-/scratch}; . $S/venv/bin/activate
CFG=$S/squeeze/gemma4-31b.json; Q="python $S/squeeze/squeeze.py"
log() { echo "== $(date -u +%H:%M:%S) $*"; }
python - "$CFG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(name="31b-16gb", target_gb=16.0, probe_chunks=c["evals"][0]["chunks"], refine_probe=8, refine_keep=4)
c.pop("target_kl", None)
json.dump(c, open(sys.argv[1] + ".fit", "w"), indent=1)
PY
log trim; $Q $CFG.fit trim
log build; $Q $CFG.fit build
log "score ours"; $Q $CFG score /mnt/squeeze/31b-16gb.gguf
log done
