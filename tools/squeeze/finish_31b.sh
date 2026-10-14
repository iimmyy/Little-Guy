#!/bin/bash
# After the 31B probes: quality-first refine (smallest mix as good as Google's official 4-bit on the probe test),
# build it, score it and the references on both texts. Runs on the build box after run_31b.sh's probe step.
set -uo pipefail
S=${S:-/scratch}; . $S/venv/bin/activate
CFG=$S/squeeze/gemma4-31b.json; Q="python $S/squeeze/squeeze.py"
log() { echo "== $(date -u +%H:%M:%S) $*"; }
while pgrep -f "squeeze.py.*probe" > /dev/null; do sleep 30; done
log "google q4 on the probe test"
T=$(python - <<PY
import json, subprocess, sys
c = json.load(open("$CFG")); ev = c["evals"][0]
r = subprocess.run(["$Q".split()[0], "$S/squeeze/squeeze.py", "$CFG", "score", "$S/google/gemma-4-31B_q4_0-it.gguf"], capture_output=True, text=True)
print(json.loads(r.stdout[r.stdout.index("{"):])["chat"]["kld"])
PY
)
log "target KL (google q4, chat): $T"
python - "$CFG" "$T" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["target_kl"] = float(sys.argv[2]); c["budget_gb"] = 17.65; c["name"] = "31b-qualityfirst"
c["probe_chunks"] = c["evals"][0]["chunks"]
json.dump(c, open(sys.argv[1] + ".qf", "w"), indent=1)
PY
log refine; $Q $CFG.qf refine
log build; $Q $CFG.qf build
log "score ours"; $Q $CFG score /mnt/squeeze/31b-qualityfirst.gguf
log "score google q4"; $Q $CFG score $S/google/gemma-4-31B_q4_0-it.gguf
log done
