#!/bin/bash
# The nightly brain tests, run by the lg-tests Container Apps job (02:00 UTC, 4 vCPU, 8 GiB). Each brain answers
# the four behavior questions N times on the cloud CPU with the same settings as the Mac test (persona_ab.py), one
# brain at a time so the container's disk holds a single brain; then the AI judge grades every reply. Results land
# in storage under results/nightly/<date>/. The job signs in with its own managed identity.
# Env: LITTLEGUY_STORAGE (account), LG_BRAINS ("C=clean-Q2_K-C ..." name=file stem), LG_N (replies per question)
set -euo pipefail
ST=$LITTLEGUY_STORAGE
DAY=$(date -u +%F)
W=/work
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')
mkdir -p "$W/run/results/behavior"
exec > >(tee "$W/run/results/behavior/job.log") 2>&1
blob() { az storage blob "$@" --account-name "$ST" --auth-mode login --no-progress -o none; }
finish() {
    rc=$?
    trap - EXIT
    set +e
    python3 - "$W/run/results/behavior/job-receipt.json" "$RUN_ID" "$rc" "$LG_BRAINS" "$LG_N" <<'PY'
import datetime, json, sys
path, run_id, code, brains, samples = sys.argv[1:]
json.dump({"run_id": run_id, "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "exit_code": int(code), "status": "completed" if code == "0" else "failed",
           "brains": brains.split(), "samples_per_question": int(samples)}, open(path, "w"), indent=2)
PY
    for f in "$W"/run/results/behavior/*; do
        blob upload -c results -n "nightly/$DAY/runs/$RUN_ID/$(basename "$f")" -f "$f" --overwrite || rc=1
    done
    exit "$rc"
}
trap finish EXIT
az login --identity -o none

mkdir -p "$W/run/weights" "$W/run/logs" "$W/run/results/behavior" && cd "$W"
blob download -c build -n tests.tar.gz -f tests.tar.gz
python3 -c "import tarfile; tarfile.open('tests.tar.gz').extractall(filter='data')"   # the image has no tar
blob download -c build -n nightly/latest/llama-server-x86 -f llama-server && chmod +x llama-server
export LITTLEGUY_RUN=$W/run LITTLEGUY_SERVER=$W/llama-server
echo "== $(date -u) tests start on $(nproc) cores"

for pair in $LG_BRAINS; do
    name=${pair%%=*}; file=${pair#*=}.gguf
    blob download -c brains -n "$file" -f "run/weights/$file"
    (cd tests/behavior && AB_BRAINS="$name=$file" python3 persona_ab.py - "$LG_N")
    mv run/results/behavior/persona_ab_brains.json "run/results/behavior/brain_$name.json"
    rm -f "run/weights/$file"
done

python3 - <<'EOF'   # one file with every brain, for the judge
import glob, json
merged = {}
for f in sorted(glob.glob("/work/run/results/behavior/brain_*.json")):
    merged |= json.load(open(f))
json.dump(merged, open("/work/run/results/behavior/all_brains.json", "w"), indent=1, ensure_ascii=False)
EOF
python3 tests/behavior/validate_results.py "$W/run/results/behavior/all_brains.json" --brains "$LG_BRAINS" --samples "$LG_N"
(cd tests/behavior && python3 judge.py "$W/run/results/behavior/all_brains.json")
python3 tests/behavior/validate_results.py "$W/run/results/behavior/all_brains.json" --brains "$LG_BRAINS" --samples "$LG_N" \
    --judged "$W/run/results/behavior/all_brains_judged.json"
for f in run/results/behavior/*.json; do blob upload -c results -n "nightly/$DAY/$(basename "$f")" -f "$f" --overwrite; done
echo "== $(date -u) tests done"
