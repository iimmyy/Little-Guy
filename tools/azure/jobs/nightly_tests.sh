#!/bin/bash
# The nightly brain tests, run by the lg-tests Container Apps job (02:00 UTC, 4 vCPU, 8 GiB). Each brain answers
# the four behavior questions N times on the cloud CPU with the same settings as the Mac test (persona_ab.py), one
# brain at a time so the container's disk holds a single brain; then the AI judge grades every reply. Results land
# in storage under results/nightly/<date>/. The job signs in with its own managed identity.
# Env: LITTLEGUY_STORAGE (account), LG_BRAINS ("C=clean-Q2_K-C ..." name=file stem), LG_N (replies per question)
set -uo pipefail
exec 1>&2   # one stream, so every line reaches the job's console log in order
ST=$LITTLEGUY_STORAGE
DAY=$(date -u +%F)
W=/work
az login --identity -o none
blob() { az storage blob "$@" --account-name "$ST" --auth-mode login --no-progress -o none; }

mkdir -p "$W/run/weights" "$W/run/logs" "$W/run/results/behavior" && cd "$W"
blob download -c build -n tests.tar.gz -f tests.tar.gz \
    && python3 -c "import tarfile; tarfile.open('tests.tar.gz').extractall()"   # tests/behavior and persona/ (the image has no tar)
blob download -c build -n nightly/latest/llama-server-x86 -f llama-server && chmod +x llama-server
export LITTLEGUY_RUN=$W/run LITTLEGUY_SERVER=$W/llama-server
echo "== $(date -u) tests start on $(nproc) cores"

for pair in $LG_BRAINS; do
    name=${pair%%=*}; file=${pair#*=}.gguf
    blob download -c brains -n "$file" -f "run/weights/$file" || { echo "missing brain $file"; continue; }
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
(cd tests/behavior && python3 judge.py "$W/run/results/behavior/all_brains.json")
for f in run/results/behavior/*.json; do blob upload -c results -n "nightly/$DAY/$(basename "$f")" -f "$f" --overwrite; done
echo "== $(date -u) tests done"
