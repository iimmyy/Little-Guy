#!/bin/bash
# Azure ML side of the autopilot: the nightly per-layer gate search on public text (tools/azure/ml/).
# Stages only the files the job needs (no weights, no persona, no conversations), registers the datastores and the
# environment, and creates the 03:00 UTC schedule. RUN_NOW=1 also submits one run right away.
#   tools/azure/ml_deploy.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RG=littleguy; WS=lg-ml
ST=lg$(grep -E '^LITTLEGUY_AZ_SUFFIX=' littleguy.env | cut -d= -f2)
STAGE=$(mktemp -d)
mkdir -p "$STAGE/code/research/gate-assay" "$STAGE/code/tests/behavior" "$STAGE/code/.run/lcpp-src/gguf-py"
cp research/gate-assay/send_skip.py research/gate-assay/mute_test.py "$STAGE/code/research/gate-assay/"
cp tests/behavior/common.py tests/behavior/scoring.py "$STAGE/code/tests/behavior/"
cp -R .run/lcpp-src/gguf-py/gguf "$STAGE/code/.run/lcpp-src/gguf-py/"
cp tools/azure/ml/run_gate_sweep.sh "$STAGE/code/"
for f in tools/azure/ml/*.yaml; do
    sed -e "s|__STORAGE__|$ST|" -e "s|__CODE__|$STAGE/code|" "$f" > "$STAGE/$(basename "$f")"
done
ml() { az ml "$@" -g "$RG" -w "$WS" -o none; }
for c in masters brains results; do ml datastore create --file "$STAGE/datastore-$c.yaml"; done
ml environment create --file "$STAGE/environment.yaml"
(cd "$STAGE" && ml schedule create --file gate-sweep-schedule.yaml)
echo "schedule: $(az ml schedule show -n lg-gate-sweep-nightly -g "$RG" -w "$WS" --query 'trigger.expression' -o tsv 2>/dev/null) UTC"
if [ -n "${RUN_NOW:-}" ]; then
    (cd "$STAGE" && az ml job create --file gate-sweep-job.yaml -g "$RG" -w "$WS" --query name -o tsv 2>/dev/null)
fi
