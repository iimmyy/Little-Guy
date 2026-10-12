#!/bin/bash
# Nightly reproduction of the Aeviant pipeline's public full-genome benchmark
# (campaign aeviant-public-full-genome-compound-het-v1: HG001 GRCh38 with a spiked compound-heterozygous pair,
# Exomiser 15.1.0 + PhEval 0.7.14 on the full 2602 data), run on an Azure Batch node through the pipeline's own
# commands. The Mac run's digests are passed as the expected values for everything that should be byte-identical
# (Exomiser JAR, lib manifest, 2602 data manifest, Aeviant adapter source, runner manifest), so a drifted copy
# fails closed. Only the Linux runtime gets new digests (Java, Python, the one-line data path in the properties,
# the bridge receipt); they are recorded in a Linux runtime receipt. The frozen results are then compared with
# the Mac's frozen results. Public data only; no patient data comes to Azure.
# Env (set by the job schedule): BIO_STORAGE, BIO_IDENTITY_CLIENT_ID
set -euo pipefail
W=${AZ_BATCH_TASK_WORKING_DIR:-$PWD}
DAY=$(date -u +%F)
OUT=$W/out; mkdir -p "$OUT"
exec > >(tee "$OUT/benchmark.log") 2>&1
echo "== $(date -u) start on $(nproc) cores, $(free -g | awk '/Mem/{print $2}') GB RAM, $(df -BG --output=avail "$W" | tail -1) free"
BLOB=https://$BIO_STORAGE.blob.core.windows.net

# Mac digests the cloud copy must reproduce (from work-cycles/public-full-genome-benchmark-v1/execution-receipt.json)
JAR_SHA=3e320171fb811fac90ec0c33cd70d21a21e5a8b2cbc0a89cfb88555438295e24
LIB_SHA=53b1d22216e45b57b24c90148f75e462ce52632f8d4bba2b130f0cba09c3222e
DATA_SHA=fe2e5437014cb514c2e76727148967647d9d1210ad390dd326b325f0081ef41d
ADAPTER_SHA=f39f6f3a1e19aee24a109c362195759bf949f5c5edaf154b78c5da16bbb14ce0
RUNNER_SHA=ed54ef1f41ac0bb1016492b0cc42d3583ed479d10e034a90df359b26ad772b32
CAMPAIGN=aeviant-public-full-genome-compound-het-v1

# --- tools: azcopy, Java (Azul Zulu 21.0.10, the Mac's build), uv + CPython 3.13.12 (the Mac's version) ---
cd "$W"
curl -sL https://aka.ms/downloadazcopy-v10-linux | tar xz --strip-components=1 --wildcards '*/azcopy'
curl -sL https://cdn.azul.com/zulu/bin/zulu21.48.17-ca-jre21.0.10-linux_x64.tar.gz -o jre.tar.gz
mkdir -p jre && tar xzf jre.tar.gz -C jre --strip-components=1
curl -sL https://github.com/astral-sh/uv/releases/download/0.9.7/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz --strip-components=1
./uv python install 3.13.12 -q
PY=$(./uv python find 3.13.12)
export AZCOPY_AUTO_LOGIN_TYPE=MSI AZCOPY_MSI_CLIENT_ID=$BIO_IDENTITY_CLIENT_ID

# --- inputs from storage: the 2602 data, the Exomiser program, the pinned wheels, the pipeline snapshot ---
./azcopy copy "$BLOB/exomiser-data/*" data/ --recursive --log-level=WARNING --output-level=essential
./azcopy copy "$BLOB/toolchain/exomiser" . --recursive --log-level=WARNING --output-level=essential
./azcopy copy "$BLOB/toolchain/wheels" . --recursive --log-level=WARNING --output-level=essential
SNAP=$(./azcopy list "$BLOB/pipeline" --output-type text 2>/dev/null | grep -o 'pipeline-[0-9-]*\.tar\.gz' | sort | tail -1)
./azcopy copy "$BLOB/pipeline/$SNAP" pipeline.tar.gz --log-level=WARNING --output-level=essential
mkdir -p pipe && tar xzf pipeline.tar.gz -C pipe --strip-components=1 && rm pipeline.tar.gz   # the disk is tight
echo "== $(date -u) inputs ready ($SNAP)"

# --- Python: the pipeline itself in one environment, the bridge's pinned dependencies in a separate directory ---
"$PY" -m venv venv
venv/bin/pip install -q ./pipe
venv/bin/pip install -q --no-deps --target deps casefy==1.1.0 jinja2==3.1.6 markupsafe==3.0.3 mypy_extensions==1.1.0 \
    numpy==2.5.2 pandas==3.0.5 phenopackets==2.0.2.post5 polars==1.44.1 polars_runtime_32==1.44.1 protobuf==6.33.6 \
    pyaml==21.10.1 pyserde==0.9.8 python_dateutil==2.9.0.post0 pyyaml==6.0.3 six==1.17.0 typing_extensions==4.16.0 \
    typing_inspect==0.9.0
PYEXE=$(readlink -f venv/bin/python)
sha() { sha256sum "$1" | cut -d' ' -f1; }
venv/bin/aeviant rare-disease-materialize-pheval-exomiser-bridge wheels/pheval_exomiser-0.4.12-py3-none-any.whl \
    wheels/pheval-0.7.14-py3-none-any.whl deps bridge --python-executable "$PYEXE" --python-sha256 "$(sha "$PYEXE")" \
    > "$OUT/bridge-materialize.json"
BRIDGE_RECEIPT=$(find bridge -maxdepth 1 -name '*receipt*.json' | head -1)

# --- the one Linux difference in the method: the data path in the benchmark's properties ---
REL=pipe/release/rare-disease-benchmark-v0.1.0/exomiser-pheval-production-campaign-v1/application.properties
sed "s|^exomiser.data-directory=.*|exomiser.data-directory=$W/data|" "$REL" > application.properties
diff <(grep -v '^exomiser.data-directory=' "$REL") <(grep -v '^exomiser.data-directory=' application.properties) \
    && echo "properties: identical to the Mac's apart from the data path"

# --- the run, through the pipeline's own run-and-freeze command ---
RUNNER=pipe/work-cycles/public-full-genome-benchmark-v1/campaign/runner
venv/bin/aeviant rare-disease-run-and-freeze-exomiser-benchmark "$RUNNER/runner-manifest.json" \
    "$OUT/raw-results.json" "$OUT/execution-receipt.json" "$OUT/frozen-results.json" "$W/scratch" \
    --bridge-receipt "$BRIDGE_RECEIPT" --bridge-receipt-sha256 "$(sha "$BRIDGE_RECEIPT")" \
    --bridge-runtime-directory bridge --dependency-directory deps \
    --python-executable "$PYEXE" --python-sha256 "$(sha "$PYEXE")" \
    --adapter-source-manifest-sha256 "$ADAPTER_SHA" --runner-manifest-sha256 "$RUNNER_SHA" \
    --expected-campaign-id "$CAMPAIGN" --expected-case-count 1 \
    --java "$W/jre/bin/java" --install-directory "$W/exomiser" --data-directory "$W/data" \
    --application-properties "$W/application.properties" \
    --java-sha256 "$(sha jre/bin/java)" --jar-sha256 "$JAR_SHA" --library-manifest-sha256 "$LIB_SHA" \
    --properties-sha256 "$(sha application.properties)" --data-manifest-sha256 "$DATA_SHA" \
    > "$OUT/run.json"
echo "== $(date -u) run finished"

# --- did the cloud reproduce the Mac? compare the frozen rankings ---
"$PY" - "$OUT/frozen-results.json" pipe/work-cycles/public-full-genome-benchmark-v1/frozen-results.json "$OUT/linux-runtime-receipt.json" <<'PY'
import hashlib, json, sys
cloud, mac = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
# the normalized rankings decide it: raw results carry paths and timestamps, the normalized ones don't
same = (cloud["normalized_results_sha256"] == mac["normalized_results_sha256"]
        and cloud["normalized_results"] == mac["normalized_results"]
        and cloud["runner_manifest_sha256"] == mac["runner_manifest_sha256"])
digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
receipt = {"schema_version": "0.1.0", "claim_authority": "REPRODUCTION_CHECK_ONLY",
           "campaign_id": "aeviant-public-full-genome-compound-het-v1",
           "frozen_results_match_mac": same,
           "cloud_normalized_results_sha256": cloud["normalized_results_sha256"],
           "mac_normalized_results_sha256": mac["normalized_results_sha256"],
           "cloud_frozen_sha256": digest(sys.argv[1]), "mac_frozen_sha256": digest(sys.argv[2]),
           "note": "Linux runtime: Azul Zulu 21.0.10 JRE, CPython 3.13.12, bridge and properties rebuilt on the node; "
                   "JAR, lib, 2602 data, adapter source and runner manifest verified against the Mac digests."}
json.dump(receipt, open(sys.argv[3], "w"), indent=1)
print("REPRODUCED: frozen results match the Mac" if same else "DIFFERENT: frozen results differ from the Mac (see receipts)")
PY

./azcopy copy "$OUT/*" "$BLOB/receipts/benchmark/$DAY/" --log-level=WARNING --output-level=essential
echo "== $(date -u) done"
