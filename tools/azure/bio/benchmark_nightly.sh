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
RUN_ID=${AZ_BATCH_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)
OUT=$W/out; mkdir -p "$OUT"
BLOB=https://$BIO_STORAGE.blob.core.windows.net
STAGE=setup
export AZCOPY_AUTO_LOGIN_TYPE=MSI AZCOPY_MSI_CLIENT_ID=$BIO_IDENTITY_CLIENT_ID
finish() {
    rc=$?
    trap - EXIT
    set +e
    exec 1>&3 2>&1
    wait "$TEE_PID"
    printf '{"run_id":"%s","exit_code":%s,"stage":"%s"}\n' "$RUN_ID" "$rc" "$STAGE" > "$OUT/job-receipt.json"
    if [ -d "$W/scratch" ]; then
        while IFS= read -r -d '' f; do
            relative=${f#"$W/scratch/"}
            mkdir -p "$OUT/raw/$(dirname "$relative")"
            cp "$f" "$OUT/raw/$relative"
        done < <(find "$W/scratch" -type f \( -name '*.parquet' -o -name 'execution.log' -o -name 'normalized-gene-result.json' -o -name '*normalizer*.py' \) -print0)
    fi
    if [ -x "$W/azcopy" ]; then
        "$W/azcopy" copy "$OUT/*" "$BLOB/receipts/benchmark/$DAY/$RUN_ID/" --recursive --log-level=WARNING --output-level=essential || rc=1
    fi
    rm -rf "$W/data" "$W/pipe" "$W/venv" "$W/deps" "$W/scratch" "$W/.uv-cache" "$W/.python" "$W/jre" "$W/exomiser" "$W/wheels" "$W/bridge"
    exit "$rc"
}
trap finish EXIT
exec 3>&1
exec > >(tee "$OUT/benchmark.log") 2>&1
TEE_PID=$!
echo "== $(date -u) start on $(nproc) cores, $(free -g | awk '/Mem/{print $2}') GB RAM, $(df -BG --output=avail "$W" | tail -1) free"

# Mac digests the cloud copy must reproduce (from work-cycles/public-full-genome-benchmark-v1/execution-receipt.json)
JAR_SHA=3e320171fb811fac90ec0c33cd70d21a21e5a8b2cbc0a89cfb88555438295e24
LIB_SHA=53b1d22216e45b57b24c90148f75e462ce52632f8d4bba2b130f0cba09c3222e
DATA_SHA=fe2e5437014cb514c2e76727148967647d9d1210ad390dd326b325f0081ef41d
ADAPTER_SHA=f39f6f3a1e19aee24a109c362195759bf949f5c5edaf154b78c5da16bbb14ce0
ORIGINAL_ADAPTER_SHA=$ADAPTER_SHA
# A separately approved source baseline is recorded alongside, never written into the original Mac receipt.
ADAPTER_SHA=${BIO_ADAPTER_SOURCE_SHA256:-$ADAPTER_SHA}
export BIO_ORIGINAL_ADAPTER_SHA=$ORIGINAL_ADAPTER_SHA BIO_ACTUAL_ADAPTER_SHA=$ADAPTER_SHA
RUNNER_SHA=ed54ef1f41ac0bb1016492b0cc42d3583ed479d10e034a90df359b26ad772b32
CAMPAIGN=aeviant-public-full-genome-compound-het-v1

# --- tools: azcopy, Java (Azul Zulu 21.0.10, the Mac's build), uv + CPython 3.13.12 (the Mac's version) ---
cd "$W"
export PIP_NO_CACHE_DIR=1 UV_CACHE_DIR=$W/.uv-cache UV_PYTHON_INSTALL_DIR=$W/.python   # keep everything on the task disk; 82 GB of data leaves ~10 GB
curl -sL https://aka.ms/downloadazcopy-v10-linux | tar xz --strip-components=1 --wildcards '*/azcopy'
curl -sL https://cdn.azul.com/zulu/bin/zulu21.48.17-ca-jre21.0.10-linux_x64.tar.gz -o jre.tar.gz
mkdir -p jre && tar xzf jre.tar.gz -C jre --strip-components=1
curl -sL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz --strip-components=1
./uv python install 3.13.12 -q
PY=$(./uv python find 3.13.12)
export AZCOPY_AUTO_LOGIN_TYPE=MSI AZCOPY_MSI_CLIENT_ID=$BIO_IDENTITY_CLIENT_ID

# --- inputs from storage: the 2602 data, the Exomiser program, the pinned wheels, the pipeline snapshot ---
./azcopy copy "$BLOB/toolchain/exomiser" . --recursive --log-level=WARNING --output-level=essential
./azcopy copy "$BLOB/toolchain/wheels" . --recursive --log-level=WARNING --output-level=essential
SNAP=${BIO_PIPELINE_SNAPSHOT:?an approved public-only pipeline snapshot is required}
./azcopy copy "$BLOB/pipeline/$SNAP" pipeline.tar.gz --log-level=WARNING --output-level=essential
echo "${BIO_PIPELINE_SNAPSHOT_SHA256:?snapshot digest required}  pipeline.tar.gz" | sha256sum -c -
mkdir -p pipe && tar xzf pipeline.tar.gz -C pipe --strip-components=1 && rm pipeline.tar.gz   # the disk is tight
echo "== $(date -u) inputs ready ($SNAP)"

# --- Python: the pipeline itself in one environment, the bridge's pinned dependencies in a separate directory ---
"$PY" -m venv venv
venv/bin/pip install -q ./pipe
export PYTHONPATH=$W/pipe/src   # preserves every source file in the explicitly fingerprinted package
venv/bin/pip install -q --no-deps --target deps click==8.1.8 packaging==26.0 casefy==1.1.0 jinja2==3.1.6 markupsafe==3.0.3 mypy_extensions==1.1.0 \
    numpy==2.5.2 pandas==3.0.5 phenopackets==2.0.2.post5 polars==1.44.1 polars_runtime_32==1.44.1 protobuf==6.33.6 \
    pyaml==21.10.1 pyserde==0.9.8 python_dateutil==2.9.0.post0 pyyaml==6.0.3 six==1.17.0 typing_extensions==4.16.0 \
    typing_inspect==0.9.0
PYEXE=$(readlink -f venv/bin/python)
sha() { sha256sum "$1" | cut -d' ' -f1; }
# Verify the installed source before invoking the scientific adapter; never replace its expected digest.
venv/bin/aeviant rare-disease-exomiser-adapter-source-manifest > "$OUT/adapter-source-manifest.json"
"$PY" - "$OUT/adapter-source-manifest.json" "$ADAPTER_SHA" <<'PY'
import json, sys
actual = json.load(open(sys.argv[1]))["manifest_sha256"]
if actual != sys.argv[2]:
    raise SystemExit(f"Adapter source changed: expected {sys.argv[2]}, got {actual}")
PY
# Copy the large reference data only after source identity and bridge construction pass.
# the pipeline requires absolute paths for every input
STAGE=bridge
venv/bin/aeviant rare-disease-materialize-pheval-exomiser-bridge "$W/wheels/pheval_exomiser-0.4.12-py3-none-any.whl" \
    "$W/wheels/pheval-0.7.14-py3-none-any.whl" "$W/deps" "$W/bridge" --python-executable "$PYEXE" --python-sha256 "$(sha "$PYEXE")" \
    > "$OUT/bridge-materialize.json" || { cat "$OUT/bridge-materialize.json"; exit 1; }
BRIDGE_RECEIPT=$(readlink -f "$(find "$W/bridge" -maxdepth 1 -name '*receipt*.json' | head -1)")
cp "$BRIDGE_RECEIPT" "$OUT/bridge-receipt.json"
venv/bin/pip freeze > "$OUT/python-packages.txt"
uname -a > "$OUT/host.txt"
jre/bin/java -version > "$OUT/java-version.txt" 2>&1
"$PY" --version > "$OUT/python-version.txt"
STAGE=data
./azcopy copy "$BLOB/exomiser-data/*" data/ --recursive --log-level=WARNING --output-level=essential

# --- the one Linux difference in the method: the data path in the benchmark's properties ---
REL=pipe/release/rare-disease-benchmark-v0.1.0/exomiser-pheval-production-campaign-v1/application.properties
echo "6ca6592266f4dfa13dfe9b0c25c87960e787bc17b014d11f04d08d58208c8cd9  $REL" | sha256sum -c -
sed "s|^exomiser.data-directory=.*|exomiser.data-directory=$W/data|" "$REL" > application.properties
diff <(grep -v '^exomiser.data-directory=' "$REL") <(grep -v '^exomiser.data-directory=' application.properties) \
    && echo "properties: identical to the Mac's apart from the data path"
cp application.properties "$OUT/effective-application.properties"

# --- the run, through the pipeline's own run-and-freeze command ---
RUNNER=$W/pipe/work-cycles/public-full-genome-benchmark-v1/campaign/runner
STAGE=analysis
venv/bin/aeviant rare-disease-run-and-freeze-exomiser-benchmark "$RUNNER/runner-manifest.json" \
    "$OUT/raw-results.json" "$OUT/execution-receipt.json" "$OUT/frozen-results.json" "$W/scratch" \
    --bridge-receipt "$BRIDGE_RECEIPT" --bridge-receipt-sha256 "$(sha "$BRIDGE_RECEIPT")" \
    --bridge-runtime-directory "$W/bridge" --dependency-directory "$W/deps" \
    --python-executable "$PYEXE" --python-sha256 "$(sha "$PYEXE")" \
    --adapter-source-manifest-sha256 "$ADAPTER_SHA" --runner-manifest-sha256 "$RUNNER_SHA" \
    --expected-campaign-id "$CAMPAIGN" --expected-case-count 1 \
    --java "$W/jre/bin/java" --install-directory "$W/exomiser" --data-directory "$W/data" \
    --application-properties "$W/application.properties" \
    --java-sha256 "$(sha jre/bin/java)" --jar-sha256 "$JAR_SHA" --library-manifest-sha256 "$LIB_SHA" \
    --properties-sha256 "$(sha application.properties)" --data-manifest-sha256 "$DATA_SHA" \
    > "$OUT/run.json" || { cat "$OUT/run.json"; exit 1; }
echo "== $(date -u) run finished"

# The reference is fetched only after the independent remote result is frozen.
STAGE=comparison
./azcopy copy "$BLOB/toolchain/benchmark-reference-frozen-results.json" "$OUT/mac-reference-frozen-results.json" --log-level=WARNING --output-level=essential
echo "d029aae3acefca3a1e9069f0f2fe6410b8302e86c6d5e563cf60b2612110a04c  $OUT/mac-reference-frozen-results.json" | sha256sum -c -
./azcopy copy "$BLOB/toolchain/compare_benchmark.py" compare_benchmark.py --log-level=WARNING --output-level=essential
echo "${BIO_COMPARATOR_SHA256:?comparator digest required}  compare_benchmark.py" | sha256sum -c -
comparison_exit=0
"$PY" compare_benchmark.py "$OUT/frozen-results.json" "$OUT/mac-reference-frozen-results.json" "$OUT/linux-runtime-receipt.json" || comparison_exit=$?
"$PY" - "$OUT/linux-runtime-receipt.json" <<'RECEIPT'
import json, os, sys
path = sys.argv[1]
receipt = json.load(open(path))
receipt.update({"original_adapter_source_manifest_sha256": os.environ["BIO_ORIGINAL_ADAPTER_SHA"],
    "approved_adapter_source_manifest_sha256": os.environ["BIO_ACTUAL_ADAPTER_SHA"],
    "same_adapter_source_as_mac": os.environ["BIO_ORIGINAL_ADAPTER_SHA"] == os.environ["BIO_ACTUAL_ADAPTER_SHA"],
    "remote_execution": "frozen before reference download", "runner_level_blindness_demonstrated": False,
    "limitation": "Joint Linux-runtime and source change on one public constructed VARIANT-only EXOME-preset case; no clinical claim."})
json.dump(receipt, open(path, "w"), indent=2)
RECEIPT
exit "$comparison_exit"
