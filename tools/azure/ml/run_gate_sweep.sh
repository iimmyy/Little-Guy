#!/bin/bash
# Inside the nightly ML job: fetch public text (WikiText-2 test, CC BY-SA), then run the per-layer gate search on
# brain C against it. His conversations never come to Azure; this is the cross-check on general text.
#   run_gate_sweep.sh <master folder> <brain C gguf> <results folder>
set -euo pipefail
python - <<'PY'
import pandas as pd
url = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
open("wikitext.txt", "w").write("".join(pd.read_parquet(url)["text"]))
PY
DAY=$(date -u +%F)
mkdir -p "$3/$DAY"
SEND_SKIP_MASTER="$1" SEND_SKIP_BRAIN="$2" SEND_SKIP_TEXT=wikitext.txt SEND_SKIP_OUT="$3/$DAY/gate_sweep.json" \
    python research/gate-assay/send_skip.py greedy 4 3 2>&1 | grep -v "Loading weights" | tee "$3/$DAY/gate_sweep.log"
