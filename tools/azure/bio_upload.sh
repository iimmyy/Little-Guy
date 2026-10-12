#!/bin/bash
# Uploads the Aeviant pipeline's Exomiser reference data (public release 2602: hg19, hg38, phenotype; ~82 GB) and a
# snapshot of the pipeline's code (no runtime data) to the bio storage. Release 2602 isn't on Monarch's public index
# any more, so the exact local copy goes up to keep the pipeline's recorded fingerprints valid.
#   tools/azure/bio_upload.sh <path to aeviant-pipeline>
set -euo pipefail
cd "$(dirname "$0")/../.."
PIPE=${1:?path to the aeviant-pipeline checkout}
ST=aevbio$(grep -E '^LITTLEGUY_AZ_SUFFIX=' littleguy.env | cut -d= -f2)
TMP=$(mktemp -d)
python3 tools/azure/bio/public_snapshot.py "$PIPE" "$TMP/pipeline.tar.gz"
NAME="pipeline-public-benchmark-$(date -u +%F).tar.gz"
az storage blob upload --account-name "$ST" -c pipeline -n "$NAME" -f "$TMP/pipeline.tar.gz" \
    --auth-mode login --overwrite --no-progress -o none
az storage blob upload --account-name "$ST" -c pipeline -n "$NAME.json" -f "$TMP/pipeline.tar.gz.json" \
    --auth-mode login --overwrite --no-progress -o none
echo "pipeline code sent ($(du -h "$TMP/pipeline.tar.gz" | cut -f1))"
rm -rf "$TMP"
for part in 2602_phenotype 2602_hg38 2602_hg19; do
    t0=$(date +%s)
    az storage blob upload-batch --account-name "$ST" -d exomiser-data --destination-path "$part" \
        -s "$PIPE/runtime-data/exomiser/data/$part" --auth-mode login --max-connections 8 --no-progress -o none
    echo "uploaded $part in $(( $(date +%s) - t0 )) s"
done
