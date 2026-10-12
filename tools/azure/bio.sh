#!/bin/bash
# The bioscience side of the Azure workspace (Aeviant pipeline), kept apart from Little Guy in its own resource group.
# Stage 1: storage for the pipeline's public reference data (Exomiser 15.1.0 data release 2602) and a Batch account
# for its compute. Nothing here runs an analysis: the pipeline's own gates decide that, and its operator authorizes
# it. Only public data comes here; patient data stays off Azure.
#
#   tools/azure/bio.sh          dry run: prints every command, creates nothing
#   GO=1 tools/azure/bio.sh     creates (spends credits)
set -euo pipefail
cd "$(dirname "$0")/../.."

LOC=eastus
RG=aeviant-bio
SUFFIX=$(grep -E '^LITTLEGUY_AZ_SUFFIX=' littleguy.env | cut -d= -f2)
ST=aevbio$SUFFIX; BATCH=aevbio$SUFFIX
SUB=$(az account show --query id -o tsv)

run() { echo "+ $*"; [ -n "${GO:-}" ] && "$@" -o none; return 0; }

echo "subscription: $(az account show --query name -o tsv) | group $RG | storage $ST | batch $BATCH"
[ -z "${GO:-}" ] && echo "(dry run: nothing is created)"

run az provider register --namespace Microsoft.Batch
run az group create -n "$RG" -l "$LOC"

# Storage (cool tier: large reference data read a few times a day): Exomiser data, pipeline code, run receipts
run az storage account create -g "$RG" -n "$ST" -l "$LOC" --sku Standard_LRS --kind StorageV2 --access-tier Cool \
    --min-tls-version TLS1_2 --allow-blob-public-access false
ME=$(az ad signed-in-user show --query id -o tsv)
run az role assignment create --assignee "$ME" --role "Storage Blob Data Contributor" \
    --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$ST"
for c in exomiser-data pipeline receipts; do
    run az storage container create --account-name "$ST" -n "$c" --auth-mode login
done

# Batch: pools and nightly job schedules for the pipeline's public-data work (added once the operator authorizes runs)
run az batch account create -g "$RG" -n "$BATCH" -l "$LOC" --storage-account "$ST"

echo "bio stage 1 done. Upload the reference data with tools/azure/bio_upload.sh"
