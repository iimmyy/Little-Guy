#!/bin/bash
# Little Guy's Azure workspace, stage 1: create the services (no project data, no GPU).
# Every service has a real job (listed by number below). Stage 2 wires the work in (images, jobs, telemetry).
#
#   tools/azure/setup.sh          dry run: prints every command, creates nothing
#   GO=1 tools/azure/setup.sh     creates (spends credits), skipping anything that already exists
#
# Safety rails: one resource group, the compile VM shuts down nightly and only accepts SSH from this Mac's IP,
# the ledger database only accepts this Mac's IP, secrets go straight into Key Vault and are never printed.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOC=eastus
PG_LOC=${PG_LOC:-centralus}   # East US refuses new PostgreSQL servers on this subscription ("location is restricted")
RG=littleguy
ENV_FILE=littleguy.env
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE (copy littleguy.env.example)"; exit 1; }
# a short random suffix keeps globally unique names (storage, registry, vault, database) ours; kept in littleguy.env
SUFFIX=$(grep -E '^LITTLEGUY_AZ_SUFFIX=' "$ENV_FILE" | cut -d= -f2 || true)
if [ -z "$SUFFIX" ]; then
    SUFFIX=$(openssl rand -hex 3 | cut -c1-5)
    [ -n "${GO:-}" ] && echo "LITTLEGUY_AZ_SUFFIX=$SUFFIX" >> "$ENV_FILE"
fi
ST=lg$SUFFIX; KV=lg-kv-$SUFFIX; ACR=lgacr$SUFFIX; PG=lg-ledger-$SUFFIX; IOT=lg-iot-$SUFFIX
SPEECH=lg-speech-$SUFFIX; AI=lg-ai-$SUFFIX; WEB=lg-dashboard-$SUFFIX; VM=lg-compile; LOGS=lg-logs; CAE=lg-jobs
MYIP=$(curl -s https://api.ipify.org)

run() {   # print the command (secrets shown as ***); run it only with GO=1
    local shown=() hide=0
    for a in "$@"; do
        if [ $hide = 1 ]; then shown+=("***"); hide=0; else shown+=("$a"); fi
        case "$a" in --admin-password|--logs-workspace-key|--value) hide=1 ;; esac
    done
    echo "+ ${shown[*]}"
    [ -n "${GO:-}" ] && "$@" -o none
    return 0
}
exists() { [ -n "${GO:-}" ] && az resource list -g "$RG" --query "[?name=='$1'] | length(@)" -o tsv 2>/dev/null | grep -q '^[1-9]'; }

echo "subscription: $(az account show --query name -o tsv) | region $LOC | group $RG | suffix $SUFFIX | this Mac's IP $MYIP"
[ -z "${GO:-}" ] && echo "(dry run: nothing is created)"

for p in Microsoft.App Microsoft.ContainerInstance Microsoft.ContainerRegistry Microsoft.Devices Microsoft.DBforPostgreSQL \
         Microsoft.Web Microsoft.MachineLearningServices Microsoft.CognitiveServices Microsoft.EventGrid Microsoft.KeyVault \
         Microsoft.OperationalInsights Microsoft.Storage Microsoft.Compute Microsoft.Insights; do
    run az provider register --namespace "$p"
done
for x in containerapp ml; do run az extension add --name "$x" --upgrade --yes; done

run az group create -n "$RG" -l "$LOC"

# 1. Log Analytics: logs from every job and the compile VM
exists "$LOGS" || run az monitor log-analytics workspace create -g "$RG" -n "$LOGS" -l "$LOC"

# 2. Blob Storage: brain versions, test results, the telemetry archive
if ! exists "$ST"; then
    run az storage account create -g "$RG" -n "$ST" -l "$LOC" --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 --allow-blob-public-access false
fi
for c in brains results telemetry; do
    run az storage container create --account-name "$ST" -n "$c" --auth-mode login
done

# 3. Key Vault: the pipelines' secrets (the ledger password is generated straight into it)
exists "$KV" || run az keyvault create -g "$RG" -n "$KV" -l "$LOC"
ME=$(az ad signed-in-user show --query id -o tsv)
SUB=$(az account show --query id -o tsv)
run az role assignment create --assignee "$ME" --role "Key Vault Secrets Officer" --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.KeyVault/vaults/$KV"
run az role assignment create --assignee "$ME" --role "Storage Blob Data Contributor" --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$ST"

# 4. Container Registry (Premium, a fixed daily fee): toolchain, test-runner and build images
exists "$ACR" || run az acr create -g "$RG" -n "$ACR" -l "$LOC" --sku Premium

# 5. PostgreSQL: the experiment ledger (every brain, every test, what each win saved and where)
if ! exists "$PG"; then
    PGPASS="Lg-$(openssl rand -hex 14)"   # upper, lower, digits and a symbol, as Azure requires
    if [ -n "${GO:-}" ]; then
        az keyvault secret set --vault-name "$KV" -n ledger-admin-password --value "$PGPASS" -o none
    fi
    echo "+ (ledger admin password generated into Key Vault $KV as ledger-admin-password, not printed)"
    run az postgres flexible-server create -g "$RG" -n "$PG" -l "$PG_LOC" --tier Burstable --sku-name Standard_B2s \
        --storage-size 32 --version 16 --admin-user lgadmin --admin-password "$PGPASS" --public-access "$MYIP" --yes
    unset PGPASS
fi

# 6. IoT Hub (B2, a fixed daily fee): the board's heat, throttle and speed history (routed to Blob in stage 2)
exists "$IOT" || run az iot hub create -g "$RG" -n "$IOT" -l "$LOC" --sku B2 --unit 1 --partition-count 2

# 7. Container Apps environment: home for the behavior-test and distortion jobs (jobs are added in stage 2)
if ! exists "$CAE"; then
    LAW_ID=$( [ -n "${GO:-}" ] && az monitor log-analytics workspace show -g "$RG" -n "$LOGS" --query customerId -o tsv || echo "<workspace id>")
    LAW_KEY=$( [ -n "${GO:-}" ] && az monitor log-analytics workspace get-shared-keys -g "$RG" -n "$LOGS" --query primarySharedKey -o tsv || echo "<key>")
    run az containerapp env create -g "$RG" -n "$CAE" -l "$LOC" --logs-workspace-id "$LAW_ID" --logs-workspace-key "$LAW_KEY"
    unset LAW_KEY
fi

# 8. AI Speech: prototyping his ears and voice
exists "$SPEECH" || run az cognitiveservices account create -g "$RG" -n "$SPEECH" -l "$LOC" --kind SpeechServices --sku S0 --yes

# 9. AI Foundry (AI Services): a model that grades his replies beyond our word rules (evaluation only, never training data)
exists "$AI" || run az cognitiveservices account create -g "$RG" -n "$AI" -l "$LOC" --kind AIServices --sku S0 --custom-domain "$AI" --yes

# 10. Static Web Apps (Standard): a private results dashboard (content deployed in stage 2)
exists "$WEB" || run az staticwebapp create -g "$RG" -n "$WEB" -l eastus2 --sku Standard

# 11. Event Grid: "a new brain landed in storage" events that start the tests (subscription added in stage 2)
exists lg-storage-events || run az eventgrid system-topic create -g "$RG" -n lg-storage-events -l "$LOC" \
    --topic-type microsoft.storage.storageaccounts --source "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$ST"

# 12. Machine Learning: tracking for training runs, sharing our storage, vault and registry
exists lg-ml || run az ml workspace create -g "$RG" -n lg-ml -l "$LOC" \
    --storage-account "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$ST" \
    --key-vault "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.KeyVault/vaults/$KV" \
    --container-registry "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.ContainerRegistry/registries/$ACR"

# 13. Virtual Machine (D4as_v4: 4 x86 cores, 16 GB; this subscription is not offered the B-series): the compile box for Qualcomm's QAIRT (the Mac's VM runs out at 11 GB)
if ! exists "$VM"; then
    [ -f ~/.ssh/littleguy_azure ] || { [ -n "${GO:-}" ] && ssh-keygen -t ed25519 -N "" -f ~/.ssh/littleguy_azure -C littleguy-azure >/dev/null; }
    run az vm create -g "$RG" -n "$VM" -l "$LOC" --image Ubuntu2204 --size Standard_D4as_v4 --admin-username lg \
        --ssh-key-values ~/.ssh/littleguy_azure.pub --public-ip-sku Standard --os-disk-size-gb 128 --nsg-rule NONE
    run az network nsg rule create -g "$RG" --nsg-name "${VM}NSG" -n ssh-from-this-mac --priority 1000 \
        --source-address-prefixes "$MYIP" --destination-port-ranges 22 --access Allow --protocol Tcp
    run az vm auto-shutdown -g "$RG" -n "$VM" --time 0700     # 07:00 UTC = midnight Pacific time
fi

# 14. Budget on the group, through the REST API (the CLI's budget command fails with "use filter interface").
#     Email alerts: add your address in the portal, Cost Management > Budgets > littleguy-monthly.
run az rest --method put --url "https://management.azure.com/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Consumption/budgets/littleguy-monthly?api-version=2023-05-01" \
    --body '{"properties":{"category":"Cost","amount":1500,"timeGrain":"Monthly","timePeriod":{"startDate":"2026-10-01T00:00:00Z","endDate":"2027-12-31T00:00:00Z"}}}'

echo "stage 1 done. Container Instances, the Container Apps jobs, Functions and the AI judge's model arrive in stage 2."
