#!/bin/bash
# Little Guy's Azure workspace, stage 2: the autopilot. Every service gets work on a timer inside Azure, so the
# workspace runs every night with nobody at the keyboard. Same rules as stage 1: dry run by default, GO=1 creates,
# no passwords on any machine (each job signs in with its own managed identity), his memory and conversations
# never leave the house (only brain files, the persona, test results and numeric telemetry come here).
#
#   tools/azure/stage2.sh          dry run: prints every command, creates nothing
#   GO=1 tools/azure/stage2.sh     creates (spends credits)
#
# A. the compile VM wakes at 00:30 UTC (lg-wake job), builds and tests the engine, and sleeps at 07:00 UTC
set -euo pipefail
cd "$(dirname "$0")/../.."

RG=littleguy
LOC=eastus
SUFFIX=$(grep -E '^LITTLEGUY_AZ_SUFFIX=' littleguy.env | cut -d= -f2)
ST=lg$SUFFIX; VM=lg-compile; CAE=lg-jobs
SUB=$(az account show --query id -o tsv)
SCOPE=/subscriptions/$SUB/resourceGroups/$RG/providers

run() { echo "+ $*"; [ -n "${GO:-}" ] && "$@" -o none; return 0; }
get() { [ -n "${GO:-}" ] && "$@" -o tsv || echo "<$*>"; }   # read a value (only when creating)

echo "subscription: $(az account show --query name -o tsv) | group $RG | storage $ST"
[ -z "${GO:-}" ] && echo "(dry run: nothing is created)"

# --- A. nightly oven check-up on the compile VM ---
run az storage container create --account-name "$ST" -n build --auth-mode login
run az vm identity assign -g "$RG" -n "$VM"
VM_ID=$(get az vm show -g "$RG" -n "$VM" --query identity.principalId)
run az role assignment create --assignee-object-id "$VM_ID" --assignee-principal-type ServicePrincipal \
    --role "Storage Blob Data Contributor" --scope "$SCOPE/Microsoft.Storage/storageAccounts/$ST"
run az vm start -g "$RG" -n "$VM"
echo "+ az vm run-command invoke ... setup-vm.sh (build tools, Azure CLI, the boot service)"
[ -n "${GO:-}" ] && az vm run-command invoke -g "$RG" -n "$VM" --command-id RunShellScript --scripts @tools/azure/vm/setup-vm.sh \
    --parameters "$ST" "$(base64 < tools/azure/vm/nightly.sh | tr -d '\n')" --query "value[0].message" -o tsv | tail -5

ENV_ID=$(get az containerapp env show -g "$RG" -n "$CAE" --query id)
sed "s|__ENV_ID__|$ENV_ID|" tools/azure/jobs/lg-wake.yaml > "${TMPDIR:-/tmp}/lg-wake.yaml"
run az containerapp job create -g "$RG" -n lg-wake --yaml "${TMPDIR:-/tmp}/lg-wake.yaml"   # the CLI's --args can't take "-c"
WAKE_ID=$(get az containerapp job show -g "$RG" -n lg-wake --query identity.principalId)
run az role assignment create --assignee-object-id "$WAKE_ID" --assignee-principal-type ServicePrincipal \
    --role "Virtual Machine Contributor" --scope "$SCOPE/Microsoft.Compute/virtualMachines/$VM"


# --- B. nightly brain tests (Container Apps) graded by the AI judge (AI Services, deployment "judge": gpt-4.1) ---
run az cognitiveservices account deployment create -g "$RG" -n "lg-ai-$SUFFIX" --deployment-name judge \
    --model-name gpt-4.1 --model-version 2025-04-14 --model-format OpenAI --sku-name Standard --sku-capacity 50
sed -e "s|__ENV_ID__|$ENV_ID|" -e "s|__STORAGE__|$ST|" -e "s|__SUFFIX__|$SUFFIX|" tools/azure/jobs/lg-tests.yaml > "${TMPDIR:-/tmp}/lg-tests.yaml"
run az containerapp job create -g "$RG" -n lg-tests --yaml "${TMPDIR:-/tmp}/lg-tests.yaml"
TESTS_ID=$(get az containerapp job show -g "$RG" -n lg-tests --query identity.principalId)
run az role assignment create --assignee-object-id "$TESTS_ID" --assignee-principal-type ServicePrincipal \
    --role "Storage Blob Data Contributor" --scope "$SCOPE/Microsoft.Storage/storageAccounts/$ST"
run az role assignment create --assignee-object-id "$TESTS_ID" --assignee-principal-type ServicePrincipal \
    --role "Cognitive Services OpenAI User" --scope "$SCOPE/Microsoft.CognitiveServices/accounts/lg-ai-$SUFFIX"

echo "stage 2 A-B done: tools/azure/push_source.sh sends the engine source, the tests and the persona"
