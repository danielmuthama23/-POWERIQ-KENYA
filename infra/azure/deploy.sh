#!/usr/bin/env bash
# Azure Container Apps deployment (NOT executed or validated in the build sandbox: review before use).
# Prereqs: az CLI logged in, a subscription, and a Key Vault holding POWERIQ-API-KEYS (and optionally OPENAI-API-KEY).
set -euo pipefail
RG=${RG:-poweriq-rg}; LOC=${LOC:-westeurope}; ACR=${ACR:?set ACR name}; ENVN=${ENVN:-poweriq-env}; APP=${APP:-poweriq}; KV=${KV:?set Key Vault name}
az group create -n "$RG" -l "$LOC"
az acr create -n "$ACR" -g "$RG" --sku Basic
az acr build -r "$ACR" -t poweriq:latest .
az monitor log-analytics workspace create -g "$RG" -n "$APP-logs"
az containerapp env create -n "$ENVN" -g "$RG" -l "$LOC"
az containerapp create -n "$APP" -g "$RG" --environment "$ENVN" --image "$ACR.azurecr.io/poweriq:latest" \
  --registry-server "$ACR.azurecr.io" --target-port 8000 --ingress external --min-replicas 1 --max-replicas 2 \
  --cpu 1.0 --memory 2.0Gi --system-assigned \
  --env-vars POWERIQ_ENV=production POWERIQ_API_KEYS=secretref:api-keys POWERIQ_AUDIT_DB=/data/audit.db \
  --secrets api-keys="$(az keyvault secret show --vault-name "$KV" -n POWERIQ-API-KEYS --query value -o tsv)"
# Grant the app's managed identity read access to Key Vault secrets, then switch --secrets to Key Vault references.
echo "Health probe path: /health  (ready=true after ~25 s model training)"
