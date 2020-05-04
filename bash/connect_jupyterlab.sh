#!/usr/bin/env bash
: '
The following script connects to JuputerLab on
deep learning VM compute instance on GCP.
'

declare -r PROJECT_ID="${PROJECT_ID:-$1}"
declare -r ZONE="${ZONE:-$2}"
declare -r INSTANCE_NAME="${INSTANCE_NAME:-$3}"

echo "PROJECT_ID=${PROJECT_ID}"
echo "ZONE=${ZONE}"
echo "INSTANCE_NAME=${INSTANCE_NAME}"
echo ""

gcloud compute ssh --project "$PROJECT_ID" --zone "$ZONE" \
  "$INSTANCE_NAME" -- -L 8080:localhost:8080
