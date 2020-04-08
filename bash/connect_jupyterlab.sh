#!/bin/bash
: '
The following script connects to JuputerLab on
deep learning VM compute instance on GCP.
'

echo "PROJECT_ID=$1"
echo "ZONE=$2"
echo "INSTANCE_NAME=$3"
echo ""

gcloud compute ssh --project "$1" --zone "$2" \
  "$3" -- -L 8080:localhost:8080