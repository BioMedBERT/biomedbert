#!/usr/bin/env bash
: '
The following script creates a TPU instance on GCP.
'

declare -r TPU_NAME="${TPU_NAME:-$1}"
declare -r ZONE="${ZONE:-$2}" # "europe-west4-a"

gcloud compute tpus create "$TPU_NAME" \
  --zone="$ZONE" \
  --network 'default' \
  --range '10.250.1.0' \
  --accelerator-type 'v3-128' \
  --version '2.1'
