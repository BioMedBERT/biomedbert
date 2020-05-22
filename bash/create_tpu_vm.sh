#!/usr/bin/env bash
: '
The following script creates a TPU instance on GCP.
'

declare -r TPU_NAME="${TPU_NAME:-$1}"
declare -r ZONE="${ZONE:-$2}" # "europe-west4-a"
declare -r PREEMPTIBLE="${PREEMPTIBLE:-$3}" # "is preemptib;e"

if [ "$PREEMPTIBLE" = "yes" ];
then
  gcloud compute tpus create "$TPU_NAME" \
  --zone="$ZONE" \
  --network 'default' \
  --range '10.250.1.0' \
  --accelerator-type 'v3-128' \
  --version '2.1' \
  --preemptible
else
  gcloud compute tpus create "$TPU_NAME" \
  --zone="$ZONE" \
  --network 'default' \
  --range '10.250.1.0' \
  --accelerator-type 'v3-128' \
  --version '2.1'
fi