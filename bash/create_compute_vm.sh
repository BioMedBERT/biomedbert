#!/usr/bin/env bash
: '
The following script creates a
deep learning VM compute instance on GCP.
'

declare -r INSTANCE_NAME="${INSTANCE_NAME:-$1}"
declare -r ZONE="${ZONE:-$2}" # "europe-west4-a"

gcloud compute instances create solvm \
      --machine-type='n1-standard-1' \
      --scopes='https://www.googleapis.com/auth/cloud-platform' \
      --zone=us-central1-a \
      --image-project='deeplearning-platform-release' \
      --image-family='tf2-2-0-cu100' \
      --boot-disk-size='300GB'
