#!/bin/bash
: '
The following script creates a
deep learning VM compute instance on GCP.
'

INSTANCE_NAME=$1
ZONE="us-central1-a"
TPU_NAME=$INSTANCE_NAME"-tpu"

gcloud compute instances create "${INSTANCE_NAME}" \
      --machine-type=n1-highmem-64 \
      --zone=$ZONE \
      --image-project=deeplearning-platform-release \
      --image-family=tf2-2-0-cu100 \
      --boot-disk-size=300GB \
      --metadata=startup-script="echo export TPU_NAME=$TPU_NAME > /etc/profile.d/tpu-env.sh"

gcloud compute tpus create "$TPU_NAME" \
  --zone=$ZONE \
  --network default \
  --range 10.240.1.0 \
  --accelerator-type 'v2-128' \
  --version 2.1