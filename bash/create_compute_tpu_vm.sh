#!/bin/bash
: '
The following script creates a
deep learning VM compute instance on GCP.
'

echo "PROJECT_ID=$1"

IMAGE=--image=tf2-latest-cpu-20200326
INSTANCE_NAME=$1 #for-ekaba
GCP_LOGIN_NAME=dvdbisong@gmail.com
ZONE="us-central1-a"
TPU_NAME=$INSTANCE_NAME"-tpu"

gcloud compute instances create "${INSTANCE_NAME}" \
      --machine-type=n1-highmem-32 \
      --zone=$ZONE \
      --scopes=cloud-platform \
      --min-cpu-platform="Intel Skylake" \
      ${IMAGE} \
      --image-project=deeplearning-platform-release \
      --image-project=ml-images \
      --image-family tf2-2-0-cpu \
      --boot-disk-size=300GB \
      --boot-disk-type=pd-ssd \
      --boot-disk-device-name="${INSTANCE_NAME}" \
      --metadata="proxy-user-mail=${GCP_LOGIN_NAME}"\
        startup-script="echo export TPU_NAME=$TPU_NAME > /etc/profile.d/tpu-env.sh"


gcloud compute tpus create "$TPU_NAME" \
  --zone=$ZONE \
  --network default \
  --range 10.240.1.0 \
  --accelerator-type 'v2-128' \
  --version 2.1