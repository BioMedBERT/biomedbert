#!/usr/bin/env bash
: '
The following script is for finetuning
classification tasks from the GLUE dataset.
'

declare -r DATASET="${DATASET:-$1}" # MRPC
declare -r BERT_BASE_DIR="${INSTANCE_NAME:-$2}" # gs://ekaba-assets/bert_model_sat_18th_april
declare -r INIT_CHECKPOINT="${INIT_CHECKPOINT:-$3}"
declare -r VOCAB_FILE="${VOCAB_FILE:-$4}"
declare -r TPU_NAME="${TPU_NAME:-$5}"
declare -r TPU_ZONE="${TPU_ZONE:-$6}"
declare -r GCP_PROJECT="${GCP_PROJECT:-$6}"

if [ "${TPU_NAME}" = "false" ]; then USE_TPU=false; else USE_TPU=true; fi

echo USE_TPU="$USE_TPU"


python3 bert/run_classifier.py \
  --task_name="$DATASET" \
  --do_train=true \
  --do_eval=true \
  --data_dir=glue_data/"$DATASET" \
  --vocab_file="$BERT_BASE_DIR"/"$VOCAB_FILE" \
  --bert_config_file="$BERT_BASE_DIR"/bert_config.json \
  --init_checkpoint="$BERT_BASE_DIR"/"$INIT_CHECKPOINT" \
  --max_seq_length=128 \
  --train_batch_size=128 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir="$BERT_BASE_DIR"/"$DATASET"_output \
  --num_tpu_cores=128 \
  --use_tpu="$USE_TPU" \
  --tpu_name="$TPU_NAME" \
  --tpu_zone="$TPU_ZONE" \
  --gcp_project="$GCP_PROJECT"
