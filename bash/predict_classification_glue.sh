#!/usr/bin/env bash
: '
The following script is for predicting
classification tasks from the GLUE dataset.
'

declare -r BERT_BASE_DIR="${BERT_BASE_DIR:-$1}"
declare -r DATASET="${DATASET:-$2}" # MRPC
declare -r INIT_CHECKPOINT="${INIT_CHECKPOINT:-$3}"
declare -r VOCAB_FILE="${VOCAB_FILE:-$4}"

python3 bert/run_classifier.py \
  --task_name="$DATASET" \
  --do_predict=true \
  --data_dir=glue_data/"$DATASET" \
  --vocab_file="$BERT_BASE_DIR"/"$VOCAB_FILE" \
  --bert_config_file="$BERT_BASE_DIR"/bert_config.json \
  --init_checkpoint="$BERT_BASE_DIR"/"$INIT_CHECKPOINT" \
  --max_seq_length=128 \
  --output_dir="$BERT_BASE_DIR"/"$DATASET"_output
