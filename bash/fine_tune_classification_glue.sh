#!/usr/bin/env bash
: '
The following script is for finetuning
classification tasks from the GLUE dataset.
'

declare -r DATASET="${DATASET:-$1}" # MRPC
declare -r BERT_BASE_DIR="${INSTANCE_NAME:-$2}" # gs://ekaba-assets/bert_model_sat_18th_april


python3 bert/run_classifier.py \
  --task_name="$DATASET" \
  --do_train=true \
  --do_eval=true \
  --data_dir=glue_data/"$DATASET" \
  --vocab_file="$BERT_BASE_DIR"/biomedbert-8M.txt \
  --bert_config_file="$BERT_BASE_DIR"/bert_config.json \
  --init_checkpoint="$BERT_BASE_DIR"/model.ckpt-1000000 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir="$BERT_BASE_DIR"/"$DATASET"_output
