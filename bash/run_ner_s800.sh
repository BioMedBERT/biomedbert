#!/usr/bin/env bash
: '
The following script is for running ner s800
'

declare -r BERT_BASE_DIR="${BERT_BASE_DIR:-$1}"
declare -r MODEL_TYPE="${MODEL_TYPE:-$2}"  # base or large
declare -r NER_DIR="${NER_DIR:-datasets/NER/s800}"
declare -r OUTPUT_DIR='ner_outputs_s800'

mkdir -p "$OUTPUT_DIR"
python3 biobert/run_ner.py \
  --do_train=true \
  --do_eval=true \
  --vocab_file="$BERT_BASE_DIR"/vocab.txt \
  --bert_config_file="$BERT_BASE_DIR"/bert_config.json \
  --init_checkpoint="$BERT_BASE_DIR"/model.ckpt-1000000 \
  --num_train_epochs=10.0 \
  --data_dir="$NER_DIR" \
  --output_dir="$OUTPUT_DIR"
