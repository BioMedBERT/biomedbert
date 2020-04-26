#!/bin/bash
: '
The following script is for running ner BC2GM
'

BERT_BASE_DIR=$1
MODEL_TYPE=$2 # base or large
NER_DIR=datasets/NER/BC2GM
OUTPUT_DIR='ner_outputs_BC2GM'

mkdir -p $OUTPUT_DIR
python3 biobert/run_ner.py \
  --do_train=true \
  --do_eval=true \
  --vocab_file='$BERT_BASE_DIR'/vocab.txt \
  --bert_config_file='$BERT_BASE_DIR'/bert_config.json \
  --init_checkpoint='$BERT_BASE_DIR'/model.ckpt-1000000 \
  --num_train_epochs=10.0 \
  --data_dir='$NER_DIR' \
  --output_dir='$OUTPUT_DIR'