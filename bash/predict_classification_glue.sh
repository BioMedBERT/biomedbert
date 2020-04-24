#!/bin/bash
: '
The following script is for predicting
classification tasks from the GLUE dataset.
'

BERT_BASE_DIR=$1 #/path/to/bert/uncased_L-12_H-768_A-12
DATASET=$2 #MRPC

python3 bert/run_classifier.py \
  --task_name="$DATASET" \
  --do_predict=true \
  --data_dir=glue_data/"$DATASET" \
  --vocab_file="$BERT_BASE_DIR"/biomedbert-8M.txt \
  --bert_config_file="$BERT_BASE_DIR"/bert_config.json \
  --init_checkpoint="$BERT_BASE_DIR"/model.ckpt-1000000 \
  --max_seq_length=128 \
  --output_dir="$BERT_BASE_DIR"/"$DATASET"_output