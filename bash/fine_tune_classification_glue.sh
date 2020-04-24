#!/bin/bash
: '
The following script is for finetuning
classification tasks from the GLUE dataset.
'

DATASET=$1 #MRPC
BERT_BASE_DIR=gs://ekaba-assets/bert_model_sat_18th_april # $2


python3 bert/run_classifier.py
  --task_name=MRPC
  --do_train=true
  --do_eval=true
  --data_dir=glue_data/"$DATASET"
  --vocab_file="$BERT_BASE_DIR"/biomedbert-8M.txt
  --bert_config_file="$BERT_BASE_DIR"/bert_config.json
  --init_checkpoint="$BERT_BASE_DIR"/model.ckpt-1000000
  --max_seq_length=128
  --train_batch_size=32
  --learning_rate=2e-5
  --num_train_epochs=3.0
  --output_dir="$BERT_BASE_DIR"/"$DATASET"_output

