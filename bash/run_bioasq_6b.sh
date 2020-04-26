#!/bin/bash
: '
The following script is for running bioasq
'

BERT_BASE_DIR=$1
MODEL_TYPE=$2 # base or large
QA_DIR=datasets/QA/BioASQ
OUTPUT_DIR=qa_outputs

python3 run_qa.py \
  --do_train=True \
  --do_predict=True \
  --vocab_file='$BERT_BASE_DIR'/vocab.txt \
  --bert_config_file='$BERT_BASE_DIR'/bert_config.json \
  --init_checkpoint='$BERT_BASE_DIR'/model.ckpt-1000000 \
  --max_seq_length=384 \
  --train_batch_size=12 \
  --learning_rate=5e-6 \
  --doc_stride=128 \
  --num_train_epochs=5.0 \
  --do_lower_case=False \
  --train_file='$QA_DIR'/BioASQ-train-factoid-6b.json \
  --predict_file='$QA_DIR'/BioASQ-test-factoid-6b-1.json \
  --output_dir='$OUTPUT_DIR'
