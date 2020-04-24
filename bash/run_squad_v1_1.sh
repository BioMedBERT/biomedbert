#!/bin/bash
: '
The following script is for running squad v1.1.
'

BERT_BASE_DIR=$1
MODEL_TYPE=$2 # base or large
SQUAD_DIR=squad_data/v1.1

python3 bert/run_squad.py \
  --vocab_file="$BERT_BASE_DIR"/biomedbert-8M.txt \
  --bert_config_file="$BERT_BASE_DIR"/bert_config.json \
  --init_checkpoint="$BERT_BASE_DIR"/model.ckpt-1000000 \
  --do_train=True \
  --train_file="$SQUAD_DIR"/train-v1.1.json \
  --do_predict=True \
  --predict_file="$SQUAD_DIR"/dev-v1.1.json \
  --train_batch_size=256 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir="$BERT_BASE_DIR"/squad_"$2"_v11/ \
  --use_tpu=True \
  --tpu_name="$TPU_NAME"