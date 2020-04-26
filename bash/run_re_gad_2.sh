#!/bin/bash
: '
The following script is for running RE GAD 2.
'

BERT_BASE_DIR=$1
MODEL_TYPE=$2 # base or large
RE_DIR=datasets/RE/GAD/2
TASK_NAME=gad
OUTPUT_DIR='re_outputs_gad_2'

mkdir -p $OUTPUT_DIR

python biobert/run_re.py \
	--task_name='$TASK_NAME' \
	--do_train=true \
	--do_eval=true \
	--do_predict=true \
	--vocab_file='$BERT_BASE_DIR'/vocab.txt \
	--bert_config_file='$BERT_BASE_DIR'/bert_config.json \
	--init_checkpoint='$BERT_BASE_DIR'/model.ckpt-1000000 \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=2e-5 \
	--num_train_epochs=3.0 \
	--do_lower_case=false \
	--data_dir='$RE_DIR' \
	--output_dir='$OUTPUT_DIR'