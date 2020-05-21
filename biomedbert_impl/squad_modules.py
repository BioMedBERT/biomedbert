#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call, CalledProcessError
from invoke import run, exceptions


def fine_tune_squad(v1: bool, model_dir: str, train_file: str, predict_file: str,
                    tpu_name: str, tpu_zone: str, gcp_project: str, vocab_file: str,
                    init_checkpoint: str):
    """fine tune squad"""
    use_tpu = True
    version_2_with_negative = True
    output_dir = 'squad_v2/'

    if tpu_name is None:
        tpu_name = 'false'
        use_tpu = False

    if v1:
        version_2_with_negative = False
        output_dir = 'squad_v1/'


    try:
        run('python3 bert/run_squad.py  --vocab_file={}/{}   '
            '--bert_config_file={}/bert_config.json   '
            '--init_checkpoint={}/{}   --do_train=true --train_file={}  '
            '--do_predict=True  --predict_file={}   --train_batch_size=256   '
            '--predict_batch_size=256  --learning_rate=3e-5  --num_train_epochs=2.0  '
            '--max_seq_length=384  --doc_stride=128  --output_dir={}/{}  '
            '--num_tpu_cores=128   --use_tpu={}   --tpu_name={}   --tpu_zone={}   '
            '--gcp_project={}  --version_2_with_negative={}'.format(
            model_dir, vocab_file, model_dir, model_dir, init_checkpoint, train_file,
            predict_file, model_dir, output_dir, use_tpu, tpu_name, tpu_zone, gcp_project,
            version_2_with_negative))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune SQuAD')


'''declare -r BERT_BASE_DIR="${BERT_BASE_DIR:-$1}"
declare -r MODEL_TYPE="${MODEL_TYPE:-$2}"  # base or large
declare -r SQUAD_DIR="${SQUAD_DIR:-squad_data/v1.1}"

python3 bert/run_squad.py \
  --vocab_file="$BERT_BASE_DIR"/biomedbert-8M.txt \
  --bert_config_file="$BERT_BASE_DIR"/bert_config.json \
  --init_checkpoint="$BERT_BASE_DIR"/model.ckpt-1000000 \
  --do_train=True \
  --train_file="$SQUAD_DIR"/train-v1.1.json \
  --do_predict=True \
  --predict_file="$SQUAD_DIR"/dev-v1.1.json \
  --train_batch_size=256 \
  --predict_batch_size=256 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir="$BERT_BASE_DIR"/squad_"$2"_v11/ \
  --use_tpu=True \
  --tpu_name="$TPU_NAME"'''
