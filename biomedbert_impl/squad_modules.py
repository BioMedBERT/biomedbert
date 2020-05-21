#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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
            '--do_predict=True  --predict_file={}   --train_batch_size=16   '
            '--predict_batch_size=16  --learning_rate=3e-5  --num_train_epochs=2.0  '
            '--max_seq_length=384  --doc_stride=128  --output_dir={}/{}  '
            '--num_tpu_cores=128   --use_tpu={}   --tpu_name={}   --tpu_zone={}   '
            '--gcp_project={}  --version_2_with_negative={}'.format(
            model_dir, vocab_file, model_dir, model_dir, init_checkpoint, train_file,
            predict_file, model_dir, output_dir, use_tpu, tpu_name, tpu_zone, gcp_project,
            version_2_with_negative))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune SQuAD')


def evaluate_squad(evaluate_file: str, predict_file: str, prediction_json: str):
    """evaluate squad"""

    try:
        if not os.path.exists('squad_evaluate'):
            run('mkdir squad_evaluate')
        run('gsutil cp {} ./squad_evaluate/'.format(evaluate_file))
        run('gsutil cp {} ./squad_evaluate/'.format(predict_file))
        run('gsutil cp {} ./squad_evaluate/'.format(prediction_json))
        run('python ./squad_evaluate/{} ./squad_evaluate/{} ./squad_evaluate/predictions.json'.format(
            evaluate_file.split('/')[-1],
            predict_file.split('/')[-1]))
    except exceptions.UnexpectedExit:
        print('Cannot evaluate SQuAD')