#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import tensorflow as tf
from invoke import run, exceptions

log = logging.getLogger('biomedbert')
log.setLevel(logging.INFO)


def fine_tune_squad(v1: bool, model_type: str, bucket_name: str, model_dir: str, train_file: str, predict_file: str,
                    tpu_name: str, tpu_zone: str, gcp_project: str, tpu_cores: str):
    """fine tune squad"""
    use_tpu = True
    sub_folder = 'v2.0'
    output_dir = 'squad_v2/'
    config = 'large_bert_config.json'
    version_2_with_negative = True

    num_tpu_cores = 8
    if tpu_cores is not None:
        num_tpu_cores = int(tpu_cores)

    if tpu_name is None:
        tpu_name = 'false'
        use_tpu = False

    if model_type == 'base':
        # bert base
        config = 'base_bert_config.json'
    elif model_type == 'large':
        # bert large
        config = 'large_bert_config.json'
    else:
        log.info('No config file')
        sys.exit(1)

    if v1:
        version_2_with_negative = False
        output_dir = 'squad_v1/'
        sub_folder = 'v1.1'

    init_checkpoint = tf.train.latest_checkpoint('gs://{}/{}'.format(bucket_name, model_dir))
    vocab_file = 'gs://{}/{}/vocab.txt'.format(bucket_name, model_dir)
    bert_config_file = 'gs://{}/{}/{}'.format(bucket_name, model_dir, config)
    output_dirs = 'gs://{}/{}/{}'.format(bucket_name, model_dir, output_dir)
    train_file_path = 'gs://{}/squad_data/{}/{}'.format(bucket_name, sub_folder, train_file)
    predict_file_path = 'gs://{}/squad_data/{}/{}'.format(bucket_name, sub_folder, predict_file)

    try:
        run('python3 bert/run_squad.py  --vocab_file={}   '
            '--bert_config_file={}   '
            '--init_checkpoint={}   --do_train=true --train_file={}  '
            '--do_predict=True  --predict_file={}   --train_batch_size=16   '
            '--predict_batch_size=16  --learning_rate=3e-5  --num_train_epochs=2.0  '
            '--max_seq_length=384  --doc_stride=128  --output_dir={}  '
            '--num_tpu_cores=128   --use_tpu={}   --tpu_name={}   --tpu_zone={}   '
            '--gcp_project={} --version_2_with_negative={}'.format(
            vocab_file, bert_config_file, init_checkpoint, train_file_path,
            predict_file_path, output_dirs, use_tpu, tpu_name, tpu_zone, gcp_project,
            version_2_with_negative))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune SQuAD')


def evaluate_squad(v1: bool, bucket_name: str, model_dir: str, evaluate_file: str,
                   predict_file: str, prediction_json: str):
    """evaluate squad"""
    sub_folder = 'v2.0'
    output_dir = 'squad_v2'

    if v1:
        output_dir = 'squad_v1'
        sub_folder = 'v1.1'

    try:
        if not os.path.exists('squad_evaluate'):
            run('mkdir squad_evaluate')
        run('gsutil cp gs://{}/squad_data/{} ./squad_evaluate/'.format(bucket_name, evaluate_file))
        run('gsutil cp gs://{}/squad_data/{}/{} ./squad_evaluate/'.format(bucket_name, sub_folder, predict_file))
        run('gsutil cp gs://{}/{}/{}/{} ./squad_evaluate/'.format(bucket_name, model_dir, output_dir, prediction_json))

        if v1:
            run('python3 ./squad_evaluate/{} ./squad_evaluate/{} ./squad_evaluate/{}'.format(
                evaluate_file, predict_file, prediction_json))
        else:
            run('gsutil cp gs://{}/{}/{}/null_odds.json ./squad_evaluate/'.format(bucket_name, model_dir, output_dir))
            run('python3 ./squad_evaluate/{} ./squad_evaluate/{} ./squad_evaluate/{} '
                './squad_evaluate/null_odds.json'.format(evaluate_file, predict_file, prediction_json))
    except exceptions.UnexpectedExit:
        print('Cannot evaluate SQuAD')
