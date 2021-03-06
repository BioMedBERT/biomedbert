#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
import tensorflow as tf
from invoke import run, exceptions

log = logging.getLogger('biomedbert')
log.setLevel(logging.INFO)


def fine_tune_re(re_dataset: str, re_dataset_no: str, model_dir: str, model_type: str, bucket_name: str,
                 tpu_name: str, tpu_zone: str, gcp_project: str, tpu_cores: str):
    """fine tune re"""
    use_tpu = True
    config = 'large_bert_config.json'
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

    init_checkpoint = tf.train.latest_checkpoint('gs://{}/{}'.format(bucket_name, model_dir))
    vocab_file = 'gs://{}/{}/vocab.txt'.format(bucket_name, model_dir)
    bert_config_file = 'gs://{}/{}/{}'.format(bucket_name, model_dir, config)
    output_dir = 'gs://{}/{}/RE_outputs/{}/{}'.format(bucket_name, model_dir, re_dataset, re_dataset_no)
    data_dir = 'gs://{}/datasets/RE/{}/{}'.format(bucket_name, re_dataset, re_dataset_no)

    try:
        run('python3 biobert/run_re.py  --task_name={}  --vocab_file={}   '
            '--bert_config_file={}  --init_checkpoint={}   --do_train=true --do_predict=true  '
            '--max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   '
            '--do_lower_case=false  --predict_batch_size=32  --num_train_epochs=100.0   --data_dir={}   '
            '--output_dir={}  --use_tpu={}  --tpu_name={}   --tpu_zone={}  '
            '--gcp_project={}   --num_tpu_cores={}'.format(
            re_dataset, vocab_file, bert_config_file, init_checkpoint,
            data_dir, output_dir, use_tpu, tpu_name, tpu_zone, gcp_project, num_tpu_cores))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune RE - {}'.format(re_dataset))


def evaluate_re(re_dataset: str, re_dataset_no: str, model_dir: str, bucket_name: str):
    """evaluate re"""
    output_dir = 'gs://{}/{}/RE_outputs/{}/{}'.format(bucket_name, model_dir, re_dataset, re_dataset_no)
    data_dir = 'gs://{}/datasets/RE/{}/{}'.format(bucket_name, re_dataset, re_dataset_no)
    try:
        run('python3 biobert/biocodes/re_eval.py --output_path={}/test_results.tsv --answer_path={}/test.tsv'.format(
            output_dir, data_dir
        ))
    except exceptions.UnexpectedExit:
        print('Cannot evaluate RE - {}',format(re_dataset))