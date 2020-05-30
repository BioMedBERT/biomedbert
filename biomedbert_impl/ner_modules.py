#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
import tensorflow as tf
from invoke import run, exceptions

log = logging.getLogger('biomedbert')
log.setLevel(logging.INFO)


def fine_tune_ner(ner_dataset: str, model_dir: str, model_type: str, bucket_name: str,
                  tpu_name: str, tpu_zone: str, gcp_project: str, tpu_cores: str):
    """fine tune ner"""
    use_tpu = True
    config = 'large_bert_config.json'

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

    init_checkpoint = tf.train.latest_checkpoint('gs://{}/{}'.format(bucket_name, model_dir))
    vocab_file = 'gs://{}/{}/vocab.txt'.format(bucket_name, model_dir)
    bert_config_file = 'gs://{}/{}/{}'.format(bucket_name, model_dir, config)
    output_dir = 'gs://{}/{}/NER_outputs/{}'.format(bucket_name, model_dir, ner_dataset)
    # output_dir = './NER_outputs/{}'.format(ner_dataset)
    data_dir = 'gs://{}/datasets/NER/{}'.format(bucket_name, ner_dataset)

    try:
        run('python3 biobert/run_ner.py  --vocab_file={}   '
            '--bert_config_file={}    --init_checkpoint={}   --do_train=true --do_eval=true  '
            '--num_train_epochs=10.0   --data_dir={}   '
            '--output_dir={}  --num_tpu_cores=128   --use_tpu={}   '
            '--tpu_name={}   --tpu_zone={}  --gcp_project={},  --num_tpu_cores={}'.format(
            vocab_file, bert_config_file, init_checkpoint, data_dir,
            output_dir, use_tpu, tpu_name, tpu_zone, gcp_project, num_tpu_cores))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune NER - {}'.format(ner_dataset))


def token_level_evaluation(ner_dataset: str, model_dir: str, model_type: str, bucket_name: str,
                           tpu_name: str, tpu_zone: str, gcp_project: str, tpu_cores: str):
    """token-level evaluation ner"""
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
    # output_dir = 'gs://{}/{}/NER_outputs/{}'.format(bucket_name, model_dir, ner_dataset)
    output_dir = './NER_outputs/{}'.format(ner_dataset)
    data_dir = 'gs://{}/datasets/NER/{}'.format(bucket_name, ner_dataset)

    try:
        run('python3 biobert/run_ner.py  --vocab_file={}   '
            '--bert_config_file={}    --init_checkpoint={}   --do_train=false --do_predict=true  '
            '--num_train_epochs=10.0   --data_dir={}   '
            '--output_dir={}  --num_tpu_cores=128   --use_tpu={}   '
            '--tpu_name={}   --tpu_zone={}  --gcp_project={},  --num_tpu_cores={}'.format(
            vocab_file, bert_config_file, init_checkpoint, data_dir,
            output_dir, use_tpu, tpu_name, tpu_zone, gcp_project, num_tpu_cores))
    except exceptions.UnexpectedExit:
        print('Cannot evaluate NER - {}'.format(ner_dataset))
