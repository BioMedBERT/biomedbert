#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
import tensorflow as tf
from subprocess import call, CalledProcessError
from invoke import run, exceptions

log = logging.getLogger('biomedbert')
log.setLevel(logging.INFO)


def fine_tune_classification_glue(glue_dataset: str, model_type: str, bucket_name: str, model_dir: str, tpu_name: str,
                                  tpu_zone: str, gcp_project: str, tpu_cores: str):
    """finetune classification tasks from the GLUE dataset"""
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
    output_dir = 'gs://{}/{}/GLUE_outputs/{}'.format(bucket_name, model_dir, glue_dataset)
    data_dir = 'gs://{}/glue_data/{}'.format(bucket_name, glue_dataset)

    try:
        run('python3 bert/run_classifier.py  --task_name={}  --do_train=true  --do_eval=true   '
            '--data_dir={}   --vocab_file={}   --bert_config_file={}   '
            '--init_checkpoint={}   --max_seq_length=128   --train_batch_size=32   '
            '--learning_rate=2e-5   --num_train_epochs=4.0   --output_dir={}   '
            '--num_tpu_cores={}   --use_tpu={}   --tpu_name={}   --tpu_zone={}   '
            '--gcp_project={}'.format(
            glue_dataset, data_dir, vocab_file, bert_config_file, init_checkpoint,
            output_dir, num_tpu_cores, use_tpu, tpu_name, tpu_zone, gcp_project))
    except exceptions.UnexpectedExit:
        print('Bad command')


def predict_classification_glue(glue_dataset: str, model_type: str, model_dir: str, bucket_name: str, tpu_cores: str,
                                tpu_name: str, tpu_zone: str, gcp_project: str):
    """predict classification tasks from the GLUE dataset"""
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

    vocab_file = 'gs://{}/{}/vocab.txt'.format(bucket_name, model_dir)
    bert_config_file = 'gs://{}/{}/{}'.format(bucket_name, model_dir, config)
    trained_classifier = tf.train.latest_checkpoint('gs://{}/{}/GLUE_outputs/{}'.format(
        bucket_name, model_dir, glue_dataset))
    output_dir = 'gs://{}/{}/GLUE_outputs/{}'.format(bucket_name, model_dir, glue_dataset)
    data_dir = 'gs://{}/glue_data/{}'.format(bucket_name, glue_dataset)

    try:
        run('python3 bert/run_classifier.py  --task_name={}  --do_predict=true   '
            '--data_dir={}   --vocab_file={}  --bert_config_file={}   '
            '--init_checkpoint={}   --max_seq_length=128   --train_batch_size=128   '
            '--output_dir={}  --num_tpu_cores={}   --use_tpu={}   --tpu_name={}   '
            '--tpu_zone={}   --gcp_project={}'.format(
            glue_dataset, data_dir, vocab_file, bert_config_file, trained_classifier, output_dir,
            num_tpu_cores, use_tpu, tpu_name, tpu_zone, gcp_project))
    except exceptions.UnexpectedExit:
        print('Bad command')


def download_glue_data():
    """download GLUE dataset"""
    try:
        call(['python3', './biomedbert_impl/download_glue_data.py',
              '--data_dir', 'glue_data', '--tasks', 'all'])
    except CalledProcessError:
        print('Cannot download GLUE dataset')
