#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call, CalledProcessError


def fine_tune_classification_glue(glue_dataset: str, model_dir: str,
                                  init_checkpoint: str, vocab_file: str,
                                  tpu_name: str, tpu_zone: str, gcp_project: str):
    """finetune classification tasks from the GLUE dataset"""
    if tpu_name is None:
        tpu_name = 'false'

    try:
        call(['bash', './bash/fine_tune_classification_glue.sh', glue_dataset,
              model_dir, init_checkpoint, vocab_file, tpu_name, tpu_zone, gcp_project])
    except CalledProcessError:
        print('Cannot finetune {} model'.format(glue_dataset))


def predict_classification_glue(glue_dataset: str, model_dir: str,
                                init_checkpoint: str, vocab_file: str,
                                tpu_name: str, tpu_zone: str, gcp_project: str):
    """predict classification tasks from the GLUE dataset"""
    if tpu_name is None:
        tpu_name = 'false'

    try:
        call(['bash', './bash/predict_classification_glue.sh', glue_dataset,
              model_dir, init_checkpoint, vocab_file, tpu_name, tpu_zone, gcp_project])
    except CalledProcessError:
        print('Cannot predict {} model'.format(glue_dataset))


def download_glue_data():
    """download GLUE dataset"""
    try:
        call(['python3', './biomedbert_impl/download_glue_data.py',
              '--data_dir', 'glue_data', '--tasks', 'all'])
    except CalledProcessError:
        print('Cannot download GLUE dataset')
