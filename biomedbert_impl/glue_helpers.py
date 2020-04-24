#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call, CalledProcessError


def fine_tune_classification_glue(glue_dataset: str, biomedbert_gcs_dir: str):
    """finetune classification tasks from the GLUE dataset"""
    try:
        call(['bash', './bash/fine_tune_classification_glue.sh', biomedbert_gcs_dir, glue_dataset])
    except CalledProcessError:
        print('Cannot finetune {} model'.format(glue_dataset))


def predict_classification_glue(glue_dataset: str, biomedbert_gcs_dir: str):
    """predict classification tasks from the GLUE dataset"""
    try:
        call(['bash', './bash/predict_classification_glue.sh', biomedbert_gcs_dir, glue_dataset])
    except CalledProcessError:
        print('Cannot predict {} model'.format(glue_dataset))


def download_glue_data():
    """download GLUE dataset"""
    try:
        call(['python3', './biomedbert_impl/download_glue_data.py',
              '--data_dir', 'glue_data', '--tasks', 'all'])
    except CalledProcessError:
        print('Cannot download GLUE dataset')
