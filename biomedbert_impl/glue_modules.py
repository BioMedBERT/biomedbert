#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call, CalledProcessError
from invoke import run, exceptions


def fine_tune_classification_glue(glue_dataset: str, model_dir: str,
                                  init_checkpoint: str, vocab_file: str,
                                  tpu_name: str, tpu_zone: str, gcp_project: str):
    """finetune classification tasks from the GLUE dataset"""
    use_tpu = True
    if tpu_name is None:
        tpu_name = 'false'
        use_tpu = False

    try:
        run('python3 bert/run_classifier.py  --task_name={}  --do_train=true  --do_eval=true   '
            '--data_dir=glue_data/{}   --vocab_file={}/{}   '
            '--bert_config_file={}/bert_config.json   '
            '--init_checkpoint={}/{}   --max_seq_length=128   --train_batch_size=32   '
            '--learning_rate=2e-5   --num_train_epochs=3.0   --output_dir={}/{}_output   '
            '--num_tpu_cores=128   --use_tpu={}   --tpu_name={}   --tpu_zone={}   '
            '--gcp_project={}'.format(
            glue_dataset, glue_dataset, model_dir, vocab_file, model_dir, model_dir, init_checkpoint,
            model_dir, glue_dataset, use_tpu, tpu_name, tpu_zone, gcp_project))
    except exceptions.UnexpectedExit:
        print('Bad command')


def predict_classification_glue(glue_dataset: str, model_dir: str,
                                trained_classifier: str, vocab_file: str,
                                tpu_name: str, tpu_zone: str, gcp_project: str):
    """predict classification tasks from the GLUE dataset"""
    use_tpu = True
    if tpu_name is None:
        tpu_name = 'false'
        use_tpu = False

    try:
        run('python3 bert/run_classifier.py  --task_name={}  --do_predict=true   '
            '--data_dir=glue_data/{}   --vocab_file={}/{}   '
            '--bert_config_file={}/bert_config.json   '
            '--init_checkpoint={}/{}_output/{}   --max_seq_length=128   --train_batch_size=128   '
            '--output_dir={}/{}_output  --num_tpu_cores=128   --use_tpu={}   --tpu_name={}   '
            '--tpu_zone={}   --gcp_project={}'.format(
            glue_dataset, glue_dataset, model_dir, vocab_file, model_dir, model_dir, glue_dataset,
            trained_classifier, model_dir, glue_dataset, use_tpu, tpu_name, tpu_zone, gcp_project))
    except exceptions.UnexpectedExit:
        print('Bad command')


def download_glue_data():
    """download GLUE dataset"""
    try:
        call(['python3', './biomedbert_impl/download_glue_data.py',
              '--data_dir', 'glue_data', '--tasks', 'all'])
    except CalledProcessError:
        print('Cannot download GLUE dataset')
