#!/usr/bin/env python
# -*- coding: utf-8 -*-

from invoke import run, exceptions


def fine_tune_ner(ner_dataset: str, model_dir: str, init_checkpoint: str, vocab_file: str,
                  tpu_name: str, tpu_zone: str, gcp_project: str):
    """fine tune ner"""
    use_tpu = True

    if tpu_name is None:
        tpu_name = 'false'
        use_tpu = False

    try:
        # TODO: parameterize ner dataset on gcs
        run('python3 biobert/run_ner.py  --vocab_file={}/{}   '
            '--bert_config_file={}/bert_config.json   '
            '--init_checkpoint={}/{}   --do_train=true --do_eval=true  '
            '--num_train_epochs=10.0   --data_dir=gs://ekaba-assets/datasets/NER/{}   '
            '--output_dir={}/NER_outputs/{}  --num_tpu_cores=128   --use_tpu={}   '
            '--tpu_name={}   --tpu_zone={}  --gcp_project={}'.format(
            model_dir, vocab_file, model_dir, model_dir, init_checkpoint,
            ner_dataset, model_dir, ner_dataset, use_tpu, tpu_name, tpu_zone, gcp_project))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune NER - {}'.format(ner_dataset))
