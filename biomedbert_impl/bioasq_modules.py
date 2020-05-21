#!/usr/bin/env python
# -*- coding: utf-8 -*-

from invoke import run, exceptions


def fine_tune_bioasq(train_file: str, predict_file: str, model_dir: str, init_checkpoint: str,
                     vocab_file: str, tpu_name: str, tpu_zone: str, gcp_project: str):
    """fine tune bioasq"""
    use_tpu = True

    if tpu_name is None:
        tpu_name = 'false'
        use_tpu = False

    p_model_dir = '/'.join(model_dir.split('/')[:-1])

    try:
        # TODO: parameterize bioasq dataset on gcs
        run('python3 biobert/run_qa.py  --vocab_file={}/{}   '
            '--bert_config_file={}/bert_config.json   '
            '--init_checkpoint={}/{}   --do_train=true --do_predict=true  '
            '--max_seq_length=384   --train_batch_size=128   --learning_rate=5e-6   '
            '--doc_stride=128   --num_train_epochs=5.0   --do_lower_case=False   '
            '--train_file=gs://ekaba-assets/datasets/QA/BioASQ/{}   '
            '--predict_file=gs://ekaba-assets/datasets/QA/BioASQ/{}   '
            '--output_dir={}/BioASQ_outputs/{}/  --num_tpu_cores=128   --use_tpu={}   '
            '--tpu_name={}   --tpu_zone={}  --gcp_project={}'.format(
            p_model_dir, vocab_file, p_model_dir, model_dir, init_checkpoint,
            train_file, predict_file, p_model_dir, train_file.split('.')[0],
            use_tpu, tpu_name, tpu_zone, gcp_project))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune BioASQ - {}'.format(train_file))
