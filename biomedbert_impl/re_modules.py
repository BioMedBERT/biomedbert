#!/usr/bin/env python
# -*- coding: utf-8 -*-

from invoke import run, exceptions


def fine_tune_re(re_dataset: str, re_dataset_no: str, model_dir: str, init_checkpoint: str,
                 vocab_file: str, tpu_name: str, tpu_zone: str, gcp_project: str):
    """fine tune re"""
    use_tpu = True

    if tpu_name is None:
        tpu_name = 'false'
        use_tpu = False

    try:
        # TODO: parameterize re dataset on gcs
        run('python3 biobert/run_re.py  --task_name={}  --vocab_file={}/{}   '
            '--bert_config_file={}/bert_config.json   '
            '--init_checkpoint={}/{}   --do_train=true --do_eval=true  --do_predict=true  '
            '--max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   '
            '--do_lower_case=false   --num_train_epochs=3.0   --data_dir=gs://ekaba-assets/datasets/RE/{}/{}   '
            '--output_dir={}/RE_outputs/{}/{}  --num_tpu_cores=128   --use_tpu={}   '
            '--tpu_name={}   --tpu_zone={}  --gcp_project={}'.format(
            re_dataset, model_dir, vocab_file, model_dir, model_dir, init_checkpoint,
            re_dataset, re_dataset_no, model_dir, re_dataset, re_dataset_no,
            use_tpu, tpu_name, tpu_zone, gcp_project))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune RE - {}'.format(re_dataset))
