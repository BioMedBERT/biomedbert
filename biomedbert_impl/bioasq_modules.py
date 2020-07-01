#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import tensorflow as tf
from invoke import run, exceptions

log = logging.getLogger('biomedbert')
log.setLevel(logging.INFO)


def fine_tune_bioasq(model_type: str, bucket_name: str, train_file: str, predict_file: str, model_dir: str,
                     tpu_name: str, tpu_zone: str, gcp_project: str, tpu_cores: int, squad_folder: str):
    """fine tune bioasq"""
    use_tpu = True
    config = 'large_bert_config.json'
    max_seq_length = 128  # 384

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

    init_checkpoint = tf.train.latest_checkpoint('gs://{}/{}/{}'.format(bucket_name, model_dir, squad_folder))
    vocab_file = 'gs://{}/{}/vocab.txt'.format(bucket_name, model_dir)
    bert_config_file = 'gs://{}/{}/{}'.format(bucket_name, model_dir, config)
    output_dir = 'gs://{}/{}/BioASQ_outputs/{}/{}'.format(bucket_name, model_dir, squad_folder, predict_file.split('.')[0])
    train_file = 'gs://{}/datasets/QA/BioASQ/{}'.format(bucket_name, train_file)
    predict_file = 'gs://{}/datasets/QA/BioASQ/{}'.format(bucket_name, predict_file)

    try:
        run('python3 biobert/run_qa.py  --vocab_file={}   '
            '--bert_config_file={}   --predict_batch_size=128   '
            '--init_checkpoint={}   --do_train=true --do_predict=true  '
            '--max_seq_length={}   --train_batch_size=32   --learning_rate=5e-6   '
            '--doc_stride=128   --num_train_epochs=5.0   --do_lower_case=False   '
            '--train_file={}  --predict_file={}   '
            '--output_dir={}/  --num_tpu_cores={}   --use_tpu={}   '
            '--tpu_name={}   --tpu_zone={}  --gcp_project={}'.format(
            vocab_file, bert_config_file, init_checkpoint, max_seq_length,
            train_file, predict_file, output_dir, num_tpu_cores,
            use_tpu, tpu_name, tpu_zone, gcp_project))
    except exceptions.UnexpectedExit:
        print('Cannot fine tune BioASQ - {}'.format(train_file))


def evaluate_bioasq(bucket_name: str, model_dir: str, predict_file: str, eval_file: str, squad_folder: str):
    """evaluate bioasq"""

    # convert results to BioASQ JSON format
    try:
        output_dir = 'BioASQ_outputs/{}/{}'.format(squad_folder, predict_file.split('.')[0])

        if not os.path.exists(output_dir):
            run('mkdir -p {}'.format(output_dir))

        run('gsutil cp gs://{}/{}/{}/nbest_predictions.json {}'.format(
            bucket_name, model_dir, output_dir, output_dir))
        run('python3 biobert/biocodes/transform_nbset2bioasqform.py   '
            '--nbest_path={}/nbest_predictions.json '
            '--output_path={}'.format(output_dir, output_dir))
    except exceptions.UnexpectedExit:
        print('Cannot convert results to BioASQ JSON format')
        sys.exit(1)

    # run BioAsq evaluation script
    try:
        if not os.path.exists('Evaluation-Measures'):
            run('git clone https://github.com/BioASQ/Evaluation-Measures.git')
        run('gsutil cp gs://ekaba-assets/datasets/QA/BioASQ/{} {}'.format(
            eval_file, output_dir))
        run('cd Evaluation-Measures')
        run('java -Xmx10G '
            '-cp $CLASSPATH:Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar '
            'evaluation.EvaluatorTask1b -phaseB -e 5 {}/{} '
            '{}/BioASQform_BioASQ-answer.json'.format(output_dir, eval_file, output_dir))
    except exceptions.UnexpectedExit:
        print('Cannot evaluate BioASQ')
        sys.exit(1)
