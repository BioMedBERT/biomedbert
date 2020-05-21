#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
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
            '--bert_config_file={}/bert_config.json   --predict_batch_size=128'
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


def evaluate_bioasq(model_dir: str, train_file: str):
    """evaluate bioasq"""

    output_dir = '{}/BioASQ_outputs/{}/'.format(model_dir, train_file.split('.')[0])
    # convert results to BioASQ JSON format
    try:
        run('python3 biobert/biocodes/transform_nbset2bioasqform.py   '
            '--nbest_path={}/nbest_predictions.json --output_path={}'.format(output_dir, output_dir))
    except exceptions.UnexpectedExit:
        print('Cannot convert results to BioASQ JSON format')
        sys.exit(1)

    # run BioAsq evaluation script
    try:
        run('git clone https://github.com/BioASQ/Evaluation-Measures.git')
        run('cd Evaluation-Measures')
        run('java -Xmx10G '
            '-cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b '
            '-phaseB -e 5 gs://ekaba-assets/datasets/QA/BioASQ/4B1_golden.json '
            '{}/BioASQform_BioASQ-answer.json'.format(output_dir))
    except exceptions.UnexpectedExit:
        print('Cannot evaluate BioASQ')
        sys.exit(1)