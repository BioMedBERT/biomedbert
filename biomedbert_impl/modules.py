#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import sentencepiece as spm
import tensorflow as tf
from invoke import run, exceptions
from subprocess import call, CalledProcessError
from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

# global parameters
voc_size = 32000 # former -> 28996

log = logging.getLogger('biomedbert')
log.setLevel(logging.INFO)


def train_biomedbert(model_type: str, model_dir: str, pretraining_dir: str, bucket_name: str, tpu_name: str,
                     project_id: str, zone: str, tpu_cores: int):
    """Method to train BioMedBERT"""

    tf.io.gfile.mkdir(model_dir)
    config = ''

    if model_type == 'base':
        # bert base
        try:
            run('cp config/base_bert_config.json {}'.format(model_dir))
            config = 'base_bert_config.json'
        except exceptions.UnexpectedExit:
            log.info('Bert Base moved to model directory')
    elif model_type == 'large':
        # bert large
        try:
            run('cp config/large_bert_config.json {}'.format(model_dir))
            config = 'large_bert_config.json'
        except exceptions.UnexpectedExit:
            log.info('Bert Large moved to model directory')
    else:
        log.info('Error in BioMedBert config')
        sys.exit(1)

    # upload model directory to GCS
    if os.path.exists(model_dir):
        try:
            run('gsutil -m cp -r {} gs://{}'.format(model_dir, bucket_name))
        except exceptions.UnexpectedExit:
            print('Could not upload {} to GCS'.format(model_dir))

    init_checkpoint = tf.train.latest_checkpoint('gs://{}/{}'.format(bucket_name, model_dir))
    input_file = 'gs://{}/{}/shard_*'.format(bucket_name, pretraining_dir)
    output_dir = 'gs://{}/{}'.format(bucket_name, model_dir)
    bert_base_dir = 'gs://{}/{}'.format(bucket_name, model_dir)

    # training procedure config
    train_batch_size = 256
    eval_batch_size = 256
    max_seq_length = 128
    max_predictions = 20
    train_steps = 1000000  # 1M
    num_warmup_steps = 10000
    learning_rate = 1e-5  # 2e-5
    num_tpu_cores = int(tpu_cores)

    # train biomedbert
    if init_checkpoint is None:
        try:
            run(
                'python3 bert/run_pretraining.py   --input_file={}   --output_dir={}   --do_train=True   '
                '--do_eval=True  --bert_config_file={}/{}   --train_batch_size={}   --eval_batch_size={}   '
                '--max_seq_length={}   --max_predictions_per_seq={}   --num_train_steps={}   --num_warmup_steps={}   '
                '--learning_rate={}   --use_tpu=true   --tpu_name={}   --tpu_zone={}   --gcp_project={}   '
                '--num_tpu_cores={}'.format(
                    input_file, output_dir, bert_base_dir, config, train_batch_size,
                    eval_batch_size, max_seq_length, max_predictions, train_steps, num_warmup_steps,
                    learning_rate, tpu_name, zone, project_id, num_tpu_cores
                ))
        except exceptions.UnexpectedExit:
            log.info('Cannot train BioMedBERT')
    else:
        try:
            run('python3 bert/run_pretraining.py   --input_file={}   --output_dir={}   --do_train=True   '
                '--do_eval=True  --bert_config_file={}/{}   --init_checkpoint={}   --train_batch_size={}   '
                '--eval_batch_size={}  --max_seq_length={}   --max_predictions_per_seq={}   --num_train_steps={}   '
                '--num_warmup_steps={}  --learning_rate={}   --use_tpu=true   --tpu_name={}   --tpu_zone={}   '
                '--gcp_project={}  --num_tpu_cores={}'.format(
                input_file, output_dir, bert_base_dir, config, init_checkpoint, train_batch_size,
                eval_batch_size, max_seq_length, max_predictions, train_steps, num_warmup_steps,
                learning_rate, tpu_name, zone, project_id, num_tpu_cores
            ))
        except exceptions.UnexpectedExit:
            log.info('Cannot train BioMedBERT')


def train_vocabulary(data_path: str, prefix: str):
    """Method to train the vocabulary using sentencepiece"""

    # download dataset to VM before training vocabulary
    try:
        run('gsutil -m cp {} .'.format(data_path))
        log.info('Dataset {} downloaded to VM'.format(data_path))
    except exceptions.UnexpectedExit:
        log.info('Could not download dataset from GCS')

    subsample_size = 8000000  # 8M
    num_placeholders = 256

    prc_data_fpath = data_path.split('/')[-1]  # "processed_ncbi_comm_use_BODY.txt"
    model_prefix = prefix  # "biomedbert"

    spm_command = ('--input={} --model_prefix={} '
                   '--vocab_size={} --input_sentence_size={} '
                   '--max_sentence_length=10000'
                   '--shuffle_input_sentence=true '
                   '--bos_id=-1 --eos_id=-1').format(
        prc_data_fpath, model_prefix,
        voc_size - num_placeholders, subsample_size)

    spm.SentencePieceTrainer.train(spm_command)

    # write processed vocab to file.
    _write_vocabulary_to_file(model_prefix, model_prefix)

    # move vocabulary to GCS
    try:
        run('gsutil -m cp -r vocabulary/ gs://ekaba-assets/')
        log.info('Dataset moved to GCS')
    except exceptions.UnexpectedExit:
        log.info('Move vocabulary to GCS')


def extract_embeddings(input_txt: str, voc_fname: str, config_fname: str, init_ckt: str):
    """extract contextual embeddings"""
    input_txt = input_txt  # 'input_fra.txt'
    output_file = "output-" + input_txt.split('.')[0] + ".jsonl"  # 'output_fra.jsonl'

    xargs_cmd = ("python3 bert/extract_features.py "
                 "--input_file={} "
                 "--output_file={} "
                 "--vocab_file={} "
                 "--bert_config_file={} "
                 "--init_checkpoint={} "
                 "--layers=-1,-2,-3,-4 "
                 "--max_seq_length=128 "
                 "--batch_size=8 ")

    xargs_cmd = xargs_cmd.format(input_txt, output_file, voc_fname,
                                 config_fname, init_ckt)

    try:
        call(xargs_cmd, shell=True)
    except CalledProcessError:
        print('Error in running {}'.format(xargs_cmd))


def generate_pre_trained_data(pretraining_dir: str, voc_fname: str, shard_path: str,
                              bucket_name: str = 'ekaba-assets'):
    """generating pre-trained data"""
    max_seq_length = 128
    masked_lm_prob = 0.15
    max_predictions = 20
    do_lower_case = True
    processes = 2
    pretraining_dir = pretraining_dir  # "pretraining_data"

    xargs_cmd = ("ls ./shards/" + shard_path + "/ | "
                                               "xargs -n 1 -P {} -I{} "
                                               "python3 bert/create_pretraining_data.py "
                                               "--input_file=./shards/" + shard_path + "/{} "
                                                                                       "--output_file={}/{}.tfrecord "
                                                                                       "--vocab_file={} "
                                                                                       "--do_lower_case={} "
                                                                                       "--max_predictions_per_seq={} "
                                                                                       "--max_seq_length={} "
                                                                                       "--masked_lm_prob={} "
                                                                                       "--random_seed=34 "
                                                                                       "--dupe_factor=5")

    xargs_cmd = xargs_cmd.format(processes, '{}', '{}', pretraining_dir, '{}',
                                 voc_fname, do_lower_case,
                                 max_predictions, max_seq_length, masked_lm_prob)

    tf.io.gfile.mkdir(pretraining_dir)

    try:
        call(xargs_cmd, shell=True)
    except CalledProcessError:
        print('Error in running {}'.format(xargs_cmd))

    try:
        run('gsutil -m cp -r {} gs://{}'.format(pretraining_dir, bucket_name))
    except exceptions.UnexpectedExit:
        print('Could not upload Pre-training data to GCS')

    # # delete dataset from VM
    # try:
    #     run('rm {}'.format(prc_data_fpath))
    #     log.info('Dataset {} removed from VM'.format(prc_data_fpath))
    # except exceptions.UnexpectedExit:
    #     log.info('Could not delete dataset')


def shard_dataset(number_of_shards: int, shard_path: str, prc_data_path: str):
    """sharding the dataset"""
    if not os.path.exists('./shards/{}'.format(shard_path)):
        call(['mkdir', '-p', './shards/{}'.format(shard_path)])

    try:
        call(['split', '-a', number_of_shards, '-l', '5560', '-d', prc_data_path,
              './shards/{}/shard_'.format(shard_path)])
        call(['ls', './shards/{}'.format(shard_path)])
    except CalledProcessError:
        log.info('Error in sharding')


def _read_sentencepiece_vocab(filepath: str):
    """read sentencepiece vocab"""
    voc = []
    with open(filepath, encoding='utf-8') as fi:
        for line in fi:
            voc.append(line.split("\t")[0])
    # skip the first <unk> token
    voc = voc[1:]
    return voc


def _parse_sentencepiece_token(token: str):
    """parse sentence token"""
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token


def _write_vocabulary_to_file(model_prefix: str, voc_fname: str):
    """write processed vocabulary to file"""
    snt_vocab = _read_sentencepiece_vocab("{}.vocab".format(model_prefix))
    bert_vocab = list(map(_parse_sentencepiece_token, snt_vocab))

    ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    bert_vocab = ctrl_symbols + bert_vocab
    bert_vocab += ["[UNUSED_{}]".format(i) for i in range(voc_size - len(bert_vocab))]

    # write vocabulary to file
    with open(voc_fname + '.txt', "w") as fo:
        for token in bert_vocab:
            fo.write(token + "\n")
