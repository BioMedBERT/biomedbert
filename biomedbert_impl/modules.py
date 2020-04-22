#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sentencepiece as spm
import tensorflow as tf
from subprocess import call, CalledProcessError

# global parameters
voc_size = 32000

def train_vocabulary(data_path: str, prefix: str):
    """Method to train the vocabulary using sentencepiece"""

    if data_path is None or prefix is None:
        return 'Assign name for data and prefix'

    subsample_size = 8000000
    num_placeholders = 256

    prc_data_fpath = data_path  # "processed_ncbi_comm_use_BODY.txt"
    model_prefix = prefix  # "biomedbert"

    spm_command = ('--input={} --model_prefix={} '
                   '--vocab_size={} --input_sentence_size={} '
                   '--shuffle_input_sentence=true '
                   '--bos_id=-1 --eos_id=-1').format(
        prc_data_fpath, model_prefix,
        voc_size - num_placeholders, subsample_size)

    spm.SentencePieceTrainer.train(spm_command)

    # write processed vocab to file.
    _write_vocabulary_to_file(model_prefix, model_prefix)


def extract_embeddings(input_txt: str, voc_fname: str, config_fname: str, init_ckt: str):
    """extract contextual embeddings"""
    input_txt = input_txt  # 'input_fra.txt'
    output_file = "output" + input_txt.split('.')[0] + ".jsonl"  # 'output_fra.jsonl'

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


def generate_pre_trained_data(pretraining_dir: str, voc_fname: str, shard_path: str):
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


def shard_dataset(number_of_shards: int, shard_path: str, prc_data_path: str):
    """sharding the dataset"""
    if not os.path.exists('./shards/{}'.format(shard_path)):
        call(['mkdir', '-p', './shards/{}'.format(shard_path)])

    try:
        call(['split', '-a', number_of_shards, '-l', '5560', '-d', prc_data_path,
              './shards/{}/shard_'.format(shard_path)])
        call(['ls', './shards/{}'.format(shard_path)])
    except CalledProcessError:
        print('Error in sharding')


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
