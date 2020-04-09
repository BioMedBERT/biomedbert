#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sentencepiece as spm
import tensorflow as tf
from invoke import run

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


def generate_pre_trained_data(pretraining_dir: str, voc_fname: str, number_of_shards: int, data_path: str):
    """generating pre-trained data"""
    max_seq_length = 128
    masked_lm_prob = 0.15
    max_predictions = 20
    do_lower_case = True
    processes = 2
    pretraining_dir = pretraining_dir  # "pretraining_data"

    # shard dataset
    _shard_dataset(number_of_shards, data_path)

    xargs_cmd = ("ls ./shards/ | "
                 "xargs -n 1 -P {} -I{} "
                 "python3 ../bert/create_pretraining_data.py "
                 "--input_file=./shards/{} "
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

    tf.io.gfile.MkDir(pretraining_dir)
    run('${}'.format(xargs_cmd))


def _shard_dataset(number_of_shards: int, data_path: str):
    """sharding the dataset"""
    run('mkdir ./ shards')
    run('split -a {} -l 5560 -d {} ./shards/shard_'.format(number_of_shards, data_path))
    run('ls ./shards/')


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
    with open(voc_fname, "w") as fo:
        for token in bert_vocab:
            fo.write(token + "\n")
