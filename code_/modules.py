#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sentencepiece as spm


def train_vocabulary(data: str, prefix: str):
    """Method to train the vocabulary using sentencepiece"""

    if data is None or prefix is None:
        return 'Assign name for data and prefix'

    prc_data_fpath = data  # "processed_ncbi_comm_use_BODY.txt"
    model_prefix = prefix  # "biomedbert"
    voc_size = 32000
    subsample_size = 8000000
    num_placeholders = 256

    spm_command = ('--input={} --model_prefix={} '
                   '--vocab_size={} --input_sentence_size={} '
                   '--shuffle_input_sentence=true '
                   '--bos_id=-1 --eos_id=-1').format(
        prc_data_fpath, model_prefix,
        voc_size - num_placeholders, subsample_size)

    spm.SentencePieceTrainer.train(spm_command)
