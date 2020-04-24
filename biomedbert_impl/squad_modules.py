#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call, CalledProcessError


def run_squad_v11(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run squad v11"""
    try:
        call(['bash', './bash/run_squad_v1_1.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run SQuAD v1.1')


def run_squad_v2(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run squad v2"""
    try:
        call(['bash', './bash/run_squad_v2.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run SQuAD v2')