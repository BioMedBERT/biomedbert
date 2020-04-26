#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call, CalledProcessError


def run_bioasq_4b(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run bioasq 4b"""
    try:
        call(['bash', './bash/run_bioasq_4b.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run BIOASQ 4b')


def run_bioasq_5b(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run bioasq 5b"""
    try:
        call(['bash', './bash/run_bioasq_5b.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run BIOASQ 5b')

def run_bioasq_6b(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run bioasq 6b"""
    try:
        call(['bash', './bash/run_bioasq_6b.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run BIOASQ 6b')