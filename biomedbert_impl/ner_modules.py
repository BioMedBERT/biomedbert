#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call, CalledProcessError


def run_ner_bc2gm(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run ner BC2GM"""
    try:
        call(['bash', './bash/run_ner_bc2gm.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run NER BC2GM')


def run_ner_bc4chemd(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run ner BC4CHEMD"""
    try:
        call(['bash', './bash/run_ner_bc4chemd.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run NER BC4CHEMD')

def run_ner_bc5cdr_chem(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run ner BC5CDR_chem"""
    try:
        call(['bash', './bash/run_ner_bc5cdr_chem.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run NER BC5CDR_chem')

def run_ner_bc5cdr_disease(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run ner BC5CDR_disease"""
    try:
        call(['bash', './bash/run_ner_bc5cdr_disease.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run NER BC5CDR_disease')

def run_ner_jnlpba(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run ner JNLPBA"""
    try:
        call(['bash', './bash/run_ner_jnlpba.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run NER JNLPBA')

def run_ner_ncbi_disease(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run ner NCBI-disease"""
    try:
        call(['bash', './bash/run_ner_ncbi_disease.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run NER NCBI-disease')

def run_ner_linnaeus(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run ner linnaeus"""
    try:
        call(['bash', './bash/run_ner_linnaeus.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run NER linnaeus')

def run_ner_s800(biomedbert_gcs_dir: str, biomedbert_model_type: str):
    """run ner s800"""
    try:
        call(['bash', './bash/run_ner_s800.sh', biomedbert_gcs_dir, biomedbert_model_type])
    except CalledProcessError:
        print('Cannot run NER s800')

