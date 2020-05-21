#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""biomedbert

Usage:
  biomedbert gcp project set <project-id> <project-zone>
  biomedbert gcp vm start <vm-instance>
  biomedbert gcp vm stop <vm-instance>
  biomedbert gcp vm notebook <vm-instance>
  biomedbert gcp vm connect <vm-instance>
  biomedbert gcp vm create compute <vm-instance> [<project-zone>]
  biomedbert gcp vm create tpu <vm-instance> [<project-zone>]
  biomedbert gcp vm delete tpu <vm-instance> [<project-zone>]
  biomedbert code train vocab <data_path> <prefix>
  biomedbert code train model (base|large) <model_dir> <pretraining_dir> <bucket_name>
  biomedbert code extract embeddings <input_txt> <voc_fname> <config_fname> <init_checkpoint>
  biomedbert code shard data <number_of_shards> <shard_path> <prc_data_path>
  biomedbert code make pretrain data <pre_trained_dir> <voc_filename> <shard_path>
  biomedbert glue finetune <dataset> <model_dir> <checkpoint_name> <vocab_file> [<tpu_name>]
  biomedbert glue predict <dataset> <model_dir> <trained_classifier> <vocab_file> [<tpu_name>]
  biomedbert glue download dataset
  biomedbert squad v1 <biomedbert_gcs_path> <biomedbert_model_type>
  biomedbert squad v2 <biomedbert_gcs_path> <biomedbert_model_type>

  biomedbert -h | --help
  biomedbert --version

Options:
  -h, --help    Show this screen.
  --version     Show version.
"""

from __future__ import unicode_literals, print_function

import configparser
from docopt import docopt
from biomedbert_impl.modules import train_vocabulary, generate_pre_trained_data, shard_dataset, \
    extract_embeddings, train_biomedbert_base
from biomedbert_impl.gcp_helpers import set_gcp_project, start_vm, stop_vm, \
    launch_notebook, connect_vm, create_compute_vm, create_tpu_vm, delete_tpu_vm
from biomedbert_impl.glue_modules import fine_tune_classification_glue, download_glue_data, \
    predict_classification_glue
from biomedbert_impl.squad_modules import run_squad_v11, run_squad_v2
from biomedbert_impl.bioasq_modules import run_bioasq_4b, run_bioasq_5b, run_bioasq_6b
from biomedbert_impl.ner_modules import run_ner_bc2gm, run_ner_bc4chemd, run_ner_bc5cdr_chem, \
    run_ner_bc5cdr_disease, run_ner_jnlpba, run_ner_ncbi_disease, run_ner_linnaeus, run_ner_s800
from biomedbert_impl.re_modules import run_re_gad_1, run_re_gad_2, run_re_gad_3, run_re_gad_4, \
    run_re_gad_5, run_re_gad_6, run_re_gad_7, run_re_gad_8, run_re_gad_9, run_re_gad_10, \
    run_re_euadr_1, run_re_euadr_2, run_re_euadr_3, run_re_euadr_4, run_re_euadr_5, \
    run_re_euadr_6, run_re_euadr_7, run_re_euadr_8, run_re_euadr_9, run_re_euadr_10

__version__ = "0.1.0"
__author__ = "AI vs COVID-19 Team"
__license__ = "MIT"

# Configurations
config = configparser.ConfigParser()


def re_commands(args: dict):
    """Command to run Relation Extraction benchmark datasets"""

    # run gad 1
    if args['re'] and args['gad_1']:
        run_re_gad_1(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 2
    if args['re'] and args['gad_2']:
        run_re_gad_2(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 3
    if args['re'] and args['gad_3']:
        run_re_gad_3(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 4
    if args['re'] and args['gad_4']:
        run_re_gad_4(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 5
    if args['re'] and args['gad_5']:
        run_re_gad_5(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 6
    if args['re'] and args['gad_6']:
        run_re_gad_6(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 7
    if args['re'] and args['gad_7']:
        run_re_gad_7(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 8
    if args['re'] and args['gad_8']:
        run_re_gad_8(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 9
    if args['re'] and args['gad_9']:
        run_re_gad_9(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run gad 10
    if args['re'] and args['gad_10']:
        run_re_gad_10(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 1
    if args['re'] and args['euadr_1']:
        run_re_euadr_1(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 2
    if args['re'] and args['euadr_2']:
        run_re_euadr_2(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 3
    if args['re'] and args['euadr_3']:
        run_re_euadr_3(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 4
    if args['re'] and args['euadr_4']:
        run_re_euadr_4(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 5
    if args['re'] and args['euadr_5']:
        run_re_euadr_5(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 6
    if args['re'] and args['euadr_6']:
        run_re_euadr_6(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 7
    if args['re'] and args['euadr_7']:
        run_re_euadr_7(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 8
    if args['re'] and args['euadr_8']:
        run_re_euadr_8(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 9
    if args['re'] and args['euadr_9']:
        run_re_euadr_9(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run euadr 10
    if args['re'] and args['euadr_10']:
        run_re_euadr_10(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])


def ner_commands(args: dict):
    """Command to run Named Entity recognition benchmark datasets"""

    # run ner bc2gm
    if args['ner'] and args['bc2gm']:
        run_ner_bc2gm(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run ner bc4chemd
    if args['ner'] and args['bc4chemd']:
        run_ner_bc4chemd(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run ner bc5cdr_chem
    if args['ner'] and args['bc5cdr_chem']:
        run_ner_bc5cdr_chem(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run ner bc5cdr_disease
    if args['ner'] and args['bc5cdr_disease']:
        run_ner_bc5cdr_disease(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run ner jnlpba
    if args['ner'] and args['jnlpba']:
        run_ner_jnlpba(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run ner ncbi_disease
    if args['ner'] and args['ncbi_disease']:
        run_ner_ncbi_disease(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run ner linnaeus
    if args['ner'] and args['linnaeus']:
        run_ner_linnaeus(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run ner s800
    if args['ner'] and args['s800']:
        run_ner_s800(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])


def bioasq_commands(args: dict):
    """Command to run BIOASQ question answering benchmark datasets"""

    # run bioasq 4b
    if args['bioasq'] and args['4b']:
        run_bioasq_4b(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run bioasq 5b
    if args['bioasq'] and args['5b']:
        run_bioasq_5b(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run bioasq 6b
    if args['bioasq'] and args['6b']:
        run_bioasq_6b(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])


def squad_commands(args: dict):
    """Command to run SQuAD question answering benchmark dataset"""

    # run squad v1.1
    if args['squad'] and args['v1']:
        run_squad_v11(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])

    # run squad v2.0
    if args['squad'] and args['v2']:
        run_squad_v2(args['<biomedbert_gcs_path>'], args['<biomedbert_model_type>'])


def glue_commands(args: dict):
    """Command to run GLUE classification"""

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # download glue dataset
    if args['glue'] and args['download'] and args['dataset']:
        download_glue_data()

    # predict glue
    if args['glue'] and args['predict']:
        predict_classification_glue(args['<dataset>'], args['<model_dir>'],
                                    args['<trained_classifier>'], args['<vocab_file>'],
                                    args['<tpu_name>'], zone, project_id)

    # finetune glue
    if args['glue'] and args['finetune']:
        fine_tune_classification_glue(args['<dataset>'], args['<model_dir>'],
                                      args['<checkpoint_name>'], args['<vocab_file>'],
                                      args['<tpu_name>'], zone, project_id)


def code_commands(args: dict):
    """Command to train BioMedBert model"""

    # train biomedbert-base
    if args['code'] and args['train'] and args['model']:
        if args['base']:
            train_biomedbert_base(args['<model_dir>'], args['<pretraining_dir>'], args['<bucket_name>'])

    # extract contextual embeddings
    if args['code'] and args['extract'] and args['embeddings']:
        extract_embeddings(args['<input_txt>'], args['<voc_fname>'],
                           args['<config_fname>'], args['<init_checkpoint>'])

    # train vocab
    if args['code'] and args['train'] and args['vocab']:
        train_vocabulary(args['<data_path>'], args['<prefix>'])

    # generate pre-trained dataset
    if args['code'] and args['make'] and args['pretrain'] and args['data']:
        generate_pre_trained_data(args['<pre_trained_dir>'], args['<voc_filename>'],
                                  args['<shard_path>'])

    # shard the dataset
    if args['code'] and args['shard'] and args['data']:
        shard_dataset(args['<number_of_shards>'], args['<shard_path>'], args['<prc_data_path>'])


def gcp_commands(args: dict):
    """GCP commands for biomedber CLI."""

    # setup GCP project
    if args['gcp'] and args['project'] and args['set']:
        # set values
        config.add_section('PROJECT')
        config.set('PROJECT', 'name', args['<project-id>'])
        config.set('PROJECT', 'zone', args['<project-zone>'])

        # write to config file
        with open('./config/gcp_config.ini', 'w') as configfile:
            config.write(configfile)

        # call set project
        set_gcp_project(args['<project-id>'], args['<project-zone>'])

    # create compute VM
    if args['gcp'] and args['vm'] and args['create'] and args['compute']:
        # create vm
        if args['<project-zone>'] is None:
            # read configurations
            config.read('config/gcp_config.ini')
            zone = config['PROJECT']['zone']
            create_compute_vm(args['<vm-instance>'], zone)
        else:
            create_compute_vm(args['<vm-instance>'], args['<project-zone>'])

    # create tpu
    if args['gcp'] and args['vm'] and args['create'] and args['tpu']:
        if args['<project-zone>'] is None:
            # read configurations
            config.read('config/gcp_config.ini')
            zone = config['PROJECT']['zone']
            create_tpu_vm(args['<vm-instance>'], zone)
        else:
            create_tpu_vm(args['<vm-instance>'], args['<project-zone>'])

    # delete tpu
    if args['gcp'] and args['vm'] and args['delete'] and args['tpu']:
        # read configurations
        config.read('config/gcp_config.ini')
        project_id = config['PROJECT']['name']
        if args['<project-zone>'] is None:
            zone = config['PROJECT']['zone']
            delete_tpu_vm(args['<vm-instance>'], project_id, zone)
        else:
            delete_tpu_vm(args['<vm-instance>'], project_id, args['<project-zone>'])

    # start VM
    if args['gcp'] and args['vm'] and args['start']:
        # start vm
        start_vm(args['<vm-instance>'])

    # stop VM
    if args['gcp'] and args['vm'] and args['stop']:
        stop_vm(args['<vm-instance>'])

    # connect VM
    if args['gcp'] and args['vm'] and args['connect']:
        # read configurations
        config.read('config/gcp_config.ini')
        project_id = config['PROJECT']['name']
        zone = config['PROJECT']['zone']

        # ssh to instance
        connect_vm(project_id, zone, args['<vm-instance>'])

    # launch jupyter notebook on VM
    if args['gcp'] and args['vm'] and args['notebook']:
        # read configurations
        config.read('config/gcp_config.ini')
        project_id = config['PROJECT']['name']
        zone = config['PROJECT']['zone']

        # lauch notebook
        launch_notebook(project_id, zone, args['<vm-instance>'])


def main():
    """Main entry point for the biomedbert CLI."""
    args = docopt(__doc__, version=__version__,
                  options_first=True)
    # print(args)

    if args['gcp']:
        gcp_commands(args)

    if args['code']:
        code_commands(args)

    if args['glue']:
        glue_commands(args)

    if args['squad']:
        squad_commands(args)


if __name__ == '__main__':
    main()
