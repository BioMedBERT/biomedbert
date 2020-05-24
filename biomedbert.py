#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""biomedbert

Usage:
  biomedbert gcp project set <project-id> <project-zone>
  biomedbert gcp vm start <vm-instance>
  biomedbert gcp vm stop <vm-instance>
  biomedbert gcp vm notebook <vm-instance>
  biomedbert gcp vm connect <vm-instance>
  biomedbert gcp vm create compute <vm-instance>
  biomedbert gcp vm create tpu <vm-instance> [preemptible]
  biomedbert gcp vm delete tpu <vm-instance>
  biomedbert code train vocab <data_path> <prefix>
  biomedbert code shard data <number_of_shards> <shard_path> <prc_data_path>
  biomedbert code make pretrain data <pre_trained_dir> <voc_filename> <shard_path>
  biomedbert code train model <model_type> <model_dir> <pretraining_dir> <bucket_name> <tpu_name> <tpu_cores>
  biomedbert code extract embeddings <input_txt> <voc_fname> <config_fname> <init_checkpoint>
  biomedbert glue download dataset
  biomedbert glue finetune <dataset> <model_dir> <checkpoint_name> <vocab_file> [<tpu_name>]
  biomedbert glue predict <dataset> <model_dir> <trained_classifier> <vocab_file> [<tpu_name>]
  biomedbert squad evaluate <evaluate_file> <predict_file> <prediction_json>
  biomedbert squad finetune (v1|v2) <model_dir> <train_file> <predict_file> <vocab_file> <init_checkpoint> <tpu_name>
  biomedbert ner finetune <model_type> <ner_dataset> <model_dir> <bucket_name> <tpu_name> <tpu_cores>
  biomedbert ner evaluate token level <model_type> <ner_dataset> <model_dir> <bucket_name> <tpu_name>
  biomedbert re finetune <re_dataset> <re_dataset_no> <model_dir> <init_checkpoint> <vocab_file> <tpu_name>
  biomedbert bioasq evaluate <model_dir> <train_file>
  biomedbert bioasq finetune <train_file> <predict_file> <model_dir> <init_checkpoint> <vocab_file> <tpu_name>
  biomedbert -h | --help
  biomedbert --version
Options:
  -h, --help                Show this screen.
  --version                 Show version.
"""

from __future__ import unicode_literals, print_function

import configparser
from docopt import docopt
from biomedbert_impl.modules import train_vocabulary, generate_pre_trained_data, shard_dataset, \
    extract_embeddings, train_biomedbert
from biomedbert_impl.gcp_helpers import set_gcp_project, start_vm, stop_vm, \
    launch_notebook, connect_vm, create_compute_vm, create_tpu_vm, delete_tpu_vm
from biomedbert_impl.glue_modules import fine_tune_classification_glue, download_glue_data, \
    predict_classification_glue
from biomedbert_impl.squad_modules import fine_tune_squad, evaluate_squad
from biomedbert_impl.bioasq_modules import fine_tune_bioasq, evaluate_bioasq
from biomedbert_impl.ner_modules import fine_tune_ner, token_level_evaluation
from biomedbert_impl.re_modules import fine_tune_re

__version__ = "0.1.0"
__author__ = "AI vs COVID-19 Team"
__license__ = "MIT"

# Configurations
config = configparser.ConfigParser()


def re_commands(args: dict):
    """Command to run Relation Extraction benchmark datasets"""

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # fine tune ner
    if args['re'] and args['finetune']:
        fine_tune_re(args['<re_dataset>'], args['<re_dataset_no>'], args['<model_dir>'],
                     args['<init_checkpoint>'], args['<vocab_file>'], args['<tpu_name>'], zone, project_id)


def ner_commands(args: dict):
    """Command to run Named Entity recognition benchmark datasets"""

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # fine tune ner
    if args['ner'] and args['finetune']:
        fine_tune_ner(args['<ner_dataset>'], args['<model_dir>'], args['<model_type>'], args['<bucket_name>'],
                      args['<tpu_name>'], zone, project_id, args['<tpu_cores>'])

    # token-level evaluation
    if args['ner'] and args['evaluate'] and args['token'] and args['level']:
        token_level_evaluation(args['<ner_dataset>'], args['<model_dir>'], args['<model_type>'], args['<bucket_name>'],
                               args['<tpu_name>'], zone, project_id, args['<tpu_cores>'])


def bioasq_commands(args: dict):
    """Command to run BIOASQ question answering benchmark datasets"""

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # fine tune bioasq
    if args['bioasq'] and args['finetune']:
        fine_tune_bioasq(args['<train_file>'], args['<predict_file>'], args['<model_dir>'],
                         args['<init_checkpoint>'], args['<vocab_file>'], args['<tpu_name>'], zone, project_id)

    # evaluate bioasq
    if args['bioasq'] and args['evaluate']:
        evaluate_bioasq(args['<model_dir>'], args['<train_file>'])


def squad_commands(args: dict):
    """Command to run SQuAD question answering benchmark dataset"""

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # fine tune squad
    if args['squad'] and args['finetune']:
        fine_tune_squad(args['v1'], args['<model_dir>'], args['<train_file>'], args['<predict_file>'],
                        args['<tpu_name>'], zone, project_id, args['<vocab_file>'], args['<init_checkpoint>'])

    # evaluate squad
    if args['squad'] and args['evaluate']:
        evaluate_squad(args['<evaluate_file>'], args['<predict_file>'], args['<prediction_json>'])


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

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # train biomedbert
    if args['code'] and args['train'] and args['model']:
        train_biomedbert(args['<model_type>'], args['<model_dir>'], args['<pretraining_dir>'],
                         args['<bucket_name>'], args['<tpu_name>'], project_id, zone, args['<tpu_cores>'])

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
        config.read('config/gcp_config.ini')
        zone = config['PROJECT']['zone']
        create_compute_vm(args['<vm-instance>'], zone)

    # create tpu
    if args['gcp'] and args['vm'] and args['create'] and args['tpu']:
        # read configurations
        config.read('config/gcp_config.ini')
        zone = config['PROJECT']['zone']
        create_tpu_vm(args['<vm-instance>'], zone, args['preemptible'])

    # delete tpu
    if args['gcp'] and args['vm'] and args['delete'] and args['tpu']:
        # read configurations
        config.read('config/gcp_config.ini')
        project_id = config['PROJECT']['name']
        zone = config['PROJECT']['zone']
        delete_tpu_vm(args['<vm-instance>'], project_id, zone)

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

    if args['re']:
        re_commands(args)

    if args['ner']:
        ner_commands(args)

    if args['bioasq']:
        bioasq_commands(args)


if __name__ == '__main__':
    main()
