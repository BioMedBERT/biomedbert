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
  biomedbert code train model <model_type> <model_dir> <pretrain_dir> <bucket_name> <tpu_name> <train_steps> <train_bs> <eval_bs> <tpu_cores>
  biomedbert code extract embeddings <input_txt> <voc_fname> <config_fname> <init_checkpoint>
  biomedbert glue download dataset
  biomedbert glue finetune <glue_dataset> <model_type> <bucket_name> <model_dir> <tpu_name> <tpu_cores>
  biomedbert glue predict <glue_dataset> <model_type> <bucket_name> <model_dir> <tpu_name> <tpu_cores>
  biomedbert squad evaluate (v1|v2) <bucket_name> <model_dir> <evaluate_file> <predict_file>
  biomedbert squad finetune (v1|v2) <model_type> <bucket_name> <model_dir> <train_file> <predict_file> <tpu_name> <tpu_cores>
  biomedbert ner finetune <model_type> <ner_dataset> <model_dir> <bucket_name> [<tpu_name> <tpu_cores>]
  biomedbert ner evaluate token level <model_type> <ner_dataset> <model_dir> <bucket_name> [<tpu_name> <tpu_cores>]
  biomedbert ner evaluate entity level <model_type> <ner_training_output_dir> <ner_data_dir>
  biomedbert re finetune <model_type> <re_dataset> <re_dataset_no> <model_dir> <bucket_name> <tpu_name> <tpu_cores>
  biomedbert re evaluate <re_dataset> <re_dataset_no> <model_dir> <bucket_name>
  biomedbert bioasq evaluate <bucket_name> <model_dir> <train_file> <eval_file> <squad_folder>
  biomedbert bioasq finetune <model_type> <train_file> <predict_file> <bucket_name> <model_dir> <squad_folder> [<tpu_name> <tpu_cores>]
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
from biomedbert_impl.ner_modules import fine_tune_ner, token_level_evaluation, word_level_prediction
from biomedbert_impl.re_modules import fine_tune_re, evaluate_re

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
        fine_tune_re(args['<re_dataset>'], args['<re_dataset_no>'], args['<model_dir>'], args['<model_type>'],
                     args['<bucket_name>'], args['<tpu_name>'], zone, project_id, args['<tpu_cores>'])

    # fine tune ner
    if args['re'] and args['evaluate']:
        evaluate_re(args['<re_dataset>'], args['<re_dataset_no>'], args['<model_dir>'], args['<bucket_name>'])


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

    # entity-level evaluation
    if args['ner'] and args['evaluate'] and args['entity'] and args['level']:
        word_level_prediction(args['<model_type>'], args['<ner_training_output_dir>'], args['<ner_data_dir>'])


def bioasq_commands(args: dict):
    """Command to run BIOASQ question answering benchmark datasets"""

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # fine tune bioasq
    if args['bioasq'] and args['finetune']:
        fine_tune_bioasq(args['<model_type>'], args['<bucket_name>'], args['<train_file>'], args['<predict_file>'],
                         args['<model_dir>'], args['<tpu_name>'], zone, project_id, args['<tpu_cores>'],
                         args['<squad_folder>'])

    # evaluate bioasq
    if args['bioasq'] and args['evaluate']:
        evaluate_bioasq(args['<bucket_name>'], args['<model_dir>'], args['<train_file>'], args['<eval_file>'],
                        args['<squad_folder>'])


def squad_commands(args: dict):
    """Command to run SQuAD question answering benchmark dataset"""

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # fine tune squad
    if args['squad'] and args['finetune']:
        fine_tune_squad(args['v1'], args['<model_type>'], args['<bucket_name>'], args['<model_dir>'],
                        args['<train_file>'], args['<predict_file>'], args['<tpu_name>'],
                        zone, project_id, args['<tpu_cores>'])

    # evaluate squad
    if args['squad'] and args['evaluate']:
        evaluate_squad(args['v1'], args['<bucket_name>'], args['<model_dir>'], args['<evaluate_file>'],
                       args['<predict_file>'])


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
        predict_classification_glue(args['<glue_dataset>'], args['<model_type>'], args['<model_dir>'],
                                    args['<bucket_name>'], args['<tpu_cores>'], args['<tpu_name>'], zone, project_id)

    # finetune glue
    if args['glue'] and args['finetune']:
        fine_tune_classification_glue(args['<glue_dataset>'], args['<model_type>'], args['<bucket_name>'],
                                      args['<model_dir>'], args['<tpu_name>'], zone, project_id, args['<tpu_cores>'])


def code_commands(args: dict):
    """Command to train BioMedBert model"""

    config.read('config/gcp_config.ini')
    zone = config['PROJECT']['zone']
    project_id = config['PROJECT']['name']

    # train biomedbert
    if args['code'] and args['train'] and args['model']:
        train_biomedbert(args['<model_type>'], args['<model_dir>'], args['<pretrain_dir>'], args['<bucket_name>'],
                         args['<tpu_name>'], project_id, zone, args['<train_steps>'],
                         args['<train_bs>'], args['<eval_bs>'], args['<tpu_cores>'])

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
