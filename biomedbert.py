#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""biomedbert

Usage:
  biomedbert gcp project set <project-id> <project-zone>
  biomedbert gcp vm start <vm-instance>
  biomedbert gcp vm stop <vm-instance>
  biomedbert gcp vm notebook <vm-instance>
  biomedbert gcp vm connect <vm-instance>
  biomedbert gcp vm create compute tpu <vm-instance> <project-zone>
  biomedbert code train vocab <data_path> <prefix>
  biomedbert code extract embeddings <input_txt> <voc_fname> <config_fname> <init_checkpoint>
  biomedbert code shard data <number_of_shards> <shard_path> <prc_data_path>
  biomedbert code make pretrain data <pre_trained_dir> <voc_filename> <shard_path>
  biomedbert -h | --help
  biomedbert --version

Options:
  -h, --help    Show this screen.
  --version     Show version.
"""

from __future__ import unicode_literals, print_function

import configparser
from docopt import docopt
from biomedbert_impl.modules import train_vocabulary, generate_pre_trained_data, shard_dataset, extract_embeddings
from biomedbert_impl.gcp_helpers import set_gcp_project, start_vm, stop_vm,\
    launch_notebook, connect_vm, create_compute_tpu_vm

__version__ = "0.1.0"
__author__ = "AI vs COVID-19 Team"
__license__ = "MIT"


def code_commands(args: dict):
    """Command to train BioMedBert model"""

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

    # Configurations
    config = configparser.ConfigParser()

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

    # create compute and tpu VM
    if args['gcp'] and args['vm'] and args['create'] and args['compute'] and args['tpu']:
        # start vm
        create_compute_tpu_vm(args['<vm-instance>'], args['<project-zone>'])

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


if __name__ == '__main__':
    main()
