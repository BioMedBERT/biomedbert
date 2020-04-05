#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""biomedbert

Usage:
  biomedbert gcp project set ([-p | --project] <project-id>) ([-z | --zone] <project-zone>)
  biomedbert gcp vm start <vm-instance>
  biomedbert gcp vm stop <vm-instance>
  biomedbert gcp vm notebook <vm-instance>
  biomedbert -h | --help
  biomedbert --version

Options:
  -h, --help    Show this screen.
  --version     Show version.
"""

from __future__ import unicode_literals, print_function
import configparser
from docopt import docopt
from gcp.gcp_helpers import set_gcp_project, start_vm, stop_vm, launch_notebook

__version__ = "0.1.0"
__author__ = "AI vs COVID-19 Team"
__license__ = "MIT"


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

    # start VM
    if args['gcp'] and args['vm'] and args['start']:
        # start vm
        start_vm(args['<vm-instance>'])

    # stop VM
    if args['gcp'] and args['vm'] and args['stop']:
        stop_vm(args['<vm-instance>'])

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


if __name__ == '__main__':
    main()
