#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''biomedbert

Usage:
  biomedbert ship new <name>...
  biomedbert ship <name> move <x> <y> [--speed=<kn>]
  biomedbert ship shoot <x> <y>
  biomedbert mine (set|remove) <x> <y> [--moored|--drifting]
  biomedbert -h | --help
  biomedbert --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --speed=<kn>  Speed in knots [default: 10].
  --moored      Moored (anchored) mine.
  --drifting    Drifting mine.
'''

from __future__ import unicode_literals, print_function
from docopt import docopt

__version__ = "0.1.0"
__author__ = "AI vs COVID-19 Team"
__license__ = "MIT"


def main():
    '''Main entry point for the biomedbert CLI.'''
    args = docopt(__doc__, version=__version__)
    print(args)

if __name__ == '__main__':
    main()