#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101/empirical_lsm
Description: Fits and runs a basic model.

Usage:
    run_model.py run <name> <site> [--no-mp] [--multivariate] [--overwrite] [--no-fix-closure]

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt

from pals_utils.data import set_config

from empirical_lsm.offline_simulation import run_simulation_mp

import logging
logger = logging.getLogger(__name__)
logger.basicConfig(filename='logs/run_model.log')

set_config(['vars', 'flux'], ['NEE', 'Qle', 'Qh'])


def main(args):
    name = args['<name>']
    site = args['<site>']

    run_simulation_mp(name, site,
                      no_mp=args['--no-mp'],
                      multivariate=args['--multivariate'],
                      overwrite=args['--overwrite'],
                      fix_closure=not args['--no-fix-closure'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
