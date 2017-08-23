#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: eval_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101/empirical_lsm
Description: Evaluates a model (sim or set of sims) and produces rst output with diagnostics

Usage:
    eval_model.py eval <name> <site> [<file>] [--no-mp] [--plot] [--no-fix-closure] [--no-qc]
    eval_model.py rst-gen <name> <site> [--no-mp]

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt

from empirical_lsm.offline_eval import eval_simulation_mp, main_rst_gen_mp

import logging
logger = logging.getLogger(__name__)
logger.basicConfig(filename='logs/eval_model.log')


def main(args):
    name = args['<name>']
    site = args['<site>']
    sim_file = args['<file>']
    plots = args['--plot']

    if args['eval']:
        eval_simulation_mp(name, site, sim_file, plots,
                           no_mp=args['--no-mp'],
                           fix_closure=not args['--no-fix-closure'],
                           qc=not args['--no-qc'])
    if args['rst-gen']:
        main_rst_gen_mp(name, site, sim_file,
                        no_mp=args['--no-mp'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
