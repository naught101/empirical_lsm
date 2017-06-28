#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model_combos.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Runs and evaluates combination models of forcings that appear to work

Usage:
    model_combos.py [--run] [--eval] [--multivariate] [--eval [--plot]] [--no-mp] [--sites=<sites>] [--overwrite] [--no-fix-closure]
    model_combos.py (-h | --help | --version)

Options:
    -h, --help       Show this screen and exit.
    --sites=<sites>  Sites to run the models at [default: PLUMBER_ext]
"""

from docopt import docopt

from empirical_lsm.model_sets import get_combo_model_names

from empirical_lsm.offline_simulation import run_simulation_mp
from empirical_lsm.offline_eval import eval_simulation_mp


def main(sites, run=False, multivariate=True, evalu=False, plots=False,
         no_mp=False, overwrite=False, fix_closure=True):

    names = get_combo_model_names()

    if args['--run']:
        for name in names:
            run_simulation_mp(name, sites, no_mp=no_mp, multivariate=multivariate,
                              overwrite=overwrite, fix_closure=fix_closure)

    if args['--eval']:
        for name in names:
            eval_simulation_mp(name, sites, plots=plots, no_mp=no_mp,
                               fix_closure=fix_closure)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(sites=args['--sites'],
         run=args['--run'],
         multivariate=args['--multivariate'],
         evalu=args['--eval'],
         plots=args['--plot'],
         no_mp=args['--no-mp'],
         overwrite=args['--overwrite'],
         fix_closure=not args['--no-fix-closure']
         )
