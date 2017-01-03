#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: eval_all.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Runs all steps of the offline runs/ubermodel search page generator

Usage:
    eval_all.py <model>... [--sites=<sites>] [--run] [--multivariate] [--eval [--no-plots]] [--rst] [--html] [--rebuild] [--no-mp] [--overwrite]
    eval_all.py (-h | --help | --version)

Options:
    -h, --help       Show this screen and exit.
    --sites=<sites>  Sites to run the models at [default: PLUMBER_ext]
"""

from docopt import docopt

import subprocess

from scripts.offline_runs.run_model import main_run_mp
from scripts.offline_runs.eval_model import main_eval_mp, main_rst_gen_mp
from scripts.offline_runs.model_search_indexes import model_site_index_rst_mp, model_search_index_rst, get_available_models


def main_eval_all(names, sites, run=False, multivariate=True, evalu=False,
                  plots=True, rst=False, html=False, rebuild=False, no_mp=False, overwrite=False):

    # All scripts already use multiprocessing
    if run:
        for name in names:
            main_run_mp(name, sites, no_mp=no_mp, multivariate=multivariate, overwrite=overwrite)

    if evalu:
        for name in names:
            main_eval_mp(name, sites, plots=plots, no_mp=no_mp)

    if rst:
        for name in names:
            main_rst_gen_mp(name, sites, no_mp=no_mp)
        model_site_index_rst_mp(names, rebuild, no_mp=no_mp)
        model_search_index_rst()

    if html:
        subprocess.call(['make', 'html'])


def main(args):

    if args['<model>'] == ['all']:
        names = get_available_models()
    else:
        names = args['<model>']

    main_eval_all(names=names,
                  sites=args['--sites'],
                  run=args['--run'],
                  multivariate=args['--multivariate'],
                  evalu=args['--eval'],
                  plots=not args['--no-plots'],
                  rst=args['--rst'],
                  html=args['--html'],
                  rebuild=args['--rebuild'],
                  no_mp=args['--no-mp'],
                  overwrite=args['--overwrite']
                  )

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
