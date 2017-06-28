#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: eval_all.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Runs all steps of the offline runs/empirical_lsm search page generator

Usage:
    eval_all.py <model>... [--sites=<sites>] [--run] [--multivariate] [--eval [--plot]] [--rst] [--html] [--rebuild] [--no-mp] [--overwrite] [--no-fix-closure]
    eval_all.py (-h | --help | --version)

Options:
    -h, --help       Show this screen and exit.
    --sites=<sites>  Sites to run the models at [default: PLUMBER_ext]
"""

from docopt import docopt

import subprocess

from empirical_lsm.offline_simulation import run_simulation_mp
from empirical_lsm.offline_eval import eval_simulation_mp, main_rst_gen_mp
from scripts.offline_runs.model_search_indexes import model_site_index_rst_mp, model_search_index_rst, get_available_models


def eval_simulation_all(names, sites, run=False, multivariate=True, evalu=False,
                        plots=False, rst=False, html=False, rebuild=False, no_mp=False,
                        overwrite=False, fix_closure=True):

    # All scripts already use multiprocessing
    if run:
        for name in names:
            run_simulation_mp(name, sites, no_mp=no_mp, multivariate=multivariate,
                              overwrite=overwrite, fix_closure=fix_closure)

    if evalu:
        for name in names:
            eval_simulation_mp(name, sites, plots=plots, no_mp=no_mp, fix_closure=fix_closure)

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

    eval_simulation_all(names=names,
                        sites=args['--sites'],
                        run=args['--run'],
                        multivariate=args['--multivariate'],
                        evalu=args['--eval'],
                        plots=args['--plot'],
                        rst=args['--rst'],
                        html=args['--html'],
                        rebuild=args['--rebuild'],
                        no_mp=args['--no-mp'],
                        overwrite=args['--overwrite'],
                        fix_closure=not args['--no-fix-closure']
                        )

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
