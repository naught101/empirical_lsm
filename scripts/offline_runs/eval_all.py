#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: eval_all.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Runs all steps of the offline runs/ubermodel search page generator

Usage:
    eval_all.py <model>... [--sites=<sites>] [--run] [--multivariate] [--eval] [--html] [--rebuild] [--no-mp]
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


def main_eval_all(names, sites, run, multivariate, evalu, html, rebuild=False, no_mp=False):

    # All scripts already use multiprocessing
    if run:
        for name in names:
            main_run_mp(name, sites, no_mp, multivariate)

    if evalu:
        for name in names:
            main_eval_mp(name, sites, no_mp)
            main_rst_gen_mp(name, sites, no_mp)

    if html:
        model_site_index_rst_mp(names, rebuild, no_mp)
        model_search_index_rst()

        subprocess.call(['make', 'html'])


def main(args):

    if args['<model>'] == ['all']:
        names = get_available_models()
    else:
        names = args['<model>']

    sites = args['--sites']
    run = args['--run']
    multivariate = args['--multivariate']
    evalu = args['--eval']
    html = args['--html']
    rebuild = args['--rebuild']
    no_mp = args['--no-mp']

    main_eval_all(names, sites, run, multivariate, evalu, html, rebuild, no_mp)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
