#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: eval_all.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Runs all steps of the offline runs/ubermodel search page generator

Usage:
    eval_all.py <model>... [--sites=<sites>] [--rebuild] [--no-mp]
    eval_all.py (-h | --help | --version)

Options:
    -h, --help       Show this screen and exit.
    --sites=<sites>  Sites to run the models at [default: PLUMBER_ext]
"""

from docopt import docopt

import subprocess

from scripts.offline_runs.run_model import main_run_mp
from scripts.offline_runs.eval_model import main_eval_mp, main_rst_gen_mp
from scripts.offline_runs.model_search_indexes import model_site_index_rst_mp, model_search_index_rst



def main_eval_all(names, sites, rebuild=False, no_mp=False):

    # All scripts already use multiprocessing
    for name in names:
        main_run_mp(name, sites, rebuild, no_mp)
        main_eval_mp(name, sites, rebuild, no_mp)
        main_rst_gen_mp(name, sites, rebuild, no_mp)

    model_site_index_rst_mp(names, rebuild, no_mp)
    model_search_index_rst()
    
    subprocess.call('make', 'html')


def main(args):

    main_eval_all(args['<model>'], args['--sites'], args['--rebuild'], args['--no-mp'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
