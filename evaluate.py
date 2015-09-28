#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: evaluation.py
Author: ned haughton
Email: ned@nedhaughton.com
Github:
Description: PALS-style model evaluation

Usage:
    evaluation.py metrics [options] <model_name> <sim_file> <flux_file>
    evaluation.py cache-clear [--cache=<cache>]
    evaluation.py plumber_plots [<sim_file>, ...]
    evaluation.py (-h | --help | --version)

Options:
    -h, --help        Show this screen and exit.
    --flux_vars=F...  Flux variables of interest [default: Qh,Qle,Rnet,NEE,GPP]
    --cache=CACHE     Cache path to use for metrics and metadata [default: cache/]
"""

from docopt import docopt

import xray
import pandas as pd
import os
import glob

from ubermodel import evaluate as ue

def main(args):

    # print(args)

    if args["cache-clear"]:

        cache_dir = args["--cache"]
        if not os.path.exists(cache_dir):
            print("Cache doesn't exist..")
            return
        for f in glob.glob(os.path.join(cache_dir, "eval_cache.hdf5")):
            os.remove(f)
        print("cleared eval data from %s" % cache_dir)

    elif args['metrics']:

        cache_dir = args["--cache"]
        if not os.path.exists(cache_dir):
            print("Using cache %s" % cache_dir)
            os.mkdir(cache_dir)

        cache = pd.HDFStore(os.path.join(cache_dir, "eval_cache.hdf5"))

        sim_data = xray.open_dataset(args['<sim_file>'])
        land_data = xray.open_dataset(args['<flux_file>'])

        flux_vars = args["--flux_vars"].split(',')

        ev = ue.evaluate_simulation(sim_data, land_data, args["<model_name>"], flux_vars, cache)

        cache.close()

        print(ev)


if (__name__ == '__main__'):
    args = docopt(__doc__)

    main(args)
