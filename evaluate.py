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
from datetime import datetime

# import pals_utils as pu
from pals_utils.stats import run_metrics
from pals_utils.helpers import short_hash


# Evaluate

def evaluation_exists(eval_id, cache):
    """Check if evaluation for this set of simulation and obs data exists"""
    if ("metric_data" in cache) and (eval_id in cache.metric_data.index):
        return cache.metric_data.ix[eval_id, 'eval_hash']


def evaluate_simulation(sim_data, flux_data, model_name, flux_vars, cache):
    """Top-level simulation evaluator.

    Compares sim_data to flux_data, using standard metrics. Stores the results in an easily accessible format.

    TODO: Maybe get model model_name from sim_data directly (this is a PITA at
          the moment, most models don't report it).
    """

    eval_hash = short_hash((sim_data, flux_data))[0:7]
    eval_time = datetime.now().isoformat()

    # TODO: This currently returns "TumbaFluxnet" - would be nice to return the proper name.
    #       Probably should be fixed in PALS.
    site = pals_site_name(flux_data)

    index = {"eval_hash": "%s_%s" % (eval_hash, flux_vars[0])}

    if "metric_data" in cache:
        metric_data = cache.metric_data
    else:
        metric_data = pd.DataFrame([index])
        metric_data = metric_data.set_index(list(index.keys()))

    for y_var in flux_vars:
        Y_sim = sim_data[y_var].values.ravel()
        Y_obs = flux_data[y_var].values.ravel()

        row_id = "%s_%s" % (eval_hash, y_var)
        metric_data.ix[row_id, "eval_hash"] = eval_hash
        metric_data.ix[row_id, "model_name"] = model_name
        metric_data.ix[row_id, "sim_hash"] = short_hash(sim_data)
        metric_data.ix[row_id, "site"] = site
        metric_data.ix[row_id, "var"] = y_var
        metric_data.ix[row_id, "eval_time"] = eval_time

        for k, v in run_metrics(Y_sim, Y_obs).items():
            metric_data.ix[row_id, k] = v

    cache["metric_data"] = metric_data
    cache.flush()

    return metric_data.query("eval_hash == '%s'" % eval_hash)


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

        ev = evaluate_simulation(sim_data, land_data, args["<model_name>"], flux_vars, cache)

        cache.close()

        print(ev)


if (__name__ == '__main__'):
    args = docopt(__doc__)

    main(args)
