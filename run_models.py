#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_models.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description: Script for running empirical models over PALS-style fluxnet sites.

Usage:
    run_models.py fit <model> [options] <metfile> <fluxfile>
    run_models.py run <fit_hash> <metfile>
    run_models.py (-h | --help | --version)

Options:
    --scale=(std|minmax)  method to rescale data.
    --pca                 method to decompose the data
    --lag=<lag>           number of timesteps to lag the data (includes unlagged data)
    --poly=<poly>         number of timesteps to lag the data (includes unlagged data)
    -h, --help            Show this screen and exit.
"""

from docopt import docopt

import xray
import pandas as pd
import sys
import os
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from ubermodel import models as um


def main_fit_model(args):
    """fit command

    :args: args passed from docopt
    :returns: TODO
    """
    # TODO: model arguments
    model = um.get_model(args["<model>"])

    pipe_args = []
    if args['--scale'] is not None:
        if args['--scale'] == 'std':
            pipe_args.append(StandardScaler())
        elif args['--scale'] == 'minmax':
            pipe_args.append(MinMaxScaler())
        else:
            sys.exit('Unknown scaler %s' % args['--scale'])
    if args['--pca']:
        pipe_args.append(PCA())
    if args['--poly'] is not None:
        poly = int(args['--poly'])
        if 1 > poly or poly > 5:
            sys.exit('Poly features with n=%d is a dumb idea' % poly)
        pipe_args.append(PolynomialFeatures(poly))
    pipe_args.append(model)

    pipe = make_pipeline(*pipe_args)
    name = um.get_pipeline_name(pipe)

    met_data = xray.open_dataset(args["<metfile>"])
    flux_data = xray.open_dataset(args["<fluxfile>"])

    if not os.path.exists("cache/"):
        os.mkdir("cache")
    fit_cache = pd.HDFStore("cache/fit_cache.hdf5")

    pipe, fit_hash = um.fit_model_pipeline(pipe, met_data, flux_data, name, fit_cache)

    print("fit hash: ", fit_hash)
    print(pipe)

    fit_cache.close()

    return


def main_run_model(args):
    """run command

    :args: args passed from docopt
    :returns: TODO
    """
    model_path = um.get_model_fit_path(args["<fit_hash>"])
    with open(model_path, 'rb') as f:
        pipe = pickle.load(f)

    # TODO: get name from cache/smarter naming?
    name = um.get_pipeline_name(pipe)

    met_data = xray.open_dataset(args["<metfile>"])

    if not os.path.exists("cache/"):
        os.mkdir("cache")
    sim_cache = pd.HDFStore("cache/sim_cache.hdf5")

    sim_data, sim_hash = um.simulate_model_pipeline(pipe, met_data, name, sim_cache)

    print("sim hash: ", sim_hash)
    print(sim_data)

    sim_cache.close()

    return


def main(args):

    # print(args)

    if args['fit']:
        main_fit_model(args)

    if args['run']:
        main_run_model(args)

    return


if (__name__ == '__main__'):
    args = docopt(__doc__)

    main(args)
