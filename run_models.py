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
    run_models.py run <model_path> <metfile>
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

# import pals_utils as pu
from pals_utils.helpers import timeit, short_hash
from pals_utils.data import pals_site_name

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


met_vars = ["SWdown", "Tair", "LWdown", "Wind", "Rainf", "PSurf", "Qair"]
flux_vars = ["Qh", "Qle", "Rnet", "NEE"]


#################
# functions
#################

# Timing helpers

@timeit
def fit_pipeline(pipe, X, Y):
    pipe.fit(X, Y)


@timeit
def get_pipeline_prediction(pipe, X):
    return(pipe.predict(X))


# Pipeline helpers

def get_pipeline_name(pipe, suffix=None):
    if suffix is not None:
        return ", ".join(list(pipe.named_steps.keys()) + [suffix])
    else:
        return ", ".join(pipe.named_steps.keys())


# #############################
# ## fit - simulate - evaluate
# #############################

def get_model_fit_path(hash):
    return "cache/model_fits/%s.pickle" % hash


def fit_exists(fit_id, cache):
    if "model_fits/" + fit_id in cache:
        return cache["model_fits"][fit_id]["fit_hash"]


def fit_model_pipeline(pipe, met_data, flux_data, name, cache, clear_cache=False):
    """Top-level pipeline fitter.

    Fits a model, stores model and metadata.

    TODO: store domain metadata

    returns (pipe, fit_hash)
    """

    if name is None:
        name = get_pipeline_name(pipe)

    fit_id = short_hash((pipe, met_data, flux_data))

    fit_hash = fit_exists(fit_id, cache)
    if fit_hash is not None and not clear_cache:
        print("Model %s already fitted for %s, loading from file." % name, pals_site_name(met_data))
        with open(get_model_fit_path(fit_hash), "rb") as f:
            pipe = pickle.load(f)
    else:
        _, fit_time = fit_pipeline(pipe, met_data, flux_data)
        fit_hash = short_hash(pipe)
        cache["model_fits"][fit_id]["fit_hash"] = fit_hash
        cache["model_fits"][fit_id]["fit_time"] = fit_time
        cache.flush()
        with open(get_model_fit_path(fit_hash), "wb") as f:
            pickle.dump(pipe, f)

    return pipe, fit_hash


# Simulate

def get_sim_path(hash):
    return "cache/simulations/%s.pickle" % hash


def sim_exists(fit_hash, cache):
    if "simulations/" + fit_hash in cache:
        return cache["simulations"][fit_hash]["sim_hash"]


def simulate_model_pipeline(pipe, met_data, name, cache, clear_cache=False):
    """Top-level pipeline predictor.

    runs model, caches model simulation.

    returns (sim_data, sim_hash)
    """

    if name is None:
        name = get_pipeline_name(pipe)

    fit_hash = short_hash((pipe, met_data))

    sim_hash = sim_exists(pipe, met_data, cache)
    if sim_hash and not clear_cache:
        print("Model %s already simulated for %s, loading from file." % name, met_data.name)
        with open(get_sim_path(sim_hash), "rb") as f:
            sim_data = pickle.load(f)
    else:
        if met_data.met is None or met_data.flux is None:
            raise KeyError("missing met or flux data")
        sim_data, fit_time = get_pipeline_prediction(pipe, met_data)
        # TODO: If a simulation can produce more than one output for a given input, this won"t be unique. Is that ok?
        sim_hash = short_hash(sim_data)
        cache["simulations"][fit_hash]["sim_hash"] = sim_hash
        cache["simulations"][fit_hash]["model_predict_time"] = fit_time
        cache.flush()
        with open(get_model_fit_path(sim_hash), "wb") as f:
            pickle.dump(sim_data, f)

    return sim_data, sim_hash


def get_model(model_name):
    """return a scikit-learn model, and the required arguments

    :model_name: name of the model
    :returns: model object
    """
    # , Perceptron, PassiveAggressiveRegressor
    # , NuSVR, LinearSVR

    if model_name == 'lin':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    if model_name == 'sgd':
        from sklearn.linear_model import SGDRegressor
        return SGDRegressor()

    if model_name == 'svr':
        from sklearn.svm import SVR
        return SVR()
        # SVR(kernel="poly")

    if model_name == 'tree':
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor()

    if model_name == 'extratree':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor()

    if model_name == 'kneighbours':
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor()
        # KNeighborsRegressor(n_neighbors=1000)

    if model_name == 'mlp':
        from sklearn.neural_network import MultilayerPerceptronRegressor
        # This is from a pull request: https://github.com/scikit-learn/scikit-learn/pull/3939
        return MultilayerPerceptronRegressor()
        # MultilayerPerceptronRegressor(activation="logistic")
        # MultilayerPerceptronRegressor(hidden_layer_sizes=(10,10,))
        # MultilayerPerceptronRegressor(hidden_layer_sizes=(10,30,))
        # MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,))
        # MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,20,))

    raise Exception("Unknown Model")


def main_fit_model(args):
    """fit command

    :args: args passed from docopt
    :returns: TODO
    """
    # TODO: model arguments
    model = get_model(args["<model>"])

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
    name = get_pipeline_name(pipe)

    met_data = xray.open_dataset(args["<metfile>"])
    flux_data = xray.open_dataset(args["<metfile>"])

    if not os.path.exists("cache/"):
        os.mkdir("cache")
    cache = pd.HDFStore("cache/cache.hdf5")

    fit_model_pipeline(pipe, met_data, flux_data, name, cache)

    print(pipe)

    cache.close()


def main(args):

    # print(args)

    if args['fit']:
        main_fit_model(args)


if (__name__ == '__main__'):
    args = docopt(__doc__)

    main(args)
