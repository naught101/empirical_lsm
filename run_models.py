#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_models.py
Author: John Doe
Email: yourname@email.com
Github: https://github.com/yourname
Description: 


## Todo

- add more metrics
    - mutual info score
- multi variate output
- table of results
- Rhys: Compare the functional form of empirical models to that of LSMs, see where they differ
    - multivariate functional form
- clustered regression
- proper lagged regression
- markov regression (use outputs as inputs to next timestep)
- auto-diagram models (optional: include parameters)

"""

from docopt import docopt

import numpy as np
import xray
import pandas as pd
import os
import joblib
import pickle

from collections import OrderedDict

# import pals_utils as pu
from pals_utils.stats import metrics
from pals_utils.helpers import timeit, short_hash


doc = '''
Usage:
    run_models.py fit <model> [--] [<model_opts>...] [options] <metfile> <fluxfile>
    run_models.py run <model_path> <metfile>
    run_models.py (-h | --help | --version)

Options:
    --scale=(std|minmax)  method to rescale data.
    --pca                 method to decompose the data
    --lag=<lag>           number of timesteps to lag the data (includes unlagged data)
    --poly=<poly>         number of timesteps to lag the data (includes unlagged data)
    -h, --help            Show this screen and exit.
'''

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


def run_metrics(Y_pred, Y_validate, metrics):
    metric_data = OrderedDict()
    for (n, m) in metrics.items():
        metric_data[n] = m(Y_pred, Y_validate)
    return metric_data


# #############################
# ## fit - simulate - evaluate
# #############################

def get_model_fit_path(hash):
    return "cache/model_fits/%s.pickle" % hash


def fit_exists(fit_id, cache):
    if "model_fits/" + fit_id in cache:
        return cache["model_fits"][fit_id]["fit_hash"]


def fit_model_pipeline(pipe, land_data, name, cache, clear_cache=False):
    """Top-level pipeline fitter.

    Fits a model, stores model and metadata.

    TODO: store domain metadata

    returns (pipe, fit_hash)
    """

    if name is None:
        name = get_pipeline_name(pipe)

    fit_id = joblib.hash((pipe, land_data))

    fit_hash = fit_exists(fit_id, cache)
    if fit_hash is not None and not clear_cache:
        print("Model %s already fitted for %s, loading from file." % name, land_data.name)
        with open(get_model_fit_path(fit_hash), "rb") as f:
            pipe = pickle.load(f)
    else:
        if land_data.met is None or land_data.flux is None:
            raise KeyError("missing met or flux data")
        fit_time = fit_pipeline(pipe, land_data.met, land_data.flux)[1]
        fit_hash = joblib.hash(pipe)
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

    fit_hash = joblib.hash((pipe, met_data))

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
        sim_hash = joblib.hash(sim_data)
        cache["simulations"][fit_hash]["sim_hash"] = sim_hash
        cache["simulations"][fit_hash]["model_predict_time"] = fit_time
        cache.flush()
        with open(get_model_fit_path(sim_hash), "wb") as f:
            pickle.dump(sim_data, f)

    return sim_data, sim_hash


# Evaluate

# TODO: Could make a wrapper around this so that you can just pass a hash, or a fit model, and auto-load the data.

def evaluation_exists(sim_data, land_data, cache):
    eval_hash = joblib.hash((sim_data, land_data))

    if ("metric_data" in cache) and (eval_hash in cache.metric_data.index[0]):
        return eval_hash


def evaluate_simulation(sim_data, land_data, y_vars, name, cache, clear_cache=False):
    """Top-level simulation evaluator.

    Compares sim_data to land_data, using standard metrics. Stores the results in an easily accessible format.
    """

    eval_hash = joblib.hash((sim_data, land_data))

    index = {"eval_hash": eval_hash,
             "name": name,
             "site": land_data.name,
             "var": y_vars[0]}

    if "metric_data" in cache and not clear_cache:
        if eval_hash in cache.metric_data.index[0]:
            print("Metrics already calculated for %s, skipping." % name)
            return cache.metric_data
        metric_data = cache.metric_data
    else:
        metric_data = pd.DataFrame([index])
        metric_data = metric_data.set_index(list(index.keys()))

    for y_var in y_vars:
        Y_sim = np.array(sim_data.flux[y_var])
        Y_obs = np.array(land_data.flux[y_var])

        row_id = tuple(list(index.values())[0:3] + [y_var])
        metric_data.ix[row_id, "name"] = "%s_%s" % (y_var)
        metric_data.ix[row_id, "sim_id"] = joblib.hash(sim_data)
        metric_data.ix[row_id, "site"] = land_data.name
        metric_data.ix[row_id, "var"] = y_var

        for k, v in run_metrics(Y_sim, Y_obs, metrics).items():
            metric_data.ix[row_id, k] = v

    cache["metric_data"] = metric_data
    cache.flush()

    return metric_data.loc[eval_hash]


def test_model_pipeline(pipe=flux_data, y_vars, name, cache, plot=False, clear_cache=False):
    """Top-level pipeline fitter and tester.

    Fits and predicts with a model, runs metrics, optionally runs some diagnostic plots.
    """

    (train_data, test_data) = flux_data.time_split(0.7)

    fit_hash = short_hash(pipe, flux_data)
    sim_hash = fit_exists(fit_hash)
    eval_hash = sim_exists(sim_hash)


def get_model(model_name):
    """return a scikit-learn model, and the required arguments

    :model_name: name of the model
    :returns: model object
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline

    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import SGDRegressor
    # , Perceptron, PassiveAggressiveRegressor
    from sklearn.svm import SVR
    # , NuSVR, LinearSVR
    # from sklearn.neural_network import MultilayerPerceptronRegressor
    # This is from a pull request: https://github.com/scikit-learn/scikit-learn/pull/3939
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor

    if model_name == 'lin':
        return LinearRegression

DecisionTreeRegressor())
ExtraTreesRegressor())
KNeighborsRegressor(n_neighbors=1000))
LinearRegression())
MultilayerPerceptronRegressor())
MultilayerPerceptronRegressor(activation="logistic"))
MultilayerPerceptronRegressor(hidden_layer_sizes=(10,10,)))
MultilayerPerceptronRegressor(hidden_layer_sizes=(10,30,)))
MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,)))
MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,20,)))
PCA()
PolynomialFeatures(2)
PolynomialFeatures(5)
SGDRegressor())
SVR())
SVR(kernel="poly"))
StandardScaler()

def main(args):

    print(args)
    return

    if args['fit']:
        met_data = xray.open_dataset(args["<metfile>"])
        flux_data = xray.open_dataset(args["<metfile>"])

        if not os.path.exists("cache/"):
            os.mkdir("cache")
        cache = pd.HDFStore("cache/cache.hdf5")

        model = get_model(args["<model_name>"])

        pipe_args = []
        if args['--scale']:
            print(' lah')

        fit_model_pipeline(pipe, met_data, flux_data, name, cache)


if (__name__ == '__main__'):
    args = docopt(doc)

    main(args)






