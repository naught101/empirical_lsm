#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: models.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description:
"""

import pandas as pd
import os
import pickle

# import pals_utils as pu
from pals_utils.helpers import timeit, short_hash
from pals_utils.data import pals_site_name, pals_xray_to_df


MET_VARS = ["SWdown", "Tair", "LWdown", "Wind", "Rainf", "PSurf", "Qair"]
FLUX_VARS = ["Qh", "Qle", "Rnet", "NEE"]


#################
# functions
#################

# Timing helpers

@timeit
def fit_pipeline(pipe, met_ds, flux_ds):
    """Fit a pipeline, given a pair of xray datasets

    :param pipe:
    :param met_ds:
    :param flux_ds:
    """
    if isinstance(met_ds, list):
        # TODO: figure out list fitting
        pass
    met_df = pals_xray_to_df(met_ds, MET_VARS)
    flux_df = pals_xray_to_df(flux_ds, FLUX_VARS)
    pipe.fit(met_df, flux_df)


@timeit
def get_pipeline_prediction(pipe, met_ds):
    met_df = pals_xray_to_df(met_ds, MET_VARS)
    return(pipe.predict(met_df))


# Pipeline helpers

def get_pipeline_name(pipe, suffix=None):
    if suffix is not None:
        return ", ".join(list(pipe.named_steps.keys()) + [suffix])
    else:
        return ", ".join(pipe.named_steps.keys())


# #############################
# ## fit - simulate - evaluate
# #############################

def get_model_fit_path(fit_hash, fit_dir="cache/model_fits"):
    if not os.path.exists(fit_dir):
        os.mkdir(fit_dir)
    return "%s/%s.pickle" % (fit_dir, fit_hash)


def fit_exists(fit_id, cache):
    if "/model_fits" in cache.keys() and fit_id in cache.model_fits.index:
        return cache.model_fits.ix[fit_id, "fit_hash"]


def fit_model_pipeline(pipe, met_data, flux_data, name, cache, clear_cache=False):
    """Top-level pipeline fitter.

    Fits a model, stores model and metadata.

    TODO: store domain metadata

    returns (pipe, fit_hash)
    """

    if name is None:
        name = get_pipeline_name(pipe)

    layout_hash = short_hash(pipe)
    fit_id = '%s_%s' % (layout_hash, pals_site_name(met_data))

    fit_hash = fit_exists(fit_id, cache)
    if fit_hash is not None and not clear_cache:
        print("Model %s already fitted for %s, loading from file." % (name, pals_site_name(met_data)))
        filename = get_model_fit_path(fit_hash)
        with open(filename, "rb") as f:
            pipe = pickle.load(f)
    else:
        _, fit_time = fit_pipeline(pipe, met_data, flux_data)
        fit_hash = short_hash(pipe)

        if "/model_fits" in cache.keys():
            model_fits = cache.model_fits
        else:
            model_fits = pd.DataFrame()

        model_fits.ix[fit_id, "fit_hash"] = fit_hash
        model_fits.ix[fit_id, "fit_time"] = fit_time
        model_fits.ix[fit_id, "layout_hash"] = layout_hash
        model_fits.ix[fit_id, "name"] = name
        model_fits.ix[fit_id, "site"] = pals_site_name(met_data)

        cache['model_fits'] = model_fits
        cache.flush()

        with open(get_model_fit_path(fit_hash), "wb") as f:
            pickle.dump(pipe, f)

    return pipe, fit_hash


# Simulate

def get_sim_path(sim_hash, sim_dir="cache/simulations"):
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)
    return "%s/%s.pickle" % (sim_dir, sim_hash)


def sim_exists(sim_id, cache):
    if "/simulations" in cache.keys() and sim_id in cache.simulations.index:
        return cache.simulations.ix[sim_id, "sim_hash"]


def simulate_model_pipeline(pipe, met_data, name, cache, clear_cache=False):
    """Top-level pipeline predictor.

    runs model, caches model simulation.

    returns (sim_data, sim_hash)
    """

    if name is None:
        name = get_pipeline_name(pipe)

    fit_hash = short_hash(pipe)

    sim_id = '%s_%s' % (fit_hash, pals_site_name(met_data))

    sim_hash = sim_exists(sim_id, cache)
    if sim_hash is not None and not clear_cache:
        print("Model %s already simulated for %s, loading from file." % (name, met_data.name))
        with open(get_sim_path(sim_hash), "rb") as f:
            sim_data = pickle.load(f)
    else:
        if met_data.met is None or met_data.flux is None:
            raise KeyError("missing met or flux data")
        sim_data, pred_time = get_pipeline_prediction(pipe, met_data)
        # TODO: If a simulation can produce more than one output for a given input, this won"t be unique. Is that ok?
        sim_hash = short_hash(sim_data)

        if "/simulations" in cache.keys():
            simulations = cache.simulations
        else:
            simulations = pd.DataFrame()

        simulations.ix[sim_id, "fit_hash"] = fit_hash
        simulations.ix[sim_id, "site"] = pals_site_name(met_data)
        simulations.ix[sim_id, "sim_hash"] = sim_hash
        simulations.ix[sim_id, "predict_time"] = pred_time
        cache.simulations = simulations

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
