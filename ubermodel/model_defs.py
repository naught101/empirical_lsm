#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model_defs.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Models created programattically
"""

from collections import OrderedDict
import re

from sklearn.linear_model import LinearRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ubermodel.clusterregression import ModelByCluster
from ubermodel.transforms import MissingDataWrapper, LagAverageWrapper, MarkovLagAverageWrapper, Mean


def km_regression(k, model):
    return MissingDataWrapper(make_pipeline(StandardScaler(), ModelByCluster(MiniBatchKMeans(k), model)))


def km_lin(k):
    return km_regression(k, LinearRegression())


def cur_3_var():
    var_lags = OrderedDict()
    [var_lags.update({v: ['cur']}) for v in ['SWdown', 'Tair', 'RelHum']]
    return var_lags


def MLP(*args, **kwargs):
    """Multilayer perceptron """
    return MissingDataWrapper(make_pipeline(StandardScaler(), MLPRegressor(*args, **kwargs)))


def get_var_name(v):
    fvars = {"S": "SWdown",
             "T": "Tair",
             "H": "RelHum",
             "W": "Wind",
             "R": "Rainf",
             "L": "LWdown",
             "Q": "Qair"}
    return fvars[v]


def add_var_lag(var_dict, v, lag='cur'):
    if v not in var_dict:
        var_dict.update({v: [lag]})
    else:
        var_dict[v] += [lag]


def parse_model_name(name):
    """parses a standard model name"
    """

    name_original = name
    var_lags = OrderedDict()

    while len(name) > 0:
        token = name[0]
        name = name[1:]
        if token in 'STHWRLQ':
            add_var_lag(var_lags, get_var_name(token))
            continue
        if token == 'd':
            v = name[0]
            add_var_lag(var_lags, v + 'growth')
            name = name[1:]
            continue
        elif token == '_':
            if name.startswith('lin'):  # linear model
                model_name = 'lin'
                name = name[3:]
                continue
            elif name.startswith('l'):  # lagged var:
                match = re.match('l([A-Z])([0-9]*[a-z]*)(M*)', name)
                groups = match.groups()
                add_var_lag(var_lags, get_var_name(groups[0]), groups[1] + groups[2])
                name = name[len(match.group()):]
                continue
            elif name.startswith('km'):  # k means regression
                model_name = 'km'
                match = re.match('km([0-9]*)', name)
                k = int(match.groups()[0])
                name = name[len(match.group()):]
                continue
            elif name.startswith('mean'):  # Cluster-mean
                model_name = 'mean'
                name = name[4:]
                continue
            elif name.startswith('RF'):
                model_name = 'randomforest'
                name = name[2:]
                continue
            elif name.startswith('ET'):
                model_name = 'extratrees'
                name = name[2:]
                continue
            elif name.startswith('AB'):
                model_name = 'adaboost'
                name = name[2:]
                continue
        elif token == '.':  # model duplicate - do nothing
            name = name.lstrip('.0123456789')
            continue
        raise NameError('Unmatched token in name: ' + name)

    if model_name == 'lin':
        model = MissingDataWrapper(LinearRegression())
        desc = 'lin'
    elif model_name == 'mean':
        model = MissingDataWrapper(Mean())
        desc = 'mean'
    elif model_name == 'km':
        model = km_regression(k, LinearRegression())
        desc = 'km' + str(k)
    elif model_name == 'randomforest':
        from sklearn.ensemble import RandomForestRegressor
        model = MissingDataWrapper(RandomForestRegressor(n_estimators=100))
        desc = 'RandomForest'
        memory_req = 20e9
    elif model_name == 'extratrees':
        from sklearn.ensemble import ExtraTreesRegressor
        model = MissingDataWrapper(ExtraTreesRegressor(n_estimators=100))
        desc = 'ExtraTrees'
        memory_req = 30e9
    elif model_name == 'adaboost':
        from sklearn.ensemble import AdaBoostRegressor
        model = MissingDataWrapper(AdaBoostRegressor(n_estimators=100))
        desc = 'AdaBoost'
        memory_req = 20e9

    desc = desc + " model with"

    if any([l != ['cur'] for l in var_lags.values()]):
        model = LagAverageWrapper(var_lags, model)

    model.forcing_vars = list(var_lags)

    cur_vars = []
    lag_vars = []
    for k, v in var_lags.items():
        if 'cur' in v:
            cur_vars += [k]
        if len(v) > 0:
            for l in v:
                if v != 'cur':
                    lag_vars += ['Lagged ' + k + ' (' + l + ')']
    desc += ' with ' + ', '.join(cur_vars)
    if len(lag_vars) > 0:
        desc += ', ' + ', '.join(lag_vars)
    desc += ' (parsed)'
    model.description = desc

    model.name = name_original

    if 'memory_req' in locals():
        model.memory_requirement = memory_req

    return model


def get_model_from_def(name):
    """returns a scikit-learn style model/pipeline

    :name: model name
    :returns: scikit-learn style mode/pipeline

    """
    # PLUMBER-style benchmarks
    if name == '1lin':
        model = MissingDataWrapper(LinearRegression())
        model.forcing_vars = ['SWdown']
        model.description = "PLUMBER-style 1lin (SWdown only)"

    elif name == '3km27':
        model = km_lin(27)
        model.forcing_vars = ['SWdown', 'Tair', 'RelHum']
        model.description = "PLUMBER-style 3km27 (SWdown, Tair, RelHum)"

    # higher non-linearity
    elif name == '3km243':
        model = km_lin(243)
        model.forcing_vars = ['SWdown', 'Tair', 'RelHum']
        model.description = "Like 3km27, but with more clusters"

    # All lagged-inputs
    elif name == '3km27_lag':
        model_dict = {
            'variable': ['SWdown', 'Tair', 'RelHum'],
            'clusterregression': {
                'class': MiniBatchKMeans,
                'args': {
                    'n_clusters': 27}
            },
            'class': LinearRegression,
            'lag': {
                'periods': 1,
                'freq': 'D'}
        }
        from .models import get_model_from_dict
        model = MissingDataWrapper(get_model_from_dict(model_dict))
        model.forcing_vars = ['SWdown', 'Tair', 'RelHum']
        model.description = "like 3km27, but includes 1-day lagged versions of all three variables"

    # Many variables, lags. Doesn't work very well... (not enough non-linearity?)
    elif name == '5km27_lag':
        var_lags = OrderedDict()
        [var_lags.update({v: ['cur', '2d', '7d']}) for v in ['SWdown', 'Tair', 'RelHum', 'Wind']]
        var_lags.update({'Rainf': ['cur', '2d', '7d', '30d', '90d']})
        model = LagAverageWrapper(var_lags, km_lin(27))
        model.forcing_vars = list(var_lags)
        model.description = "km27 linear regression with SW, T, RH, Wind, Rain, and 2 and 7 day lagged-averages for each, plus 30- and 90-day lagged averages for Rainf (probably needs more clusters...)"

    # 3km243 with lagged Rainf
    elif name == 'STH_lR2d30d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2d', '30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2d,30d)"

    # 3km243 with lagged Wind
    elif name == 'STH_lW2d30d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['2d', '30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (2d,30d)"

    # Lagged and non-lagged rainfall
    elif name == 'STHR_lR_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['cur', '2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, Rainf, and Lagged Rainf (2d)"

    # Markov-lagged Qle variants (doesn't seem to be working very well)
    elif name == 'STH_lQle30min_km243':
        var_lags = cur_3_var()
        var_lags.update({'Qle': ['30min']})
        model = MarkovLagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(['SWdown', 'Tair', 'RelHum'])
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Markov-Lagged Qle (30min)"
    elif name == 'STH_lQle1h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Qle': ['1h']})
        model = MarkovLagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(['SWdown', 'Tair', 'RelHum'])
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Markov-Lagged Qle (1h)"
    elif name == 'STH_lQle2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Qle': ['2d']})
        model = MarkovLagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(['SWdown', 'Tair', 'RelHum'])
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Markov-Lagged Qle (2d)"

    # Neural network models
    elif name == 'STH_MLP':
        var_lags = cur_3_var()
        model = LagAverageWrapper(var_lags, MLP((15, 10, 5, 10)))
        model.forcing_vars = list(var_lags)
        model.description = "Neural-network model with Swdown, Tair, RelHum"
    elif name == 'STH_MLP_lR2d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, MLP((15, 10, 5, 10)))
        model.forcing_vars = list(var_lags)
        model.description = "Neural-network model with Swdown, Tair, RelHum, and Lagged Rainf (2d)"

    else:
        raise NameError("unknown model")

    return model
