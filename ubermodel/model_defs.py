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


def km_regression(n, model):
    return MissingDataWrapper(ModelByCluster(MiniBatchKMeans(27), model))


def km_lin(n):
    return km_regression(n, LinearRegression())


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

    var_lags = OrderedDict()
    kmeans = None
    model = 'lin'

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
            if name.startswith('l'):  # lagged var:
                lag = re.match('l([A-Z])([0-9]*[a-z]*)', name)
                add_var_lag(var_lags, get_var_name(lag.groups()[0]), lag.groups()[1])
                name = name[len(lag.group()):]
                continue
            elif name.startswith('km'):  # k means regression
                match = re.match('km([0-9]*)', name)
                kmeans = int(match.groups()[0])
                name = name[len(match.group()):]
                continue
            elif name.startswith('mean'):
                model = 'mean'
                name = name[4:]
                continue
        raise NameError('Unmatched token in name: ' + name)

    desc = "model with"

    if model == 'lin':
        model = LinearRegression()
        desc = 'lin ' + desc
    elif model == 'mean':
        model = Mean()
        desc = 'mean ' + desc

    if kmeans is not None:
        model = km_regression(kmeans, model)
        desc = 'km' + str(kmeans) + ' ' + desc

    if any([l != ['cur'] for l in var_lags.values()]):
        model = LagAverageWrapper(var_lags, model)
    else:
        model.forcing_vars = list(var_lags)

    cur_vars = []
    lag_vars = []
    for k, v in var_lags.items():
        if 'cur' in v:
            v.remove('cur')
            cur_vars += [k]
        if len(v) > 0:
            for l in v:
                lag_vars += ['Lagged ' + k + ' (' + l + ')']
    desc += ' with ' + ', '.join(cur_vars)
    if len(lag_vars) > 0:
        desc += ', ' + ', '.join(lag_vars)
    desc += ' (parsed)'
    model.description = desc

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
    elif name == 'STH_lR30min_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['30min']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (30min)"
    elif name == 'STH_lR1h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['1h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (1h)"
    elif name == 'STH_lR2h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2h)"
    elif name == 'STH_lR6h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['6h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (6h)"
    elif name == 'STH_lR12h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['12h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (12h)"
    elif name == 'STH_lR2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2d)"
    elif name == 'STH_lR10d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['10d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (10d)"
    elif name == 'STH_lR30d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (30d)"
    elif name == 'STH_lR60d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['60d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (60d)"
    elif name == 'STH_lR90d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['90d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (90d)"
    elif name == 'STH_lR180d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['180d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (180d)"
    elif name == 'STH_lR2d30d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2d', '30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2d,30d)"

    # 3km243 with lagged Wind
    elif name == 'STH_lW30min_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['30min']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (30min)"
    elif name == 'STH_lW1h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['1h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (1h)"
    elif name == 'STH_lW2h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['2h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (2h)"
    elif name == 'STH_lW6h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['6h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (6h)"
    elif name == 'STH_lW12h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['12h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (12h)"
    elif name == 'STH_lW2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (2d)"
    elif name == 'STH_lW10d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['10d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (10d)"
    elif name == 'STH_lW30d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (30d)"
    elif name == 'STH_lW180d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['180d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (180d)"
    elif name == 'STH_lW2d30d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['2d', '30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (2d,30d)"

    # 3km243 with lagged LWdown
    elif name == 'STH_lL30min_km243':
        var_lags = cur_3_var()
        var_lags.update({'LWdown': ['30min']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged LWdown (30min)"
    elif name == 'STH_lL1h_km243':
        var_lags = cur_3_var()
        var_lags.update({'LWdown': ['1h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged LWdown (1h)"
    elif name == 'STH_lL12h_km243':
        var_lags = cur_3_var()
        var_lags.update({'LWdown': ['12h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged LWdown (12h)"
    elif name == 'STH_lL2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'LWdown': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged LWdown (2d)"

    # 3km243 with lagged SWdown
    elif name == 'STH_lS30min_km243':
        var_lags = cur_3_var()
        var_lags.update({'SWdown': ['30min']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged SWdown (30min)"
    elif name == 'STH_lS1h_km243':
        var_lags = cur_3_var()
        var_lags.update({'SWdown': ['1h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged SWdown (1h)"
    elif name == 'STH_lS12h_km243':
        var_lags = cur_3_var()
        var_lags.update({'SWdown': ['12h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged SWdown (12h)"
    elif name == 'STH_lS2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'SWdown': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged SWdown (2d)"

    # 3km243 with lagged Tair
    elif name == 'STH_lT30min_km243':
        var_lags = cur_3_var()
        var_lags.update({'Tair': ['30min']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Tair (30min)"
    elif name == 'STH_lT1h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Tair': ['1h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Tair (1h)"
    elif name == 'STH_lT12h_km243':
        var_lags = cur_3_var()
        var_lags.update({'Tair': ['12h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Tair (12h)"
    elif name == 'STH_lT2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Tair': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Tair (2d)"

    # 3km243 with lagged RelHum
    elif name == 'STH_lH30min_km243':
        var_lags = cur_3_var()
        var_lags.update({'RelHum': ['30min']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged RelHum (30min)"
    elif name == 'STH_lH1h_km243':
        var_lags = cur_3_var()
        var_lags.update({'RelHum': ['1h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged RelHum (1h)"
    elif name == 'STH_lH12h_km243':
        var_lags = cur_3_var()
        var_lags.update({'RelHum': ['12h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged RelHum (12h)"
    elif name == 'STH_lH2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'RelHum': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged RelHum (2d)"

    # Building the Heirarchy
    elif name == 'STHR_lR2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['cur', '2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, Rainf, and Lagged Rainf (2d)"
    elif name == 'STHW_lR2d_km243':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['cur']})
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, Wind, and Lagged Rainf (2d)"
    elif name == 'STH_lR2d_km729':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(729))
        model.forcing_vars = list(var_lags)
        model.description = "km729 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2d)"

    # Lagged and non-lagged rainfall
    elif name == 'STHR_lR_km243':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['cur', '2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, Rainf, and Lagged Rainf (2d)"

    # Wind plus lagged Rainf
    elif name == 'STHW_lR2d_km729':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['cur']})
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(729))
        model.forcing_vars = list(var_lags)
        model.description = "km729 Linear model with Swdown, Tair, RelHum, Wind, and Lagged Rainf (2d)"

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

    # Stupid mean models
    elif name == 'STH_km27_mean':
        model = km_regression(27, Mean())
        model.forcing_vars = ['SWdown', 'Tair', 'RelHum']
        model.description = "km27 mean model with Swdown, Tair, RelHum"
    elif name == 'STH_km243_mean':
        model = km_regression(243, Mean())
        model.forcing_vars = ['SWdown', 'Tair', 'RelHum']
        model.description = "km243 mean model with Swdown, Tair, RelHum"
    elif name == 'STH_km729_mean':
        model = km_regression(729, Mean())
        model.forcing_vars = ['SWdown', 'Tair', 'RelHum']
        model.description = "km729 mean model with Swdown, Tair, RelHum"
    elif name == 'STH_km2187_mean':
        model = km_regression(2187, Mean())
        model.forcing_vars = ['SWdown', 'Tair', 'RelHum']
        model.description = "km2187 mean model with Swdown, Tair, RelHum"

    else:
        raise NameError("unknown model")

    return model
