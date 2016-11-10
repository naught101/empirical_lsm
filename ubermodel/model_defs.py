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
    elif name == 'STH_km243_lR30min':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['30min']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (30min)"
    elif name == 'STH_km243_lR1h':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['1h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (1h)"
    elif name == 'STH_km243_lR2h':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2h)"
    elif name == 'STH_km243_lR6h':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['6h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (6h)"
    elif name == 'STH_km243_lR12h':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['12h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (12h)"
    elif name == 'STH_km243_lR2d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2d)"
    elif name == 'STH_km243_lR10d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['10d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (10d)"
    elif name == 'STH_km243_lR30d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (30d)"
    elif name == 'STH_km243_lR60d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['60d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (60d)"
    elif name == 'STH_km243_lR90d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['90d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (90d)"
    elif name == 'STH_km243_lR180d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['180d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (180d)"
    elif name == 'STH_km243_lR2d30d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2d', '30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2d,30d)"

    # 3km243 with lagged Wind
    elif name == 'STH_km243_lW30min':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['30min']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (30min)"
    elif name == 'STH_km243_lW1h':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['1h']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (1h)"
    elif name == 'STH_km243_lW2d':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (2d)"
    elif name == 'STH_km243_lW10d':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['10d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (10d)"
    elif name == 'STH_km243_lW30d':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (30d)"
    elif name == 'STH_km243_lW180d':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['180d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (180d)"
    elif name == 'STH_km243_lW2d30d':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['2d', '30d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Lagged Wind (2d,30d)"

    # Building the Heirarchy
    elif name == 'STHR_km243_lR2d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['cur', '2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, Rainf, and Lagged Rainf (2d)"
    elif name == 'STHW_km243_lR2d':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['cur']})
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, Wind, and Lagged Rainf (2d)"
    elif name == 'STH_km729_lR2d':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(729))
        model.forcing_vars = list(var_lags)
        model.description = "km729 Linear model with Swdown, Tair, RelHum, and Lagged Rainf (2d)"

    # Lagged and non-lagged rainfall
    elif name == 'STHR_km243_lR':
        var_lags = cur_3_var()
        var_lags.update({'Rainf': ['cur', '2d']})
        model = LagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(var_lags)
        model.description = "km243 Linear model with Swdown, Tair, RelHum, Rainf, and Lagged Rainf (2d)"

    # Wind plus lagged Rainf
    elif name == 'STHW_km729_lR2d':
        var_lags = cur_3_var()
        var_lags.update({'Wind': ['cur']})
        var_lags.update({'Rainf': ['2d']})
        model = LagAverageWrapper(var_lags, km_lin(729))
        model.forcing_vars = list(var_lags)
        model.description = "km729 Linear model with Swdown, Tair, RelHum, Wind, and Lagged Rainf (2d)"

    # Markov-lagged Qle variants (doesn't seem to be working very well)
    elif name == 'STH_km243_lQle30min':
        var_lags = cur_3_var()
        var_lags.update({'Qle': ['30min']})
        model = MarkovLagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(['SWdown', 'Tair', 'RelHum'])
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Markov-Lagged Qle (30min)"
    elif name == 'STH_km243_lQle1h':
        var_lags = cur_3_var()
        var_lags.update({'Qle': ['1h']})
        model = MarkovLagAverageWrapper(var_lags, km_lin(243))
        model.forcing_vars = list(['SWdown', 'Tair', 'RelHum'])
        model.description = "km243 Linear model with Swdown, Tair, RelHum, and Markov-Lagged Qle (1h)"
    elif name == 'STH_km243_lQle2d':
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
