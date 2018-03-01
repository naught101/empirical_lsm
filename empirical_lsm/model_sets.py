#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model_sets.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/empirical_lsm
Description: Gets named sets of models, for use in various scripts
"""

from itertools import chain, combinations

import logging
logger = logging.getLogger(__name__)


def powerset(iterable, start=0):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(start, len(s) + 1))


def get_combo_model_names():

    opts = ['W', 'dT', 'dQ', '_lS30d', '_lR30d', '_lH10d', '_lT6h']
    res = ['243', '729']
    # res = ['2187']

    combos = powerset(opts, 0)
    names = []
    for c in combos:
        for r in res:
            names += ['STH' + ''.join(c) + '_km' + r]

    return names


def get_model_set(model_set=None):
    """Gets pre-defined model sets
    """

    lags = ['30min', '1h', '2h', '6h', '12h', '1d', '2d', '7d', '10d', '30d', '60d', '90d', '180d']
    lag_fmts = ['STH_l%%s%s_km243' % l for l in lags]

    base_models = ['S_lin', 'ST_lin', 'STH_km27', 'STH_km243']

    model_sets = {
        "Added variables": base_models + [
            'STHL_km27', 'STHW_km27', 'STHR_km27',
            'STHL_km243', 'STHW_km243', 'STHR_km243'],

        "Increasing articulation": [
            'S_lin', 'ST_lin', 'STH_km27',
            'STH_km243', 'STH_km729', 'STH_km2187'],

        "Lagged SWdown": base_models + [f % 'S' for f in lag_fmts],
        "Lagged Tair":   base_models + [f % 'T' for f in lag_fmts],
        "Lagged RelHum": base_models + [f % 'H' for f in lag_fmts],
        "Lagged LWdown": base_models + ['STHL_km243'] +
                            [f % 'L' for f in lag_fmts],
        "Lagged Rainf":  base_models + ['STHR_km243'] +
                            [f % 'R' for f in lag_fmts],
        "Lagged Wind":   base_models + ['STHW_km243'] +
                            [f % 'W' for f in lag_fmts],

        "Final ensemble": ["S_lin", "ST_lin", "STH_km27",
                           "STH_km243", "STHW_km243",
                           "short_term243",  # "STHWdTdQ_lT6hM_km243",
                           "long_term243",  # "STHWdTdQ_lS30d_lR30d_lH10d_lT6hM_km243",
                           "long_term729"],  # "STHWdTdQ_lS30d_lR30d_lH10d_lT6hM_km729"],

        "Trafficlights ensemble": ["S_lin", "ST_lin", "STH_km27",
                                   "STH_km81", "STHW_km81",
                                   "STH_km243", "STHW_km243",
                                   "short_term243",  # "STHWdTdQ_lT6hM_km243",
                                   "long_term243",  # "STHWdTdQ_lS30d_lR30d_lH10d_lT6hM_km243",
                                   "long_term729"],  # "STHWdTdQ_lS30d_lR30d_lH10d_lT6hM_km729"],

        "Combo models": ["S_lin", "ST_lin", "STH_km27"] + get_combo_model_names()
    }
    model_sets['all_paper_models'] = list(set(
        model_sets["Added variables"] +
        model_sets["Increasing articulation"] +
        model_sets["Lagged SWdown"] +
        model_sets["Lagged Tair"] +
        model_sets["Lagged RelHum"] +
        model_sets["Lagged LWdown"] +
        model_sets["Lagged Rainf"] +
        model_sets["Lagged Wind"] +
        model_sets["Final ensemble"]))

    if model_set is None:
        return model_sets
    else:
        return model_sets[model_set]
