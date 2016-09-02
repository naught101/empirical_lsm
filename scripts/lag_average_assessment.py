#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: lag_average_assessment.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: assess lagg structure of flux/met data

Usage:
    lag_average_assessment.py self_lag <var> <metric>
    lag_average_assessment.py xy_lag <var_y> <var_x> <metric>
    lag_average_assessment.py (-h | --help | --version)

Options:
    metric        [corr|MI]
    -h, --help    Show this screen and exit.
"""

from docopt import docopt

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

import pals_utils.data as pud
from ubermodel.data import get_sites
from ubermodel.transforms import rolling_mean, get_lags

from mutual_info.mutual_info import mutual_information_2d


def get_data(sites, var, qc=True):
    """load arbitrary data """
    if var in ['SWdown', 'LWdown', 'Tair', 'RelHum', 'Qair', 'Wind', 'Rainf']:
        data = pud.get_met_df(sites, [var], qc=qc)
    else:
        data = pud.get_flux_df(sites, [var], qc=qc)

    return data


def metric_complete(func, vec1, vec2):
    """Get the mutual information of two vectors, dropping NAs."""
    index = np.isfinite(vec1) & np.isfinite(vec2)
    if index.sum() == 0:
        return np.nan
    return func(vec1[index], vec2[index])


def plot_self_lag(var, metric, sites):
    """Plots a variable's metric against moving window averages of varying lengths of itself."""
    data = get_data(sites, var)

    lags = get_lags()

    lagged_data = pd.DataFrame(np.concatenate([rolling_mean(data[[var]].values, l) for l in lags], axis=1), columns=lags)

    if metric == 'corr':
        image_data = lagged_data.corr()
    elif metric == 'MI':
        image_data = pd.DataFrame(np.nan, index=lags, columns=lags)
        for i in lags:
            for j in lags:
                image_data.ix[i, j] = metric_complete(mutual_information_2d, lagged_data[i], lagged_data[j])
    else:
        assert False, "metric {m} not implemented".format(m=metric)

    fig = plt.imshow(image_data, interpolation='none')
    fig.axes.set_xticks(range(len(lagged_data.columns)))
    fig.axes.set_yticks(range(len(lagged_data.columns)))
    fig.axes.set_xticklabels(lagged_data.columns)
    fig.axes.set_yticklabels(lagged_data.columns)
    plt.colorbar()
    plt.title("{v} lagged averages' {m}".format(v=var, m=metric))

    os.makedirs("plots/self_lags", exist_ok=True)
    filename = 'plots/self_lags/{v}_lagged_avg_{m}.png'.format(v=var, m=metric)
    print("Saving to {fn}".format(fn=filename))
    plt.savefig(filename)


def plot_xy_lag(var_y, var_x, metric, sites):
    """Plots correlations between a variable and lagged versions of another variable"""

    lags = get_lags()

    plot_data = pd.DataFrame(np.nan, index=lags, columns=sites)

    if metric == 'corr':
        def func(x, y):
            return np.corrcoef(x, y)[0, 1]
    elif metric == 'MI':
        func = mutual_information_2d
    else:
        assert False, "metric {m} not implemented".format(m=metric)

    for site in sites:
        try:
            data_y = get_data([site], var_y)[var_y].reset_index(drop=True)
            data_x = get_data([site], var_x)
        except RuntimeError:
            print("Loading data for {s} failed, skipping".format(s=site))
            continue

        lagged_x = pd.DataFrame(np.concatenate([rolling_mean(data_x[[var_x]].values, l) for l in lags], axis=1), columns=lags)

        for l in lags:

            plot_data.ix[l, site] = metric_complete(func, lagged_x[l], data_y)

    ax = plot_data.plot()
    ax.set_prop_cycle("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, len(sites))])

    plt.title("{vy} {m} with lagged averages of {vx}".format(vx=var_x, vy=var_y, m=metric))
    plt.legend(ncol=4, fontsize='xx-small')

    path = "plots/xy_lags/{m}".format(m=metric)
    os.makedirs(path, exist_ok=True)
    filename = '{p}/{vy}_lagged_avg_{vx}_{m}.png'.format(p=path, vx=var_x, vy=var_y, m=metric)
    print("Saving to {fn}".format(fn=filename))
    plt.savefig(filename)


def main(args):

    if args['self_lag']:
        sites = ['Tumba']
        plot_self_lag(args['<var>'], args['<metric>'], sites)
    elif args['xy_lag']:
        sites = get_sites('all')
        plot_xy_lag(args['<var_y>'], args['<var_x>'], args['<metric>'], sites)


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
