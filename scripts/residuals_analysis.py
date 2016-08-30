#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: residuals_analysis.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: TODO: File description

Usage:
    residuals_analysis.py <plot_type> <site> [<var>]
    residuals_analysis.py (-h | --help | --version)

Options:
    plot_type     "scatter" or "hexbin"
    site          name of a PALS site, or "all"
    var           name of a driving variable, or "all"
    -h, --help    Show this screen and exit.
"""

from docopt import docopt

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pals_utils import data as pud

from scripts.lag_average_assessment import rolling_mean


def get_lagged_df(df, lags=['30min', '2h', '6h', '2d', '7d', '30d', '90d']):
    """Get lagged variants of variables"""

    data = {}
    for v in df.columns:
        data[v] = pd.DataFrame(
            np.concatenate([rolling_mean(df[[v]].values, l) for l in lags], axis=1),
            columns=lags, index=df.index)
    return pd.concat(data, axis=1)


def time_fmt(t):
    if 'min' in t:
        return "000.00:{m:02}".format(m=int(t.rstrip('min')))
    elif 'h' in t:
        return "000.{h:02}:00".format(h=int(t.rstrip('h')))
    elif 'd' in t:
        return "{d:03}.00:00".format(d=int(t.rstrip('d')))


def threekm27_residuals(sites, var):
    """Save 3km27 residuals and met to a csv."""

    met_vars = ['SWdown', 'Tair', 'RelHum', 'Wind', 'Rainf']
    flux_vars = ['Qle', 'Qh']

    flux_df = (pud.get_flux_df(sites, flux_vars, name=True, qc=True)
                  .reorder_levels(['site', 'time'])
                  .sort_index())
    threekm27 = (pud.get_pals_benchmark_df('3km27', sites, ['Qle', 'Qh'])
                    .sort_index())
    residuals = (threekm27 - flux_df)
    residuals.columns = ['3km27 %s residual' % v for v in residuals.columns]

    if var in met_vars:
        forcing = (pud.get_met_df(sites, [var], name=True, qc=True)
                      .reorder_levels(['site', 'time'])
                      .sort_index())
    else:
        forcing = (pud.get_flux_df(sites, [var], name=True, qc=True)
                      .reorder_levels(['site', 'time'])
                      .sort_index())

    lagged_forcing = forcing.groupby(level='site').apply(get_lagged_df)

    lagged_forcing.columns = ["{v}_{t}_mean".format(v=c[0], t=time_fmt(c[1])) for c in lagged_forcing.columns.values]

    out_df = pd.concat([residuals, lagged_forcing], axis=1)

    return out_df


def scatter(df, x, y):
    return df[[x, y]].dropna().plot.scatter(x, y, s=3, alpha=0.5, edgecolors='face')


def hexbin(df, x, y):
    return df[[x, y]].dropna().plot.hexbin(x, y, bins='log')


def plot_stuff(plot_type, site, var):
    """Plots some stuff, you know?"""

    if site == 'all':
        sites = ["Amplero", "Blodgett", "Bugac", "ElSaler", "ElSaler2",
                 "Espirra", "FortPeck", "Harvard", "Hesse", "Howard", "Howlandm",
                 "Hyytiala", "Kruger", "Loobos", "Merbleue", "Mopane", "Palang",
                 "Sylvania", "Tumba", "UniMich"]
    else:
        sites = [site]

    if var == 'all':
        variables = ['Qle', 'Qh', 'SWdown', 'Tair', 'RelHum', 'Wind', 'Rainf']
    else:
        variables = [var]

    for var in variables:
        out_df = threekm27_residuals(sites, var)

        # out_df.dropna().to_csv('Tumba3km27residuals_lagged.csv')

        y_vars = ['3km27 %s residual' % v for v in ['Qle', 'Qle']]
        x_vars = list(set(out_df.columns).difference(y_vars))

        if plot_type == 'scatter':
            plot_fn = scatter
        if plot_type == 'hexbin':
            plot_fn = hexbin

        for y in y_vars:
            for x in x_vars:
                try:
                    plot_fn(out_df, x, y)
                    plt.title("{y} by {x} at site: {s}".format(x=x, y=y, s=site))
                    path = 'plots/lag_plots_{pt}'.format(pt=plot_type)
                    os.makedirs(path, exist_ok=True)
                    plt.savefig('{p}/{s}_{y}_by_{x}_{pt}.png'.format(s=site, x=x, y=y, pt=plot_type, p=path))
                    print(plot_type, y, x)
                    plt.close()
                except Exception as e:
                    print('Warning:', plot_type, 'for', y, x, 'failed:', e)


def main(args):

    plot_stuff(args['<plot_type>'], args['<site>'], args['<var>'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
