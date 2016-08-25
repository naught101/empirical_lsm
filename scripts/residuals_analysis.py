#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: residuals_analysis.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: TODO: File description

Usage:
    residuals_analysis.py
    residuals_analysis.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
"""

from docopt import docopt

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
            columns=lags)
    return pd.concat(data, axis=1)


def time_fmt(t):
    if 'min' in t:
        return "000.00:{m:02}".format(m=int(t.rstrip('min')))
    elif 'h' in t:
        return "000.{h:02}:00".format(h=int(t.rstrip('h')))
    elif 'd' in t:
        return "{d:03}.00:00".format(d=int(t.rstrip('d')))


def threekm27_residuals():
    """Save 3km27 residuals and met to a csv."""

    met_vars = ['SWdown', 'Tair', 'RelHum', 'Wind', 'Rainf']
    flux_vars = ['Qle', 'Qh']

    Tumba_met = pud.get_met_df(['Tumba'], met_vars, qc=True)
    Tumba_flux = pud.get_flux_df(['Tumba'], flux_vars, qc=True)
    threekm27 = (pud.get_pals_benchmark('3km27', 'Tumba')
                    .sel(x=1, y=1)
                    .to_dataframe()[['Qle', 'Qh']])

    lagged_met = get_lagged_df(Tumba_met)
    lagged_met.columns = ["{v}_{t}".format(v=c[0], t=time_fmt(c[1])) for c in lagged_met.columns.values]

    lagged_flux = get_lagged_df(Tumba_flux)
    lagged_flux.columns = ["{v}_{t}".format(v=c[0], t=time_fmt(c[1])) for c in lagged_flux.columns.values]

    residuals = (threekm27-Tumba_flux).reset_index(drop=True)
    out_df = pd.concat([residuals, lagged_flux, lagged_met], axis=1)

    return out_df


def plot_stuff():
    """Plots some stuff, you know?  """
    out_df = threekm27_residuals()

    # out_df.dropna().to_csv('Tumba3km27residuals_lagged.csv')

    y_vars = ['Qh', 'Qle']
    x_vars = list(out_df.columns)
    x_vars.remove('Qh')
    x_vars.remove('Qle')

    for y in y_vars:
        for x in x_vars:
            try:
                out_df[[x, y]].dropna().plot.scatter(x, y, s=3, alpha=0.5, edgecolors='face')
                plt.savefig('plots/lag_scatter_plots/{y}_by_{x}_scatter.png'.format(x=x, y=y))
                print('scatter', y, x)
                plt.close()
            except Exception as e:
                print('Warning: hexbin for', y, x, 'failed:', e)

    for y in y_vars:
        for x in x_vars:
            try:
                out_df[[x, y]].dropna().plot.hexbin(x, y, bins='log')
                plt.savefig('plots/lag_hexbins/{y}_by_{x}_hexbin.png'.format(x=x, y=y))
                print('hexbin', y, x)
                plt.close()
            except Exception as e:
                print('Warning: hexbin for', y, x, 'failed:', e)


def main(args):

    plot_stuff()

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
