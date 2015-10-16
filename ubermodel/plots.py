#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: plots.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description: diagnostic plots for evaluating models
"""

import matplotlib.pyplot as pl
import seaborn as sns
import pandas as pd
import os

from pals_utils.data import pals_site_name, pals_xray_to_df, FLUX_VARS


def diagnostic_plots(sim_data, flux_data, name):
    """Plot standard diagnostic plots

    :sim_data: TODO
    :flux_data: TODO
    :name: TODO
    :returns: TODO

    """
    site = pals_site_name(flux_data)

    fig_path = os.path.join('source', 'models', name, 'figures')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    files = []

    # TODO: For variables
    for var in FLUX_VARS:
        data = pd.concat([pals_xray_to_df(sim_data, [var]), pals_xray_to_df(flux_data, [var])], axis=1)
        data.columns = ['modelled', 'observed']

        for plot in PLOTS:
            filename = plot(data, name, var, site)
            plot_path = os.path.join(fig_path, filename)
            pl.savefig(plot_path)

            rel_plot_path = os.path.join('figures', filename)
            files.append(rel_plot_path)

    return files


def plot_weekly_timeseries(data, name, var, site):
    data.resample('1W').plot()
    pl.title('{0}: Weekly average {1} at {2}'.format(name, var, site))

    filename = '{0}_{1}_{2}_weekly_timeseries.png'.format(name, var, site)
    return filename


def plot_scatter(data, name, var, site):
    data.plot.scatter('observed', 'modelled', c='black', s=1, alpha=0.5)
    pl.title('{0}: Scatterplot of {1} at {2}'.format(name, var, site))

    filename = '{0}_{1}_{2}_scatterplot.png'.format(name, var, site)
    return filename


PLOTS = [plot_weekly_timeseries, plot_scatter]
