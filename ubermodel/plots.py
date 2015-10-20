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
import warnings

from pals_utils.data import pals_site_name, pals_xray_to_df, get_pals_benchmark, FLUX_VARS, MissingDataError


def diagnostic_plots(sim_data, flux_data, name):
    """Plot standard diagnostic plots

    :sim_data: TODO
    :flux_data: TODO
    :name: TODO
    :returns: TODO

    """
    site = pals_site_name(flux_data)

    print('Running standard plots for %s at %s' % (name, site))

    fig_path = 'source/models/{n}/figures/{s}'.format(n=name, s=site)
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    files = []

    benchmark_names = ['1lin', '2lin', '3km27']
    benchmarks = [get_pals_benchmark(bname, site) for bname in benchmark_names]

    sns.set_palette(sns.color_palette(['red', 'pink', 'orange', 'black', 'blue']))

    # TODO: For variables
    for var in FLUX_VARS:
        try:
            data = pd.concat([pals_xray_to_df(ds, [var]) for ds in
                              benchmarks + [flux_data, sim_data]], axis=1)
        except MissingDataError:
            warnings.warn('Data missing for {v} at {s}, skipping.'.format(v=var, s=site))
            continue

        data.columns = benchmark_names + ['observed', 'modelled']

        for plot in DIAGNOSTIC_PLOTS:
            filename = plot(data, name, var, site)
            plot_path = os.path.join(fig_path, filename)
            pl.savefig(plot_path)
            pl.close()

            rel_plot_path = 'figures/{s}/{f}'.format(s=site, f=filename)
            files.append(rel_plot_path)

    return files


def plot_weekly_timeseries(data, name, var, site):
    data.resample('1W').plot()
    pl.title('{n}: Weekly average {v} at {s}'.format(n=name, v=var, s=site))

    filename = '{n}_{v}_{s}_weekly_timeseries.png'.format(n=name, v=var, s=site)
    return filename


def plot_scatter(data, name, var, site):
    data.plot.scatter('observed', 'modelled', c='black', s=1, alpha=0.5)
    pl.title('{n}: Scatterplot of {v} at {s}'.format(n=name, v=var, s=site))

    filename = '{n}_{v}_{s}_scatterplot.png'.format(n=name, v=var, s=site)
    return filename


def plot_annual_cycle(data, name, var, site):
    data.groupby(data.index.month).mean().plot()
    pl.title('{n}: Annual average {v} cycle at {s}'.format(n=name, v=var, s=site))

    filename = '{n}_{v}_{s}_annual_cycle.png'.format(n=name, v=var, s=site)
    return filename


def plot_daily_cycle(data, name, var, site):
    data.groupby(data.index.time).mean().plot()
    pl.title('{n}: daily average {v} cycle at {s}'.format(n=name, v=var, s=site))

    filename = '{n}_{v}_{s}_daily_cycle.png'.format(n=name, v=var, s=site)
    return filename


def plot_qq_plot(data, name, var, site):
    """qqplot!
    """
    data.apply(sorted).plot.scatter('observed', 'modelled', c='black', s=1, alpha=0.5)
    pl.title('{n}: Quantile-quantile plot for {v} at {s}'.format(n=name, v=var, s=site))

    filename = '{n}_{v}_{s}_qq_plot.png'.format(n=name, v=var, s=site)
    return filename


def plot_residuals(data, name, var, site):
    """Residual errors plot
    """
    data['residuals'] = data['modelled'] - data['observed']
    data.plot.scatter('observed', 'residuals', c='black', s=1, alpha=0.5)

    pl.title('{n}: residual plot for {v} at {s}'.format(n=name, v=var, s=site))

    filename = '{n}_{v}_{s}_residual_plot.png'.format(n=name, v=var, s=site)
    return filename


DIAGNOSTIC_PLOTS = [plot_weekly_timeseries,
         plot_scatter,
         plot_annual_cycle,
         plot_daily_cycle,
         plot_qq_plot,
         plot_residuals]
