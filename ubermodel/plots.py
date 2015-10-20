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

from pals_utils.data import pals_site_name, pals_xray_to_df, get_pals_benchmark, MissingDataError
from pals_utils.constants import FLUX_VARS, DATASETS


def diagnostic_plots(sim_data, flux_data, name):
    """Plot standard diagnostic plots for a single site

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

    # Generalise if more multi-variable plots needed
    for plot in [plot_PLUMBER_sim_metrics]:
        filename = plot(name, site)
        plot_path = os.path.join(fig_path, filename)
        pl.savefig(plot_path)
        pl.close()

        rel_plot_path = 'figures/{s}/{f}'.format(s=site, f=filename)
        files.append(rel_plot_path)

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


#######################
# metric plots
#######################

def plot_PLUMBER_sim_metrics(name, site):
    """Plot metrics from a site, with benchmarks for comparison

    :returns: TODO
    """
    csv_file = './source/models/{n}/metrics/{n}_{s}_metrics.csv'

    benchmark_names = ['1lin', '2lin', '3km27']

    if site == 'all':
        sites = DATASETS
    else:
        sites = [site]

    metric_df = []

    for s in sites:
        site_metrics = pd.read_csv(csv_file.format(n=name, s=s))
        site_metrics = pd.melt(site_metrics, id_vars='metric')
        site_metrics['name'] = name
        site_metrics['site'] = s
        metric_df.append(site_metrics)

        for b in benchmark_names:
            benchmark_metrics = pd.read_csv(csv_file.format(n=b, s=s))
            benchmark_metrics = pd.melt(benchmark_metrics, id_vars='metric')
            benchmark_metrics['name'] = b
            benchmark_metrics['site'] = s
            metric_df.append(benchmark_metrics)

    metric_df = pd.concat(metric_df).reset_index(drop=True)

    metric_df.ix[metric_df['metric'] == 'corr', 'value'] = - metric_df.ix[metric_df['metric'] == 'corr', 'value']

    metric_df['rank'] = metric_df.groupby(['variable', 'metric', 'site'])['value'].rank()

    mean_df = metric_df.groupby(['variable', 'name'])['rank'].mean().reset_index()

    mean_df.pivot(index='variable', columns='name', values='rank').plot()
    pl.title('{n}: PLUMBER plot: all metrics at {s}'.format(n=name, s=site))

    filename = '{n}_{s}_PLUMBER_plot_all_metrics.png'.format(n=name, s=site)
    return filename
