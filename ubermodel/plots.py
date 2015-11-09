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
import numpy as np

from dateutil.parser import parse

from pals_utils.data import pals_site_name, pals_xray_to_df, get_pals_benchmark, MissingDataError
from pals_utils.constants import FLUX_VARS, DATASETS

from .utils import print_bad, print_warn


def save_plot(base_path, rel_path, filename):
    """Save a figure and return the relative path (for rst)

    :base_path: path to directory with final RST
    :rel_path: relative path to figure directory
    :filename: plot filename
    :returns: rel_path/filename.png for RST

    """
    plot_path = os.path.join(base_path, rel_path, filename)
    pl.savefig(plot_path)
    pl.close()

    return os.path.join(rel_path, filename)


def diagnostic_plots(sim_data, flux_data, name):
    """Plot standard diagnostic plots for a single site

    :sim_data: TODO
    :flux_data: TODO
    :name: TODO
    :returns: TODO

    """
    site = pals_site_name(flux_data)

    print('Running standard plots for %s at %s' % (name, site))

    base_path = 'source/models/{n}'.format(n=name)
    rel_path = 'figures/{s}'.format(s=site)
    fig_path = os.path.join(base_path, rel_path)
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    files = []

    benchmark_names = ['1lin', '2lin', '3km27']
    benchmarks = [get_pals_benchmark(bname, site) for bname in benchmark_names]

    sns.set_palette(sns.color_palette(['red', 'pink', 'orange', 'black', 'blue']))

    # Generalise if more multi-variable plots needed
    for plot in [plot_PLUMBER_sim_metrics]:
        filename = plot(name, site)
        rel_plot_path = save_plot(base_path, rel_path, filename)
        files.append(rel_plot_path)

    for var in FLUX_VARS:
        try:
            data = pd.concat([pals_xray_to_df(ds, [var]) for ds in
                              benchmarks + [flux_data, sim_data]], axis=1)
        except MissingDataError:
            print_warn('Data missing for {v} at {s}, skipping.'.format(v=var, s=site))
            continue

        data.columns = benchmark_names + ['observed', 'modelled']

        for plot in DIAGNOSTIC_PLOTS:
            filename = plot(data, name, var, site)
            rel_plot_path = save_plot(base_path, rel_path, filename)
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

def get_PLUMBER_plot(model_dir):
    """generate PLUMBER plot and get filename

    :name: model name
    :returns: TODO

    """
    name = model_dir.replace('source/models/', '')

    sns.set_palette(sns.color_palette(['red', 'pink', 'orange', 'black', 'blue']))

    filename = plot_PLUMBER_sim_metrics(name, 'all')
    if filename is None:
        return

    rel_path = save_plot('source/models', name + '/figures', filename)

    files = [rel_path]

    plots = '\n\n'.join([
        ".. image :: {file}".format(file=f) for f in files])

    return plots


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

    failures = []
    for s in sites:
        try:
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
        except Exception:
            failures.append(s)
            continue

    if len(failures) > 0:
        print('Skipped {l} sites: {f}'.format(l=len(failures), f=', '.join(failures)))

    if len(metric_df) == 0:
        print_bad('Failed to load any csv files for {n} at {s} - skipping plot.'.format(n=name, s=site))
        return

    metric_df = pd.concat(metric_df).reset_index(drop=True)

    metric_df.ix[metric_df['metric'] == 'corr', 'value'] = - metric_df.ix[metric_df['metric'] == 'corr', 'value']

    metric_df['rank'] = metric_df.groupby(['variable', 'metric', 'site'])['value'].rank()

    mean_df = metric_df.groupby(['variable', 'name'])['rank'].mean().reset_index()

    mean_df.pivot(index='variable', columns='name', values='rank').plot()
    pl.title('{n}: PLUMBER plot: all metrics at {s}'.format(n=name, s=site))

    filename = '{n}_{s}_PLUMBER_plot_all_metrics.png'.format(n=name, s=site)
    return filename


# Drought workshop plots

def plot_drydown(sim_data, flux_data, met_data, name, date_range):
    """Plot behaviour during dry-downs.

    Plots rainfall, as well as Qh and Qle for obs and simulations.

    :sim_data: xray dataset from a simulation
    :flux_data: xray dataset from a simulation
    :met_data: xray dataset from a simulation
    :name: model name
    :returns: plot filename
    """
    year_range = [parse(d).year for d in date_range]
    year_range[1] += 1
    year_range = ['%s-01-01' % d for d in year_range]

    site = pals_site_name(met_data)

    sns.set_palette(sns.color_palette(['#aa0000', '#ff4444', '#0000aa', '#4477ff']))

    # Plot rainfall in mm
    Rainf = (pals_xray_to_df(met_data.sel(time=slice(*year_range)), ['Rainf'])
             .resample('1W', how='sum') * 1000)

    obs = (pals_xray_to_df(flux_data.sel(time=slice(*year_range)), ['Qh', 'Qle'])
           .resample('1D'))

    sim = (pals_xray_to_df(sim_data.sel(time=slice(*year_range)), ['Qh', 'Qle'])
           .resample('1D'))

    x_vals = Rainf.index.to_pydatetime()

    fig, ax = pl.subplots()
    ax.bar(x_vals, Rainf.values, width=7, color='lightblue', linewidth=0)

    x_vals = obs.index.to_pydatetime()
    for c in obs:
        if c == 'Qle':
            offset = 0
        if c == 'Qh':
            offset = 100
        ax.plot(x_vals, pd.rolling_mean(obs[c], 14) + offset, label='Obs %s + %d' % (c, offset))
        ax.plot(x_vals, pd.rolling_mean(sim[c], 14) + offset, label='Sim %s + %d' % (c, offset))

    ax.axvspan(parse(date_range[0]), parse(date_range[1]), color='black', alpha=0.1, zorder=-100)

    pl.legend(loc=0)

    pl.title('{n}: drydown plot at {s}'.format(n=name, s=site))

    filename = '{n}_{s}_drydown_plot.png'.format(n=name, s=site)
    return filename


def plot_drydown_daily_cycles(sim_data, flux_data, met_data, name, date_range):
    """Plot daily cycle behaviour during dry-downs.

    Plots rainfall, as well as Qh and Qle for obs and simulations.

    :sim_data: xray dataset from a simulation
    :flux_data: xray dataset from a simulation
    :met_data: xray dataset from a simulation
    :name: model name
    :returns: plot filename
    """
    date_range = [np.datetime64(parse(d)) for d in date_range]
    del_t = np.timedelta64(7, 'D')
    first_cycle = date_range[0] + [-del_t, del_t]
    last_cycle = date_range[1] + [-del_t, del_t]

    site = pals_site_name(met_data)

    sns.set_palette(sns.color_palette(['#aa0000', '#ff4444', '#0000aa', '#4477ff']))

    pl.subplots(2, 1, sharey=True)

    periods = {0: 'start', 1: 'end'}

    for i, drange in enumerate([first_cycle, last_cycle]):
        obs_df = pals_xray_to_df(flux_data.sel(time=slice(*drange)), ['Qh', 'Qle'])
        obs = obs_df.groupby(obs_df.index.time).mean()

        sim_df = pals_xray_to_df(sim_data.sel(time=slice(*drange)), ['Qh', 'Qle'])
        sim = sim_df.groupby(sim_df.index.time).mean()

        x_vals = obs.index.values
        ax = pl.subplot(2, 1, i + 1)
        for c in obs:
            if c == 'Qle':
                offset = 0
            if c == 'Qh':
                offset = 100
            ax.plot(x_vals, pd.rolling_mean(obs[c], 14) + offset, label='Obs %s + %d' % (c, offset))
            ax.plot(x_vals, pd.rolling_mean(sim[c], 14) + offset, label='Sim %s + %d' % (c, offset))

        pl.title('{n}: daily cycles for {p} of drydown at {s}'.format(n=name, s=site, p=periods[i]))

        pl.legend(loc=0)

    filename = '{n}_{s}_drydown_daily_cycles_plot.png'.format(n=name, s=site)
    return filename
