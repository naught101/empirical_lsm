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
import xray as xr

from dateutil.parser import parse

from pals_utils.data import pals_site_name, pals_xr_to_df
from pals_utils.constants import FLUX_VARS

from .utils import print_bad, print_warn
from .data import get_sites


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


def get_benchmark(name, site):
    """returns an xarray dataset

    :name: mode name
    :site: fluxnet site name
    :returns: xarray dataset

    """
    file_path = 'model_data/{name}/{name}_{site}.nc'

    benchmark = xr.open_dataset(file_path.format(name=name, site=site))

    return benchmark


def diagnostic_plots(sim_data, flux_data, name):
    """Plot standard diagnostic plots for a single site

    :sim_data: xarray dataset
    :flux_data: xarray dataset
    :name: mode name
    :returns: list of paths to plots

    """
    site = pals_site_name(flux_data)

    print('Running standard plots for %s at %s' % (name, site))

    base_path = 'source/models/{n}'.format(n=name)
    rel_path = 'figures/{s}'.format(s=site)
    fig_path = os.path.join(base_path, rel_path)
    os.makedirs(fig_path, exist_ok=True)

    files = []

    # benchmark_names = ['1lin', '2lin', '3km27']
    benchmark_names = ['S_lin', 'ST_lin', 'STH_km27_lin']

    try:
        benchmarks = [get_benchmark(bname, site) for bname in benchmark_names]
    except RuntimeError as e:
        print_warn("Benchmark(s) not available at {s}, skipping. {e}".format(s=site, e=e))
        return

    sns.set_palette(sns.color_palette(['red', 'pink', 'orange', 'black', 'blue']))

    # Generalise if more multi-variable plots needed
    metric_df = get_PLUMBER_metrics(name, site)
    for plot in [p_plumber_metrics, p_metric_rank_counts]:
        filename = plot(metric_df, name, site)
        rel_plot_path = save_plot(base_path, rel_path, filename)
        files.append(rel_plot_path)

    for var in FLUX_VARS:
        try:
            data = pd.concat([pals_xr_to_df(ds, [var]) for ds in
                              benchmarks + [flux_data, sim_data]], axis=1)
        except Exception as e:
            print_warn('Data missing for {v} at {s}, skipping. {e}'.format(v=var, s=site, e=e))
            continue

        data.columns = benchmark_names + ['observed', 'modelled']

        for plot in DIAGNOSTIC_PLOTS.values():
            filename = plot(data, name, var, site)
            rel_plot_path = save_plot(base_path, rel_path, filename)
            files.append(rel_plot_path)

    return files


def plot_weekly_timeseries(data, name, var, site):
    data.resample('1W').mean().plot()
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
    residuals = data[['observed']].copy()
    residuals['residuals'] = data['modelled'] - data['observed']
    residuals.plot.scatter('observed', 'residuals', c='black', s=1, alpha=0.5)

    pl.title('{n}: residual plot for {v} at {s}'.format(n=name, v=var, s=site))

    filename = '{n}_{v}_{s}_residual_plot.png'.format(n=name, v=var, s=site)
    return filename


DIAGNOSTIC_PLOTS = {
    "source/models/{n}/figures/{s}/{n}_{v}_{s}_annual_cycle.png": plot_annual_cycle,
    "source/models/{n}/figures/{s}/{n}_{v}_{s}_daily_cycle.png": plot_daily_cycle,
    "source/models/{n}/figures/{s}/{n}_{v}_{s}_qq_plot.png": plot_qq_plot,
    "source/models/{n}/figures/{s}/{n}_{v}_{s}_residual_plot.png": plot_residuals,
    "source/models/{n}/figures/{s}/{n}_{v}_{s}_scatterplot.png": plot_scatter,
    "source/models/{n}/figures/{s}/{n}_{v}_{s}_weekly_timeseries.png": plot_weekly_timeseries,
}


#######################
# metric plots
#######################

def get_PLUMBER_plot(model_dir, site='all'):
    """generate PLUMBER plot and get filename

    :name: model name
    """
    name = model_dir.replace('source/models/', '')

    sns.set_palette(sns.color_palette(['red', 'pink', 'orange', 'black', 'blue']))

    metric_df = get_PLUMBER_metrics(name, site)

    for metrics in ['all', 'standard', 'distribution']:
        filename = p_plumber_metrics(metric_df, name, site, metrics)
        save_plot('source/models', name + '/figures', filename)
        filename = p_metric_rank_counts(metric_df, name, site, metrics)
        save_plot('source/models', name + '/figures', filename)


def plot_PLUMBER_sim_metrics(name, site, metrics='all'):
    """Plot metrics from a site, with benchmarks for comparison

    :name: model name
    :site: fluxnet site name
    :returns: path to plotted file
    """
    metric_df = get_PLUMBER_metrics(name, site)

    filename = p_plumber_metrics(metric_df, name, site, metrics)

    return filename


def get_PLUMBER_metrics(name, site='all', variables=['Qle', 'Qh', 'NEE']):
    """get metrics dataframe from a site, with benchmarks for comparison

    :returns: dataframe with metrics for model at site
    """
    csv_file = './source/models/{n}/metrics/{n}_{s}_metrics.csv'

    # benchmark_names = ['1lin', '2lin', '3km27']
    benchmark_names = ['S_lin', 'ST_lin', 'STH_km27_lin']

    if site == 'all':
        sites = get_sites('PLUMBER_ext')
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
            metric_df.append(site_metrics[site_metrics.variable.isin(variables)])

            for b in benchmark_names:
                benchmark_metrics = pd.read_csv(csv_file.format(n=b, s=s))
                benchmark_metrics = pd.melt(benchmark_metrics, id_vars='metric')
                benchmark_metrics['name'] = b
                benchmark_metrics['site'] = s
                metric_df.append(benchmark_metrics[benchmark_metrics.variable.isin(variables)])
        except Exception:
            failures.append(s)
            continue

    if len(failures) > 0:
        print('Skipped {l} sites: {f}'.format(l=len(failures), f=', '.join(failures)))

    if len(metric_df) == 0:
        print_bad('Failed to load any csv files for {n} at {s} - skipping plot.'.format(n=name, s=site))
        return

    metric_df = pd.concat(metric_df).reset_index(drop=True)

    # invert 1-centred metrics
    one_metrics = ['corr', 'overlap']
    metric_df.ix[metric_df['metric'].isin(one_metrics), 'value'] = 1 - metric_df.ix[metric_df['metric'].isin(one_metrics), 'value']

    metric_df['rank'] = metric_df.groupby(['variable', 'metric', 'site'])['value'].apply(lambda x: x.abs().rank())

    # and reinvert
    metric_df.ix[metric_df['metric'].isin(one_metrics), 'value'] = 1 - metric_df.ix[metric_df['metric'].isin(one_metrics), 'value']

    return metric_df


def subset_metric_df(metric_df, metrics):
    """Return only the metrics required
    """
    if metrics == 'standard':
        metrics_list = ['nme', 'mbe', 'sd_diff', 'corr']
        return metric_df[metric_df.metric.isin(metrics_list)].copy()
    elif metrics == 'distribution':
        metrics_list = ['extreme_5', 'extreme_95', 'skewness', 'kurtosis', 'overlap']
        return metric_df[metric_df.metric.isin(metrics_list)].copy()
    else:
        return metric_df


def p_plumber_metrics(metric_df, name, site='all', metrics='all'):
    """Plot metric results as averages over site and metric

    :metric_df: pandas dataframe of results
    :name: site name for title
    :metrics: metrics to include: 'all', 'standard', 'distribution'
    :returns: plotted filename

    """
    models = ['S_lin', 'ST_lin', 'STH_km27_lin', name]

    metric_df = subset_metric_df(metric_df, metrics)

    mean_df = metric_df.groupby(['variable', 'name'])['rank'].mean().reset_index()

    mean_df_wide = mean_df.pivot(index='variable', columns='name', values='rank')

    mean_df_wide[models].plot()
    pl.title('{n}: PLUMBER plot: {m} metrics at {s}'.format(n=name, s=site, m=metrics))

    filename = '{n}_{s}_PLUMBER_plot_{m}_metrics.png'.format(n=name, s=site, m=metrics)

    return filename


def p_metric_rank_counts(metric_df, name, site='all', metrics='all'):
    """plots hostograms of ranks for each variable and model
    """
    models = ['S_lin', 'ST_lin', 'STH_km27_lin', name]
    if name not in models:
        models.append(name)

    metric_df = subset_metric_df(metric_df, metrics)

    metric_df['name'] = pd.Categorical(metric_df['name'], models)
    metric_df.sort_values('name', inplace=True)

    count_df = (metric_df[['rank', 'name', 'variable', 'value']]
                .groupby(['rank', 'variable', 'name'])
                .agg('count')
                .fillna(0)
                .reset_index()
                .rename(columns={'value': 'count'}))

    sns.factorplot(y="rank", x="count", col="variable", hue="name", data=count_df, orient='h')
    pl.gca().invert_yaxis()

    pl.suptitle('{n}: Rank counts: {m} metrics at {s}'.format(n=name, s=site, m=metrics))

    filename = '{n}_{s}_rank_counts_{m}_metrics.png'.format(n=name, s=site, m=metrics)

    return filename


def p_metric_normalised_violins(metric_df, name, site='all', metrics='all'):
    """plots violins of metrics normalised for each variable and model
    """
    models = ['S_lin', 'ST_lin', 'STH_km27_lin', name]

    metric_df = subset_metric_df(metric_df, metrics)

    metric_df['name'] = pd.Categorical(metric_df['name'], models)
    metric_df.sort_values('name', inplace=True)

    one_metrics = ['corr', 'overlap']
    metric_df.ix[metric_df['metric'].isin(one_metrics), 'value'] = 1 - metric_df.ix[metric_df['metric'].isin(one_metrics), 'value']

    metric_df['value'] = (metric_df
                          .groupby(['site', 'variable', 'metric'])['value']
                          .apply(lambda x: (x - x.min()) / (x.max() - x.min())))

    fg = sns.factorplot(y="value", x="variable", hue="name", data=metric_df, orient='v', kind='violin', bw=0.1)
    sns.factorplot(y="value", x="variable", hue="name", data=metric_df, orient='v', kind='point', ax=fg.ax, ci=None)
    fg.ax.legend().set_visible(False)

    pl.suptitle('{n}: Minmax normalised metrics: {m} metrics at {s}'.format(n=name, s=site, m=metrics))

    filename = '{n}_{s}_minmax_normalised_{m}_metrics.png'.format(n=name, s=site, m=metrics)

    return filename


PLUMBER_PLOTS = {
    "source/models/{n}/figures/{n}_all_PLUMBER_plot_{m}_metrics.png": get_PLUMBER_plot,
    "source/models/{n}/figures/{s}/{n}_{s}_PLUMBER_plot_{m}_metrics.png": plot_PLUMBER_sim_metrics,
}


# Drought workshop plots

def plot_drydown(sim_data, flux_data, met_data, name, date_range):
    """Plot behaviour during dry-downs.

    Plots rainfall, as well as Qh and Qle for obs and simulations.

    :sim_data: xarray dataset from a simulation
    :flux_data: xarray dataset from a simulation
    :met_data: xarray dataset from a simulation
    :name: model name
    :returns: plot filename
    """
    year_range = [parse(d).year for d in date_range]
    year_range[1] += 1
    year_range = ['%s-01-01' % d for d in year_range]

    site = pals_site_name(met_data)

    sns.set_palette(sns.color_palette(['#aa0000', '#ff4444', '#0000aa', '#4477ff']))

    # Plot rainfall in mm
    Rainf = (pals_xr_to_df(met_data.sel(time=slice(*year_range)), ['Rainf'])
             .resample('1W', how='sum') * 1000).mean()

    obs = (pals_xr_to_df(flux_data.sel(time=slice(*year_range)), ['Qh', 'Qle'])
           .resample('1D')).mean()

    sim = (pals_xr_to_df(sim_data.sel(time=slice(*year_range)), ['Qh', 'Qle'])
           .resample('1D')).mean()

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

    filename = '{n}_{s}_drydown_timeseries_plot.png'.format(n=name, s=site)
    return filename


def plot_drydown_daily_cycles(sim_data, flux_data, met_data, name, date_range):
    """Plot daily cycle behaviour during dry-downs.

    Plots rainfall, as well as Qh and Qle for obs and simulations.

    :sim_data: xarray dataset from a simulation
    :flux_data: xarray dataset from a simulation
    :met_data: xarray dataset from a simulation
    :name: model name
    :returns: plot filename
    """
    d_range = [np.datetime64(parse(d)) for d in date_range]
    del_t = np.timedelta64(7, 'D')
    first_cycle = d_range[0] + [-del_t, del_t]
    last_cycle = d_range[1] + [-del_t, del_t]

    site = pals_site_name(met_data)

    sns.set_palette(sns.color_palette(['#aa0000', '#ff4444', '#0000aa', '#4477ff']))

    fig, _ = pl.subplots(2, 1, sharey=True)

    periods = {0: 'start', 1: 'end'}

    flux_vars = list(set(['Qh', 'Qle']).intersection(list(sim_data.data_vars))
                                       .intersection(list(sim_data.data_vars)))

    for i, dr in enumerate([first_cycle, last_cycle]):
        obs_df = pals_xr_to_df(flux_data.sel(time=slice(*dr)), flux_vars)
        obs = obs_df.groupby(obs_df.index.time).mean()

        sim_df = pals_xr_to_df(sim_data.sel(time=slice(*dr)), flux_vars)
        sim = sim_df.groupby(sim_df.index.time).mean()

        x_vals = obs.index.values
        ax = pl.subplot(2, 1, i + 1)
        for c in obs:
            if c == 'Qle':
                offset = 0
            if c == 'Qh':
                offset = 100
            ax.plot(x_vals, obs[c] + offset, label='Obs %s + %d' % (c, offset))
            ax.plot(x_vals, sim[c] + offset, label='Sim %s + %d' % (c, offset))

        pl.title('{n}: daily cycles for {p} of drydown at {s}'.format(n=name, s=site, p=periods[i]))

        pl.legend(loc=0)

    fig.tight_layout()

    filename = '{n}_{s}_drydown_daily_cycles_plot.png'.format(n=name, s=site)
    return filename

DRYDOWN_PLOTS = {
    "source/models/{n}/figures/{s}/{n}_{s}_drydown_daily_cycles_plot.png": plot_drydown_daily_cycles,
    "source/models/{n}/figures/{s}/{n}_{s}_drydown_timeseries_plot.png": plot_drydown,
}


# PLUMBER replacement: Parallel coordinate plots
def p_parallel_coord(df):
    """Parallel coordinate plot, replacement for PLUMBER plots
    """
    required = ['site', 'model', 'variable', 'metric', 'value']
    missing = [c for c in required if c not in df.columns]
    assert len(missing), "Dataframe is missing variables: {0}".format(missing)

    df['value'] = (df.groupby(['variable', 'metric'])['value']
                     .apply(quantile_normalise))

    metrics = df.metric.unique()
    variables = df.variable.unique()
    models = df.name.unique()
    sites = df.site.unique()

    col_idx = list(range(len(metrics)))
    colours = ['k', 'r', 'pink', 'orange']

    fig, axes = pl.subplots(len(variables), 1)
    for v, var in enumerate(variables):
        ax = axes[v]
        for m, mod in enumerate(models):
            # for s, site in enumerate(sites):
            mat = (df.ix[(df.variable == var) & (df.name == mod), ['site', 'metric', 'value']]
                     .pivot(index='site', columns='metric', values='value')[metrics])
            for site in sites:
                if site in mat.index:
                    ax.plot(col_idx, mat.loc[site], c=colours[m], alpha=0.5, label=mod)
        ax.xaxis.set_ticklabels(metrics)
        ax.yaxis.set_label_text(var)

    labels = ax.get_legend_handles_labels()
    unique_labels = {v: k for k, v in {v: k for k, v in dict(zip(*labels)).items()}.items()}
    fig.legend(unique_labels.keys(), unique_labels.values())


# PLUMBER replacement: Parallel coordinate plots
def p_parallel_coord_summary(df):
    """Parallel coordinate plot, replacement for PLUMBER plots
    """
    required = ['site', 'model', 'variable', 'metric', 'value']
    missing = [c for c in required if c not in df.columns]
    assert len(missing), "Dataframe is missing variables: {0}".format(missing)

    df['quants'] = (df.groupby(['variable', 'metric'])['value']
                      .apply(quantile_normalise))

    df_wide = df.set_index(['site', 'metric', 'name', 'variable'])['quants'].unstack()


def quantile_normalise(x, dist='uniform'):
    """Quantile normalise to a standard distribution
    """
    if dist == 'uniform':
        return np.argsort(np.argsort(x)) / (len(x) - 1)


ALL_PLOTS = dict()
ALL_PLOTS.update(DIAGNOSTIC_PLOTS)
ALL_PLOTS.update(PLUMBER_PLOTS)
ALL_PLOTS.update(DRYDOWN_PLOTS)
