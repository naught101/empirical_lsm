#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: evaluation.py
Author: ned haughton
Email: ned@nedhaughton.com
Github:
Description: PALS-style model evaluation
"""

import pandas as pd
import numpy as np
import glob
import os

from pals_utils.data import pals_site_name
from pals_utils.stats import run_metrics

from .utils import print_bad, print_good
from .data import get_sites


# Evaluate

def evaluate_simulation(sim_data, flux_data, name, qc=True):
    """Top-level simulation evaluator.

    Compares sim_data to flux_data, using standard metrics.

    TODO: Maybe get model model_name from sim_data directly (this is a PITA at
          the moment, most models don't report it).
    """
    site = pals_site_name(flux_data)
    print_good('Evaluating data for {n} at {s}'.format(n=name, s=site))

    flux_vars = ['NEE', 'Qh', 'Qle']
    eval_vars = sorted(set(flux_vars).intersection(sim_data.data_vars)
                                     .intersection(flux_data.data_vars))

    metric_data = pd.DataFrame()
    metric_data.index.name = 'metric'
    for v in eval_vars:
        if qc:
            v_qc = v + '_qc'
            sim_v = sim_data[v].values.ravel()[flux_data[v_qc].values.ravel() == 1]
            obs_v = flux_data[v].values.ravel()[flux_data[v_qc].values.ravel() == 1]
        else:
            sim_v = sim_data[v].values.ravel()
            obs_v = flux_data[v].values.ravel()

        for m, val in run_metrics(sim_v, obs_v).items():
            metric_data.ix[m, v] = val

    eval_dir = 'source/models/{n}/metrics/'.format(n=name)
    os.makedirs(eval_dir, exist_ok=True)
    eval_path = '{d}/{n}_{s}_metrics.csv'.format(d=eval_dir, n=name, s=site)
    metric_data.to_csv(eval_path)

    return metric_data


def load_sim_evaluation(name, site):
    """Load an evaluation saved from evaluate_simulation.

    :name: model name
    :site: PALS site name
    :returns: pandas dataframe with metrics
    """
    eval_path = 'source/models/{n}/metrics/{n}_{s}_metrics.csv'.format(n=name, s=site)
    metric_data = pd.DataFrame.from_csv(eval_path)

    return metric_data


def get_metric_df(models, sites):
    """Gets a DF of models metrics at the given sites (pre-computed evaluations)

    :models: TODO
    :sites: TODO
    :returns: TODO

    """
    var_order = ['metric', 'NEE', 'Qh', 'Qle', 'Qg', 'Rnet']
    metric_df = []
    for m in models:
        for s in sites:
            try:
                df = pd.read_csv('source/models/{m}/metrics/{m}_{s}_metrics.csv'.format(m=m, s=s))
                df = df[[c for c in var_order if c in df.columns]]
                df['site'] = s
                df['name'] = m
                df = df.set_index(['name', 'site', 'metric']).stack().to_frame()
                df.index.names = ['name', 'site', 'metric', 'variable']
                df.columns = ['value']
                metric_df.append(df)
            except Exception:
                print('skipping {m} at {s}'.format(m=m, s=s))

    metric_df = pd.concat(metric_df).reset_index()

    return metric_df


def get_metric_data(names):
    """Get a dataframe of metric means for each model

    :names: List of models to grab data from
    :returns: All metric data as a pandas dataframe

    """
    import re
    import pandas as pd

    pat = re.compile('[^\W_]+(?=_metrics.csv$)')

    data = []

    for name in names:
        model_dir = 'source/models/' + name
        csv_files = glob.glob(model_dir + '/metrics/*csv')
        if len(csv_files) == 0:
            continue
        model_dfs = []
        for csvf in csv_files:
            site = re.search(pat, csvf).group(0)
            df = pd.DataFrame.from_csv(csvf)
            df['site'] = site
            model_dfs.append(df)
        model_df = pd.concat(model_dfs)
        model_df['name'] = name
        data.append(model_df)
    data = pd.concat(data)

    return data


def normalise_metric(data, metric, quantile=False):
    """Normalises metrics between 0 and 1, where 0 is the best."""
    if metric in ['corr', 'overlap']:  # 1-optimal metrics
        data = np.abs(1 - data)

    if quantile:
        normalised = quantile_normalise(data)
    else:
        normalised = (data - np.min(data)) / (np.max(data) - np.min(data))

    return normalised


def quantile_normalise(x, dist='uniform'):
    """Quantile normalise to a standard distribution
    """
    if dist == 'uniform':
        return np.argsort(np.argsort(x)) / (len(x) - 1)


#################
# Model ranking #
#################

def rank_metric_df(df):
    """Ranks all models by metric, at each site/variable, correcting for inverted metrics"""
    # invert 1-centred metrics
    one_metrics = ['corr', 'overlap']
    df.ix[df['metric'].isin(one_metrics), 'value'] = 1 - df.ix[df['metric'].isin(one_metrics), 'value']

    # use worst option for tied ranks
    df['rank'] = df.groupby(['variable', 'metric', 'site'])['value'].apply(lambda x: x.abs().rank(method='max'))

    # and reinvert
    df.ix[df['metric'].isin(one_metrics), 'value'] = 1 - df.ix[df['metric'].isin(one_metrics), 'value']

    return df


def get_PLUMBER_metrics(name, site='all', variables=['Qle', 'Qh', 'NEE']):
    """get metrics dataframe from a site, with benchmarks for comparison

    :returns: dataframe with metrics for model at site
    """
    csv_file = './source/models/{n}/metrics/{n}_{s}_metrics.csv'

    # benchmark_names = ['1lin', '2lin', '3km27']
    benchmark_names = ['S_lin', 'ST_lin', 'STH_km27']

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

    metric_df = rank_metric_df(metric_df)

    return metric_df


def subset_metric_df(metric_df, metrics):
    """Return only the metrics required
    """
    if metrics == 'standard' or metrics == 'common':
        metrics_list = ['nme', 'mbe', 'sd_diff', 'corr']
        return metric_df[metric_df.metric.isin(metrics_list)].copy()
    elif metrics == 'extremes':
        metrics_list = ['extreme_5', 'extreme_95']
        return metric_df[metric_df.metric.isin(metrics_list)].copy()
    elif metrics == 'distribution':
        metrics_list = ['skewness', 'kurtosis', 'overlap']
        return metric_df[metric_df.metric.isin(metrics_list)].copy()
    else:
        return metric_df
