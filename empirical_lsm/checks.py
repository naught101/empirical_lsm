#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: checks.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/empirical_lsm
Description: model sanity checks.
"""

import os
import numpy as np

import pandas as pd
import xarray as xr

import logging
logger = logging.getLogger(__name__)


flux_vars = ['NEE', 'Qh', 'Qle']


def check_var_too_low(data):
    if (data.min() < -1500):
        logger.warning("data too low!")
        return True
    else:
        return False


def check_var_too_high(data):
    if (data.max() > 5000):
        logger.warning("data too high!")
        return True
    else:
        return False


def check_var_change_too_fast(data):
    if (np.abs(np.diff(data)) > 1500):
        logger.warning("data changing too fast!")
        return True
    else:
        return False


def run_var_checks(data):
    return (check_var_too_low(data) or
            check_var_too_high(data) or
            check_var_change_too_fast(data))


def check_model_data(models, sites):
    """Checks all models

    :models: list of model names
    """
    bad_simulations = []

    print("Checking {nm} models at {ns} sites.".format(nm=len(models), ns=len(sites)))
    for model in models:
        print('Checking {m}:'.format(m=model))
        for site in sites:
            file_path = 'model_data/{m}/{m}_{s}.nc'.format(m=model, s=site)
            if not os.path.exists(file_path):
                logger.warning('missing model run: {m} at {s}', dict(m=model, s=site))
                print('x', end='', flush=True)
                continue
            with xr.open_dataset(file_path) as ds:
                try:
                    model_sanity_check(ds, model, site)
                except RuntimeError as e:
                    print('\n' + str(e))
                    logger.error(str(e))
                    bad_simulations.append((model, site))
                else:
                    print('.', end='', flush=True)
                    logger.info('model data for {m} at {s} looks ok', dict(m=model, s=site))
        print('')

    return bad_simulations


def check_metrics(models, sites):
    """Checks metrics to see if they're bullshit

    :models: TODO
    :sites: TODO
    :returns: TODO

    """
    # Glob all metric filenames

    bad_simulations = []

    for model in models:
        for site in sites:
            csv_path = 'source/models/{m}/metrics/{m}_{s}_metrics.csv'.format(m=model, s=site)
            if not os.path.exists(csv_path):
                continue

            metrics = pd.read_csv(csv_path, index_col=0)
            if ((metrics > 500).any().any() or
                    (metrics.loc['corr'] > 1).any() or
                    (metrics.loc['corr'] < -1).any() or
                    (metrics.loc['overlap'] > 1).any() or
                    (metrics.loc['overlap'] < 0).any()):

                logger.error("Crazy value(s) for {m} at {s}:".format(m=model, s=site))

                indices = [(metrics.index[a[0]], metrics.columns[a[1]])
                           for a in np.where(metrics == 2)]
                for m, v in indices:
                    logger.error("    {v} {m}: {val}".format(v=v, m=m, val=metrics.loc[m, v]))

                bad_simulations.append((model, site))

    return bad_simulations


def model_sanity_check(sim_data, name, site):
    """Checks a model's output for clearly incorrect values, warns the user,
    and saves debug output

    :sim_data: xarray dataset with flux output
    :name: model name
    :site: site name
    """
    warning = ""
    for v in flux_vars:
        if v not in sim_data:
            warning = v + " missing"

    if warning == "":
        for v in flux_vars:
            # Check output sanity
            if check_var_too_low(sim_data[v].values):
                warning = v + " too low"
                break
            if check_var_too_high(sim_data[v].values):
                warning = v + " too high"
                break

    if warning == "":
        sim_diff = sim_data.diff('time')
        for v in flux_vars:
            # Check rate-of-change sanity
            if (abs(sim_diff[v].values).max() > 1500):
                warning = v + " changing rapidly"
                break

    if warning != "":
        warning = "Probable bad model output: {w} at {s} for {n}".format(
            w=warning, s=site, n=name)
        raise RuntimeError(warning)

    return
