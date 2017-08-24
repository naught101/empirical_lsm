#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: data.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101/empirical_lsm
Description: empirical_lsm data helper functions
"""

import xarray as xr
import numpy as np
import pandas as pd
import os

from pals_utils.data import get_sites, copy_data, get_met_data, get_config, set_config, \
    pals_xr_to_df, get_multisite_met_df, get_multisite_flux_df

from empirical_lsm.transforms import rolling_mean

import logging
logger = logging.getLogger(__name__)


set_config(['datasets', 'train'], 'PLUMBER_ext')
set_config(['qc_format'], 'PALS')


def sim_dict_to_xr(sim_dict, old_ds):
    """Converts a dictionary of arrays into a xarray dataset with the same geo data as old_ds

    Also works with dataframes (by column)

    :sim_dict: Dictionary of simulated variable vectors
    :old_ds: xarray dataset from which to copy metadata
    :returns: xarray dataset with sim_dict data

    """
    sim_data = copy_data(old_ds)

    for v in sim_dict:
        sim_var = np.array(sim_dict[v])
        sim_var.shape = (sim_var.shape[0], 1, 1)
        # Currently only works with single variable predictions...
        sim_array = xr.DataArray(sim_var, dims=['time', 'y', 'x'],
                                 coords=dict(time=old_ds.coords['time'], y=[1.0], x=[1.0]))
        sim_data[v] = sim_array

    return sim_data


def get_sim_nc_path(name, site):
    """return the sim netcdf path, and make parent directories if they don't already exist.

    :name: name of the model
    :site: PALS site name to run the model at
    :returns: sim netcdf path
    """
    model_path = 'model_data/{n}/'.format(n=name)
    os.makedirs(model_path, exist_ok=True)
    nc_path = '{p}{n}_{s}.nc'.format(p=model_path, n=name, s=site)

    return nc_path


def get_multimodel_data(site, names, variables):
    """Returns a DF with variable columns, time/model indices."""

    data = []
    for n in names:
        path = "model_data/{n}/{n}_{s}.nc".format(n=n, s=site)
        try:
            with xr.open_dataset(path) as ds:
                df = pals_xr_to_df(ds, variables)
                df['name'] = n

                # offset time indices
                if n == 'COLASSiB.2.0':
                    df.index = df.index + pd.Timedelta('-30min')
                if n == 'ORCHIDEE.trunk_r1401':
                    df.index = df.index + pd.Timedelta('-15min')

                df.set_index('name', append=True, inplace=True)
                data.append(df)

        except (OSError, RuntimeError) as e:
            raise Exception(path, e)

    df = pd.concat(data)
    df.columns.name = 'variable'
    return df


def get_multimodel_wide_df(site, names, variables):
    """Returns a DF with model columns, time/variable indices."""
    if isinstance(variables, list):
        df = get_multimodel_data(site, names, variables)
        return df.stack().unstack(level='name')[names]
    elif isinstance(variables, str):
        df = get_multimodel_data(site, names, [variables])
        return df.stack().unstack(level='name')[names].xs(variables, level=1)
    else:
        raise Exception("WTF is variables? %s" % type(variables))


def get_train_data(train_sites, met_vars, flux_vars, use_names, qc=True, fix_closure=True):

    met_train = get_multisite_met_df(train_sites, variables=met_vars, qc=qc, name=use_names)
    flux_train = get_multisite_flux_df(train_sites, variables=flux_vars, qc=qc, name=use_names, fix_closure=fix_closure)

    return dict(
        train_sites=train_sites,
        met_vars=met_vars,
        flux_vars=flux_vars,
        fix_closure=fix_closure,
        met_train=met_train,
        flux_train=flux_train)


def get_test_data(site, met_vars, use_names, qc=False, fix_closure=True):

    # We use gap-filled data for the testing period, or the model fails.
    met_test_xr = get_met_data(site)[site]
    met_test = pals_xr_to_df(met_test_xr, variables=met_vars, qc=qc, name=use_names)

    return dict(
        site=site,
        met_vars=met_vars,
        fix_closure=fix_closure,
        met_test=met_test,
        met_test_xr=met_test_xr)


def get_train_test_data(site, met_vars, flux_vars, use_names, qc=True, fix_closure=True):
    """Gets training and testing data, PLUMBER style (leave one out)

    Set the training set using pals.data.set_config(['datasets', 'train'])"""

    if site == 'debug':
        train_sites = ['Amplero']
        test_site = 'Tumba'

        # Use non-quality controlled data, to ensure there's enough to train
        qc = False

    else:
        train_sites = get_sites(get_config(['datasets', 'train']))
        test_site = site
        if test_site not in train_sites:
            # Running on a non-training site, train on all training sites.
            train_sites = train_sites
        else:
            # Running on a training site, so leave it out.
            train_sites = [s for s in train_sites if s != test_site]
        logger.info("Training with {n} datasets".format(n=len(train_sites)))

    train_dict = get_train_data(train_sites, met_vars, flux_vars, use_names=use_names,
                                qc=qc, fix_closure=fix_closure)

    test_dict = get_test_data(test_site, met_vars, use_names=use_names, qc=False)

    train_test_data = train_dict
    train_test_data.update(test_dict)
    train_test_data['site'] = site

    return train_test_data


def get_lagged_df(df, lags=['30min', '2h', '6h', '2d', '7d', '30d', '90d']):
    """Get lagged variants of variables"""

    data = {}
    for v in df.columns:
        data[v] = pd.DataFrame(
            np.concatenate([rolling_mean(df[[v]].values, l) for l in lags], axis=1),
            columns=lags, index=df.index)
    return pd.concat(data, axis=1)
