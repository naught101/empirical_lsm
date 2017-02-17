#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: data.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description: ubermodel data helper functions
"""

import xarray as xr
import numpy as np
import pandas as pd
import joblib as jl
import os

from pals_utils.data import copy_data, get_met_data, get_flux_data, pals_xr_to_df, xr_list_to_df

from ubermodel.transforms import rolling_mean


def get_data_dir():
    """get data directory """

    return './data'


def get_sites(site_set='all'):
    """load names of available sites"""

    data_dir = get_data_dir()

    if site_set == 'debug':
        sites = ['Tumba']
    else:
        if site_set == 'all':
            filename = 'sites.txt'
        elif site_set == 'PLUMBER':
            filename = 'sites_PLUMBER.txt'
        elif site_set == 'PLUMBER_ext':
            filename = 'sites_PLUMBER_ext.txt'

        path = '{d}/PALS/datasets/{f}'.format(d=data_dir, f=filename)
        with open(path) as f:
            sites = [s.strip() for s in f.readlines()]

    return sites


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
                df.set_index('name', append=True, inplace=True)
                data.append(df)
        except OSError as e:
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


def get_multisite_met_df(sites, variables, name=False, qc=False):
    """Load some data and convert it to a dataframe

    :sites: str or list of strs: site names
    :variables: list of variable names
    :qc: Whether to replace bad quality data with NAs
    :names: Whether to include site-names
    :returns: pandas dataframe

    """
    if isinstance(sites, str):
        sites = [sites]

    print("Met data: loading... ", end='')
    # TODO: Split this up into single sites, and use Joblib to cache each site load. Multiple models should be able to re-load the same data, saving time.
    data = get_met_data(sites)
    print("converting... ")
    return xr_list_to_df(data.values(), variables=variables, qc=True, name=name)


def get_multisite_flux_df(sites, variables, name=False, qc=False, fix_closure=True):
    """Load some data and convert it to a dataframe

    :sites: str or list of strs: site names
    :variables: list of variable names
    :qc: Whether to replace bad quality data with NAs
    :names: Whether to include site-names
    :returns: pandas dataframe

    """
    if isinstance(sites, str):
        sites = [sites]

    print("Flux data: loading... ", end='')
    data = get_flux_data(sites, fix_closure=True)
    print("converting... ")
    return xr_list_to_df(data.values(),
                         variables=variables, qc=True, name=name)

mem = jl.Memory(cachedir=os.path.join(os.path.expanduser('~'), 'tmp', 'cache'))

get_multisite_met_df_cached = mem.cache(get_multisite_met_df)
get_multisite_flux_df_cached = mem.cache(get_multisite_flux_df)


def get_train_test_sets(site, met_vars, flux_vars, use_names, fix_closure=True):

    if site == 'debug':
        train_sets = ['Amplero']
        test_site = 'Tumba'

        # Use non-quality controlled data, to ensure there's enough to train
        met_train = get_multisite_met_df(train_sets, variables=met_vars, name=use_names)
        flux_train = get_multisite_flux_df(train_sets, variables=flux_vars, name=use_names, fix_closure=fix_closure)

        met_test_xr = get_met_data(test_site)[test_site]
        met_test = pals_xr_to_df(met_test_xr, variables=met_vars)

        met_test_xr = met_test_xr.isel(time=slice(0, 5000))
        met_train = met_train[0:5000]
        flux_train = flux_train[0:5000]
        met_test = met_test[0:5000]

    else:
        plumber_datasets = get_sites('PLUMBER_ext')
        if site not in plumber_datasets:
            # Using a non-PLUMBER site, train on all PLUMBER sites.
            train_sets = plumber_datasets
        else:
            # Using a PLUMBER site, leave one out.
            train_sets = [s for s in plumber_datasets if s != site]
        print("Training with {n} datasets".format(n=len(train_sets)))

        met_train = get_multisite_met_df(train_sets, variables=met_vars, qc=True, name=use_names)

        # We use gap-filled data for the testing period, or the model fails.
        met_test_xr = get_met_data(site)[site]
        met_test = pals_xr_to_df(met_test_xr, variables=met_vars)

        flux_train = get_multisite_flux_df(train_sets, variables=flux_vars, qc=True, name=use_names)

    return met_train, met_test, met_test_xr, flux_train


def get_lagged_df(df, lags=['30min', '2h', '6h', '2d', '7d', '30d', '90d']):
    """Get lagged variants of variables"""

    data = {}
    for v in df.columns:
        data[v] = pd.DataFrame(
            np.concatenate([rolling_mean(df[[v]].values, l) for l in lags], axis=1),
            columns=lags, index=df.index)
    return pd.concat(data, axis=1)
