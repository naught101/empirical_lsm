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
import os

from pals_utils.data import copy_data


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

    :sim_dict: Dictionary of simulated variable vectors
    :old_ds: xarray dataset from which to copy metadata
    :returns: xarray dataset with sim_dict data

    """
    sim_data = copy_data(old_ds)

    for v in sim_dict:
        sim_var = sim_dict[v]
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
