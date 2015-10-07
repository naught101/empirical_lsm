#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: land_data.py
Author: ned haughton
Email: ned@nedhaughton.com
Github:
Description: Land surface data handler for PALS data.
"""

import numpy as np
import xray
# import pandas as pd
# import joblib
# import time

MET_VARS = ["SWdown", "LWdown", "Tair", "Qair", "Rainf", "Wind", "PSurf"]
MET_QC = [v + "_qc" for v in MET_VARS]
FLUX_VARS = ["Qh", "Qle", "Qg", "Rnet", "NEE", "GPP"]
FLUX_QC = [v + "_qc" for v in FLUX_VARS]
GEO_VARS = ["latitude", "longitude", "elevation", "reference_height"]

DATAPATH = 'data/PALS/datasets'
DATASETS = ['Amplero', 'Blodgett', 'Bugac', 'ElSaler2', 'ElSaler',
            'Espirra', 'FortPeck', 'Harvard', 'Hesse', 'Howard',
            'Howlandm', 'Hyytiala', 'Kruger', 'Loobos', 'Merbleue',
            'Mopane', 'Palang', 'Sylvania', 'Tumba', 'UniMich']

met_paths = ['%s/met/%sFluxnet.1.4_met.nc' % (DATAPATH, s) for s in DATASETS]
flux_paths = ['%s/flux/%sFluxnet.1.4_flux.nc' % (DATAPATH, s) for s in DATASETS]


def get_datasets(paths):
    return [xray.open_dataset(path) for path in paths]


def get_met_datasets():
    return get_datasets(met_paths)


def get_flux_datasets():
    return get_datasets(flux_paths)


def copy_data(dataset):
    """Return a copy of the land dataset.

    met and flux components optional: use dataset.met, dataset.flux
    """
    return dataset[GEO_VARS]


def time_split(dataset, ratio):
    """Split data along the time axis, by ratio"""
    first_len = np.floor(ratio * dataset.dims['time'])
    first = dataset[dataset['time'] < dataset['time'][first_len]]
    second = dataset[dataset['time'] >= dataset['time'][first_len]]
    return first, second


def load_ncdf(filename, with_qc=False):
    """Load data from a PALS-style netCDF file
    
    TODO: deal with quality control flags (return NaN where qc=0?)

    :param filename: path to file to open
    """
    data = xray.open_dataset(filename)
    data_vars = list(data.vars.keys())
    if with_qc:
        vars = list(set(MET_VARS).union(MET_QC).union(FLUX_VARS).union(FLUX_QC).intersection(data_vars))
    else:
        vars = list(set(MET_VARS).union(FLUX_VARS).intersection(data_vars))

    return(data[met_vars + flux_vars])
