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


MET_VARS = ["SWdown", "Tair", "LWdown", "Wind", "Rainf", "PSurf", "Qair"]
MET_QC = [v + "_qc" for v in MET_VARS]
FLUX_VARS = ["Qh", "Qle", "Rnet", "NEE"]
FLUX_QC = [v + "_qc" for v in FLUX_VARS]
GEO_VARS = ["latitude", "longitude", "elevation", "reference_height"]


def copy_data(dataset, met=None, flux=None):
    """Return a copy of the land dataset.

    met and flux components optional: use dataset.met, dataset.flux
    """
    return dataset[GEO_VARS]

def time_split(dataset, ratio):
    """Split data along the time axis, by ratio"""
    first_len = np.floor(ratio * dataset.dims['time'])
    first = dataset.copy_data()
    second = dataset.copy_data()
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
