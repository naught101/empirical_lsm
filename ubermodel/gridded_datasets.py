#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: gridded_datasets.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Helpers for gridded datasets
"""

import sys
import glob

import numpy as np
import xarray as xr

import pals_utils.data as pud

from ubermodel.data import get_data_dir


def get_dataset_vars(dataset, met_vars, in_file=False):
    """get the names of dataset variables for each dataset

    :dataset: gridded dataset name
    :met_vars: list of met variables (must match know variables)
    :returns: returns dictionary of variables as named in the dataset file

    """
    if dataset == "PRINCETON":
        var_dict = dict(
            LWdown="dlwrf",
            PSurf="pres",
            Wind="wind",
            SWdown="dswrf",
            Tair="tas",
            Qair="shum",
            Rainf="prcp")
    elif dataset == "CRUNCEP":
        if in_file:
            var_dict = dict(
                LWdown="Incoming_Long_Wave_Radiation",
                PSurf="Pression",
                Wind="U_wind_component",
                SWdown="Incoming_Short_Wave_Radiation",
                Tair="Temperature",
                Qair="Air_Specific_Humidity",
                Rainf="Total_Precipitation")
        else:
            var_dict = dict(
                LWdown="lwdown",
                PSurf="press",
                Wind="*wind",
                SWdown="swdown",
                Tair="tair",
                Qair="qair",
                Rainf="rain")
    elif dataset in ["WATCH_WFDEI", "GSWP3"]:
        return met_vars
    else:
        sys.exit("Unknown dataset %s - more coming later" % dataset)

    return {v: var_dict[v] for v in met_vars}


def get_dataset_freq(dataset):
    """returns the data rate in hours for each dataset"""
    if dataset in ["PRINCETON", "GSWP3"]:
        return 3
    elif dataset == "CRUNCEP":
        return 6
    elif dataset in ["WATCH_WFDEI"]:
        return 8
    else:
        sys.exit("Unknown dataset %s - more coming later" % dataset)


def get_cruncep_datetime(timestep, year):
    """Converts CRUNCEP's dodgy time counters to real times"""
    doy = timestep // 4

    month_lens = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_starts = np.cumsum(month_lens)
    month = np.argmax(doy < month_starts)

    day = doy - month_starts[month - 1] + 1

    hour = (timestep % 4) * 6

    return np.datetime64("{y}-{m:02d}-{d:02d}T{h:02d}:00".format(y=year, m=month, d=day, h=hour))


def correct_CRUNCEP_coords(data, v, year):
    """Converts coordinates to match PRINCETON dataset"""

    data = data.rename({"longitude": "lon",
                        "latitude": "lat",
                        "time_counter": "time"})
    data['time'] = np.vectorize(get_cruncep_datetime)(data.time, year)
    return data[v].copy()


def correct_WATCH_WFDEI_coords(data, v, year):
    """Converts coordinates to match PRINCETON dataset"""

    fixed_times = xr.decode_cf(xr.DataArray(data[v].values,
                                            coords=[data.time, data.lat, data.lon],
                                            dims=['time', 'lat', 'lon'],
                                            name=v)
                                 .to_dataset())
    return fixed_times[v].copy()


# Individual dataloaders

def get_CRUNCEP_data(met_vars, year):

    file_tpl = "{d}/gridded/CRUNCEP/cruncep2015_1_{v}_{y}.nc"

    dataset_vars = get_dataset_vars('CRUNCEP', met_vars)
    infile_vars = get_dataset_vars('CRUNCEP', met_vars, in_file=True)

    data = {}
    for v, fv in dataset_vars.items():
        # TODO: CRUNCEP uses a mask variable, so replace with NANs?
        with xr.open_dataset(file_tpl.format(d=get_data_dir(), v=fv, y=year)) as ds:
            data[v] = correct_CRUNCEP_coords(ds, infile_vars[v], year)
    data = xr.Dataset(data)

    # CRUNCEP uses stupid units: ftp://nacp.ornl.gov/synthesis/2009/frescati/model_driver/cru_ncep/analysis/readme.htm
    data['SWdown'] = data.SWdown / 21600

    return data


def get_GSWP3_data(met_vars, year):

    file_tpl = "{d}/gridded/GSWP3/{v}/GSWP3.BC.{v}.3hrMap.{y}.nc"

    data = {}
    for v in met_vars:
        with xr.open_dataset(file_tpl.format(d=get_data_dir(), v=v, y=year)) as ds:
            data[v] = ds[v].copy()
    data = xr.Dataset(data)

    return data


def get_PRINCETON_data(met_vars, year):

    file_tpl = "{d}/gridded/PRINCETON/0_5_3hourly/{v}_3hourly_{y}-{y}.nc"

    dataset_vars = get_dataset_vars('PRINCETON', met_vars)

    data = {}
    for v, fv in dataset_vars.items():
        with xr.open_dataset(file_tpl.format(d=get_data_dir(), v=fv, y=year)) as ds:
            data[v] = ds[fv].copy()
    data = xr.Dataset(data)

    return data


def get_WATCH_WFDEI_data(met_vars, year):

    file_tpl = "{d}/gridded/WATCH_WFDEI/{v}_WFDEI/{v}_WFDEI_{y}*.nc"

    data = {}
    for v in met_vars:
        files = sorted(glob.glob(file_tpl.format(d=get_data_dir(), v=v, y=year)))
        datasets = [xr.open_dataset(f, decode_times=False) for f in files]
        # TODO: WATCH_FDEI uses a fill-value mask, so replace with NANs?
        data[v] = xr.concat(
            [correct_WATCH_WFDEI_coords(ds, v, year) for ds in datasets],
            dim='time')
        [ds.close() for ds in datasets]
    data = xr.Dataset(data)

    return data


def get_MODIS_data(met_vars, year):
    assert met_vars == ['Qle'], 'MODIS loading is incomplete'

    file_tpl = '{d}/gridded/MODIS/MOD16_{v}_GSWP3_{y}.nc'

    data = {}
    for v in met_vars:
        with xr.open_dataset(file_tpl.format(d=get_data_dir(), v='ET', y=year)) as ds:
            data['Qle'] = ds['et'].copy()
    data = xr.Dataset(data)

    return data


def get_MPI_data(met_vars, year):
    assert met_vars == ['Qle'], 'MPI loading is incomplete'

    file_tpl = '{d}/gridded/MPI/Ensemble{v}cor_May12.{y}.nc'

    data = {}
    for v in met_vars:
        with xr.open_dataset(file_tpl.format(d=get_data_dir(), v='LE', y=year), decode_times=False) as ds:
            data['Qle'] = ds['EnsembleLEcor_May12'].copy()
    data = xr.Dataset(data)

    return data


def get_GLEAM3a_data(met_vars, year):
    assert met_vars == ['Qle'], 'GLEAM loading is incomplete'

    file_tpl = '{d}/gridded/GLEAM_v3a_BETA/{v}_{y}_GLEAM_v3a_BETA.nc'

    data = {}
    for v in met_vars:
        with xr.open_dataset(file_tpl.format(d=get_data_dir(), v='Et', y=year), decode_times=False) as ds:
            data['Qle'] = ds['Et'].copy()
    data = xr.Dataset(data)

    return data


def get_relhum(data):
    """Get relative humidity from specific humidity"""

    data['RelHum'] = pud.Spec2RelHum(data['Qair'], data['Tair'], data['PSurf'])


def get_dataset_data(dataset, met_vars, year):
    """Loads a single xarray dataset from multiple files.
    """
    relhum = 'RelHum' in met_vars
    if relhum:
        orig_met_vars = list(met_vars)
        met_vars = orig_met_vars.copy()  # not sure if this is necessary to avoid side-effects
        met_vars.remove('RelHum')
        met_vars = set(met_vars).union(['Tair', 'Qair', 'PSurf'])

    if dataset == 'CRUNCEP':
        data = get_CRUNCEP_data(met_vars, year)
    if dataset == 'GSWP3':
        data = get_GSWP3_data(met_vars, year)
    if dataset == 'PRINCETON':
        data = get_PRINCETON_data(met_vars, year)
    if dataset == 'WATCH_WFDEI':
        data = get_WATCH_WFDEI_data(met_vars, year)

    if relhum:
        get_relhum(data)
        data = data[orig_met_vars]

    return(data)
