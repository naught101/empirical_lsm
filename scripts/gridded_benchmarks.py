#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: gridded_benchmarks.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: TODO: File description

Usage:
    gridded_benchmarks.py generate <benchmark> <forcing> [--years=<years>]
    gridded_benchmarks.py (-h | --help | --version)

Options:
    benchmark:       1lin, 3km27
    forcing:         PRINCETON, CRUNCEP
    --years=<years>  2012-2013, python indexing style
    -h, --help       Show this screen and exit.
"""

from docopt import docopt

import os
import sys
import glob
import datetime
import numpy as np
import xarray as xr

from sklearn.linear_model import LinearRegression
from sklearn.cluster import MiniBatchKMeans

import pals_utils.data as pud

from ubermodel.clusterregression import ModelByCluster


def get_data_dir():
    """get data directory """

    return './data'


def get_sites():
    """load names of available sites"""

    # return ['Tumba']

    data_dir = get_data_dir()

    with open(data_dir + '/PALS/datasets/sites.txt') as f:
        sites = [s.strip() for s in f.readlines()]

    return sites


def get_model_vars(benchmark):

    if benchmark == '1lin':
        met_vars = ['SWdown']
        flux_vars = ['Qh', 'Qle']
        return met_vars, flux_vars
    if benchmark == '3km27':
        met_vars = ['SWdown', 'Tair', 'RelHum']
        flux_vars = ['Qh', 'Qle']
        return met_vars, flux_vars
    else:
        sys.exit("Unknown benchmark %s, exiting" % benchmark)


def get_forcing_vars(forcing, met_vars, in_file=False):
    """get the names of forcing variables for each forcing dataset

    :forcing: TODO
    :met_vars: TODO
    :returns: TODO

    """
    if forcing == "PRINCETON":
        var_dict = dict(
            LWdown="dlwrf",
            PSurf="pres",
            Wind="wind",
            SWdown="dswrf",
            Tair="tas",
            Qair="shum")
    elif forcing == "CRUNCEP":
        if in_file:
            var_dict = dict(
                LWdown="Incoming_Long_Wave_Radiation",
                PSurf="Pression",
                Wind="U_wind_component",
                SWdown="Incoming_Short_Wave_Radiation",
                Tair="Temperature",
                Qair="Air_Specific_Humidity")
        else:
            var_dict = dict(
                LWdown="lwdown",
                PSurf="press",
                Wind="*wind",
                SWdown="swdown",
                Tair="tair",
                Qair="qair")
    elif forcing in ["WATCH_WFDEI", "GSWP3"]:
        var_dict = dict(
            LWdown="LWdown",
            PSurf="PSurf",
            Wind=None,
            SWdown="SWdown",
            Qair="Qair")
    else:
        sys.exit("Unknown forcing dataset %s - more coming later" % forcing)

    return {v: var_dict[v] for v in met_vars}


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


def get_relhum(data):
    """Get relative humidity from specific humidity"""

    data['RelHum'] = pud.Spec2RelHum(data['Qair'], data['Tair'], data['PSurf'])


# Individual dataloaders

def get_CRUNCEP_data(met_vars, year):

    file_tpl = "{d}/gridded/CRUNCEP/cruncep2015_1_{v}_{y}.nc"

    forcing_vars = get_forcing_vars('CRUNCEP', met_vars)
    infile_vars = get_forcing_vars('CRUNCEP', met_vars, in_file=True)

    data = {}
    for v, fv in forcing_vars.items():
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

    forcing_vars = get_forcing_vars('PRINCETON', met_vars)

    data = {}
    for v, fv in forcing_vars.items():
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


def get_forcing_data(forcing, met_vars, year):
    """Loads a single xarray dataset from multiple files.
    """
    relhum = 'RelHum' in met_vars
    if relhum:
        orig_met_vars = met_vars
        met_vars = orig_met_vars.copy()  # not sure if this is necessary to avoid side-effects
        met_vars.remove('RelHum')
        met_vars = set(met_vars).union(['Tair', 'Qair', 'PSurf'])

    if forcing == 'CRUNCEP':
        data = get_CRUNCEP_data(met_vars, year)
    if forcing == 'GSWP3':
        data = get_GSWP3_data(met_vars, year)
    if forcing == 'PRINCETON':
        data = get_PRINCETON_data(met_vars, year)
    if forcing == 'WATCH_WFDEI':
        data = get_WATCH_WFDEI_data(met_vars, year)

    if relhum:
        get_relhum(data)
        data = data[orig_met_vars]

    return(data)


def get_benchmark_model(benchmark):
    """returns a scikit-learn style model/pipeline

    :benchmark: TODO
    :returns: TODO

    """
    if benchmark == '1lin':
        return LinearRegression()
    if benchmark == '3km27':
        return ModelByCluster(MiniBatchKMeans(27),
                              LinearRegression())
    else:
        sys.exit("Unknown benchmark {b}".format(b=benchmark))


def predict_gridded(model, forcing_data, flux_vars):
    """predict model results for gridded data

    :model: TODO
    :data: TODO
    :returns: TODO

    """
    # set prediction metadata
    prediction = forcing_data[list(forcing_data.coords)]

    # Arrays like (var, lon, lat, time)
    result = np.full([len(flux_vars),
                      forcing_data.dims['lon'],
                      forcing_data.dims['lat'],
                      forcing_data.dims['time']],
                     np.nan)

    print("lon:    ", end='', flush=True)

    for lon in range(len(forcing_data['lon'])):
        print("\b\b\b\b\b", str(lon).rjust(4), end='', flush=True)
        for lat in range(len(forcing_data['lat'])):
            # If data has fill values, only predict with masked data
            if np.all(-1e8 < forcing_data.isel(time=0, lat=lat, lon=lon).to_array() < 1e8):
                result[:, lon, lat, :] = model.predict(
                    forcing_data.isel(lat=lat, lon=lon).to_array().T
                ).T
    print("")

    for i, fv in enumerate(flux_vars):
        prediction.update(
            {fv: xr.DataArray(result[i, :, :, :],
                              dims=['lon', 'lat', 'time'],
                              coords=forcing_data.coords
                              )
             }
        )

    return prediction


def xr_add_attributes(ds, benchmark, forcing, sites):

    ds.attrs["Model name"] = benchmark
    met_vars, flux_vars = get_model_vars(benchmark)
    ds.attrs["Forcing_variables"] = ', '.join(met_vars)
    ds.attrs["Forcing_dataset"] = forcing
    ds.attrs["Training_dataset"] = "Fluxnet_1.4"
    ds.attrs["Training_sites"] = ', '.join(sites)
    ds.attrs["Production_time"] = datetime.datetime.now().isoformat()
    ds.attrs["Production_source"] = "gridded_benchmarks.py"
    ds.attrs["PALS_dataset_version"] = "1.4"
    ds.attrs["Contact"] = "ned@nedhaughton.com"


def fit_and_predict(benchmark, forcing, years='2012-2013'):
    """Fit a benchmark to some PALS files, then generate an output matching a gridded dataset
    """

    met_vars, flux_vars = get_model_vars(benchmark)

    sites = get_sites()

    print("Loading fluxnet data for %d sites" % len(sites))
    met_data = pud.get_met_df(sites, met_vars, qc=True)
    flux_data = pud.get_flux_df(sites, flux_vars, qc=True)
    qc_index = (np.all(np.isfinite(met_data), axis=1, keepdims=True) &
                np.all(np.isfinite(flux_data), axis=1, keepdims=True)).ravel()

    model = get_benchmark_model(benchmark)

    print("Fitting model {b} using {n} samples of {m} to predict {f}".format(
        b=benchmark, n=qc_index.shape[0], m=met_vars, f=flux_vars))
    model.fit(met_data.ix[qc_index, :], flux_data.ix[qc_index, :])

    # prediction datasets
    outdir = "{d}/gridded_benchmarks/{b}_{f}".format(d=get_data_dir(), b=benchmark, f=forcing)
    os.makedirs(outdir, exist_ok=True)
    outfile_tpl = outdir + "/{b}_{f}_{v}_{y}.nc"
    years = [int(s) for s in years.split('-')]
    for year in range(*years):

        print("Loading Forcing data for", year)
        try:
            data = get_forcing_data(forcing, met_vars, year)
        except Exception as e:
            print("error in year {y}, skipping: {e}".format(y=year, e=e))
            continue

        print("Predicting", year, end=': ', flush=True)
        try:
            result = predict_gridded(model, data, flux_vars)
        except Exception as e:
            print("error in year {y}, skipping: {e}".format(y=year, e=e))
            continue

            xr_add_attributes(result, benchmark, forcing, sites)
        for fv in flux_vars:
            filename = outfile_tpl.format(b=benchmark, f=forcing, v=fv, y=year)
            print("saving to ", filename)
            result[[fv]].to_netcdf(filename, encoding={fv: {'dtype': 'float32'}})

    return


def main(args):

    if args['generate']:
        if args['--years'] is not None:
            fit_and_predict(args['<benchmark>'], args['<forcing>'], args['--years'])
        else:
            fit_and_predict(args['<benchmark>'], args['<forcing>'])
    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
