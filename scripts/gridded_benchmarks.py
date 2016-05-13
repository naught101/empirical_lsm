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
    benchmark:       1lin, etc.
    forcing:         Princeton, etc.
    --years=<years>  2012-2013, python style
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

import pals_utils.data as pud


def get_data_dir():
    """get data directory """

    return './data'


def get_sites():
    """load names of available sites"""

    data_dir = get_data_dir()

    with open(data_dir + '/PALS/datasets/sites.txt') as f:
        sites = [s.strip() for s in f.readlines()]

    return sites


def get_vars(benchmark):

    if benchmark == '1lin':
        met_vars = ['SWdown']
        flux_vars = ['Qh', 'Qle']
        return met_vars, flux_vars
    else:
        sys.exit("Unknown benchmark %s, exiting" % benchmark)


def get_forcing_vars(forcing, met_vars):
    """get the names of forcing variables for each forcing dataset

    :forcing: TODO
    :met_vars: TODO
    :returns: TODO

    """
    if forcing == "Princeton":
        var_dict = dict(
            LWdown="dlwrf",
            PSurf="pres",
            Wind="wind",
            SWdown="dswrf",
            Qair="shum")
        return {v: var_dict[v] for v in met_vars}
    else:
        sys.exit("Unknown forcing dataset %s - more coming later" % forcing)


def get_forcing_files(forcing, met_vars, year):
    """Gets all forcing files

    :forcing: which forcing type

    """
    data_dir = get_data_dir()

    if forcing == 'Princeton':
        forcing_vars = get_forcing_vars(forcing, met_vars)
        fileset = dict()
        for mv, fv in forcing_vars.items():
            file_tpl = "{d}/gridded/PRINCETON/0_5_3hourly/{v}_3hourly_{y}-{y}.nc"
            files = glob.glob(file_tpl.format(d=data_dir, y=year, v=fv))
            if len(files) == 1:
                fileset[mv] = files[0]
        if len(fileset) == len(met_vars):
            return fileset
    else:
        sys.exit("Unknown forcing dataset %s - more coming later" % forcing)


def get_forcing_data(forcing, met_vars, year):
    """Loads a single xarray dataset from multiple files.

    :forcing: TODO
    :fileset: TODO
    :returns: TODO

    """
    forcing_vars = get_forcing_vars(forcing, met_vars)

    fileset = get_forcing_files(forcing, met_vars, year)

    datasets = {v: xr.open_dataset(f) for v, f in fileset.items()}
    data = xr.Dataset({v: ds[forcing_vars[v]].copy() for v, ds in datasets.items()})
    for v, ds in datasets.items():
        ds.close()
    return(data)


def get_benchmark_model(benchmark):
    """returns a scikit-learn style model/pipeline

    :benchmark: TODO
    :returns: TODO

    """
    if benchmark == '1lin':
        return LinearRegression()
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
    met_vars, flux_vars = get_vars(benchmark)
    ds.attrs["Forcing_variables"] = met_vars
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

    met_vars, flux_vars = get_vars(benchmark)

    sites = get_sites()

    print("Loading fluxnet data for %d sites" % len(sites))
    met_data = pud.get_met_df(sites, met_vars, qc=True)
    flux_data = pud.get_flux_df(sites, flux_vars, qc=True)
    qc_index = (np.isfinite(met_data) & np.all(np.isfinite(flux_data), axis=1, keepdims=True)).values.ravel()

    model = get_benchmark_model(benchmark)

    print("Fitting model {b} using {m} to predict {f}".format(b=benchmark, m=met_vars, f=flux_vars))
    model.fit(met_data.ix[qc_index, :], flux_data.ix[qc_index, :])

    # prediction datasets
    outdir = "{d}/gridded_benchmarks/{b}_{f}".format(d=get_data_dir(), b=benchmark, f=forcing)
    os.makedirs(outdir, exist_ok=True)
    outfile_tpl = outdir + "/{b}_{f}_{v}_{y}.nc"
    years = [int(s) for s in years.split('-')]
    for year in range(*years):
        print("Predicting", year, end=': ', flush=True)
        data = get_forcing_data(forcing, met_vars, year)
        result = predict_gridded(model, data, flux_vars)
        xr_add_attributes(result, benchmark, forcing, sites)
        for fv in flux_vars:
            filename = outfile_tpl.format(b=benchmark, f=forcing, v=fv, y=year)
            print("saving to ", filename)
            result[[fv]].to_netcdf(filename)

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
