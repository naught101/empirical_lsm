#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: Fits and runs a basic model and produces rst output with diagnostics

Usage:
    run_model.py run <name> <site> [--no-mp] [--multivariate]

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt

import joblib as jl
import pandas as pd
import numpy as np
import sys
import os

from multiprocessing import Pool

from pals_utils.constants import MET_VARS
from pals_utils.data import get_met_data, get_flux_data, pals_xr_to_df, xr_list_to_df

from ubermodel.transforms import LagWrapper
from ubermodel.models import get_model
from ubermodel.data import get_sites, sim_dict_to_xr
from ubermodel.utils import print_good, print_warn


def get_multisite_df(sites, typ, variables, name=False, qc=False):
    """Load some data and convert it to a dataframe

    :sites: str or list of strs: site names
    :variables: list of variable names
    :qc: Whether to replace bad quality data with NAs
    :names: Whether to include site-names
    :returns: pandas dataframe

    """
    if isinstance(sites, str):
        sites = [sites]

    if typ == 'met':
        return xr_list_to_df(get_met_data(sites).values(),
                             variables=variables, qc=True, name=name)
    elif typ == 'flux':
        return xr_list_to_df(get_flux_data(sites).values(),
                             variables=variables, qc=True, name=name)
    else:
        assert False, "Bad dataset type: %s" % typ

mem = jl.Memory(cachedir=os.path.join(os.path.expanduser('~'), 'tmp', 'cache'))

get_multisite_df_cached = mem.cache(get_multisite_df)


def get_train_test_sets(site, met_vars, flux_vars, use_names):
    print("Loading all data... ")

    if site == 'debug':
        train_sets = ['Amplero']
        test_site = 'Tumba'

        print("Converting... ")
        # Use non-quality controlled data, to ensure there's enough to train
        met_train = get_multisite_df(train_sets, typ='met', variables=met_vars, name=use_names)
        flux_train = get_multisite_df(train_sets, typ='flux', variables=flux_vars, name=use_names)

        met_test_xr = get_met_data(test_site)[test_site]
        met_test = pals_xr_to_df(met_test_xr, variables=met_vars)

        met_test_xr = met_test_xr.isel(time=slice(0, 5000))
        met_train = met_train[0:5000]
        flux_train = flux_train[0:5000]
        met_test = met_test[0:5000]

    else:
        plumber_datasets = get_sites('PLUMBER')
        if site not in plumber_datasets:
            # Using a non-PLUMBER site, train on all PLUMBER sites.
            train_sets = plumber_datasets
        else:
            # Using a PLUMBER site, leave one out.
            train_sets = [s for s in plumber_datasets if s != site]

        print("Converting... ")
        met_train = get_multisite_df(train_sets, typ='met', variables=met_vars, qc=True, name=use_names)

        # We use gap-filled data for the testing period, or the model fails.
        met_test_xr = get_met_data(site)[site]
        met_test = pals_xr_to_df(met_test_xr, variables=met_vars)

        flux_train = get_multisite_df(train_sets, typ='flux', variables=flux_vars, qc=True, name=use_names)

    return met_train, met_test, met_test_xr, flux_train


def fit_predict_univariate(model, flux_vars, met_train, met_test, met_test_xr, flux_train):
    """Fits a model one output variable at a time """
    sim_data_dict = dict()
    for v in flux_vars:
        # TODO: Might eventually want to update this to run multivariate-out models
        # There isn't much point right now, because there is almost no data where all variables are available.
        flux_train_v = flux_train[[v]]

        # Ditch all of the incomplete data
        qc_index = (~pd.concat([met_train, flux_train_v], axis=1).isnull()).apply(all, axis=1)
        if qc_index.sum() > 0:
            print("Training {v} using {count} complete samples out of {total}"
                  .format(v=v, count=qc_index.sum(), total=met_train.shape[0]))
        else:
            print("No training data, skipping variable %s" % v)
            continue

        model.fit(X=met_train[qc_index], y=flux_train_v[qc_index])
        print("Fitting complete.")

        sim_data_dict[v] = model.predict(met_test)
        print("Prediction complete.")

    if len(sim_data_dict) < 1:
        print("No fluxes successfully fitted, quitting")
        sys.exit()

    sim_data = sim_dict_to_xr(sim_data_dict, met_test_xr)

    return sim_data


def fit_predict_multivariate(model, flux_vars, met_train, met_test, met_test_xr, flux_train):
    """Fits a model one output variable at a time """

    # Ditch all of the incomplete data
    qc_index = (~pd.concat([met_train, flux_train], axis=1).isnull()).apply(all, axis=1)
    if qc_index.sum() > 0:
        print("Training {v} using {count} complete samples out of {total}"
              .format(v=flux_vars, count=qc_index.sum(), total=met_train.shape[0]))
    else:
        print("No training data, failing")
        return

    model.fit(X=met_train[qc_index], y=flux_train[qc_index])
    print("Fitting complete.")

    sim_data = model.predict(met_test)
    print("Prediction complete.")

    # TODO: some models return arrays... convert to dicts in that case.
    if isinstance(sim_data, np.ndarray):
        sim_data = {v: sim_data[:, i] for i, v in enumerate(flux_vars)}

    sim_data = sim_dict_to_xr(sim_data, met_test_xr)

    return sim_data


def PLUMBER_fit_predict(model, name, site, multivariate=False):
    """Fit and predict a model

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at
    :returns: xarray dataset of simulation

    """
    if hasattr(model, 'forcing_vars'):
        met_vars = model.forcing_vars
    else:
        print("Warning: no forcing vars, using defaults (all)")
        met_vars = MET_VARS

    flux_vars = ['Qle', 'Qh', 'NEE']

    use_names = isinstance(model, LagWrapper)

    met_train, met_test, met_test_xr, flux_train = \
        get_train_test_sets(site, met_vars, flux_vars, use_names)

    print_good("Running {n} at {s}".format(n=name, s=site))

    print('Fitting and running {f} using {m}'.format(f=flux_vars, m=met_vars))
    if multivariate:
        sim_data = fit_predict_multivariate(model, flux_vars, met_train, met_test, met_test_xr, flux_train)
    else:
        sim_data = fit_predict_univariate(model, flux_vars, met_train, met_test, met_test_xr, flux_train)

    return sim_data


def main_run(model, name, site, multivariate=False):
    """Main function for fitting and running a model.

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at (or 'all', or 'debug')
    """
    sim_dir = 'model_data/{n}'.format(n=name)
    os.makedirs(sim_dir, exist_ok=True)

    nc_file = '{d}/{n}_{s}.nc'.format(d=sim_dir, n=name, s=site)

    sim_data = PLUMBER_fit_predict(model, name, site, multivariate)

    if os.path.exists(nc_file):
        print_warn("Overwriting sim file at {f}".format(f=nc_file))
    else:
        print_good("Writing sim file at {f}".format(f=nc_file))
    sim_data.to_netcdf(nc_file)

    return


def main_run_mp(name, site, no_mp=False, multivariate=False):
    """Multi-processor run handling."""

    model = get_model(name)

    if site in ['all', 'PLUMBER_ext', 'PLUMBER']:
        datasets = get_sites(site)
        if no_mp:
            for s in datasets:
                main_run(model, name, s, multivariate)
        else:
            f_args = [(model, name, s, multivariate) for s in datasets]
            ncores = min(os.cpu_count(), 1 + int(os.cpu_count() * 0.5))
            with Pool(ncores) as p:
                p.starmap(main_run, f_args)
    else:
        main_run(model, name, site, multivariate)

    return


def main(args):
    name = args['<name>']
    site = args['<site>']

    main_run_mp(name, site, args['--no-mp'], args['--multivariate'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
