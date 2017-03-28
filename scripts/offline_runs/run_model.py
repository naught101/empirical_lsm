#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: Fits and runs a basic model and produces rst output with diagnostics

Usage:
    run_model.py run <name> <site> [--no-mp] [--multivariate] [--overwrite] [--no-fix-closure]

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime as dt

from multiprocessing import Pool

from pals_utils.constants import MET_VARS

from ubermodel.transforms import LagWrapper
from ubermodel.models import get_model
from ubermodel.data import get_sites, sim_dict_to_xr, get_train_test_sets
from ubermodel.utils import print_good, print_warn


def fit_predict_univariate(model, flux_vars, met_train, met_test, met_test_xr, flux_train):
    """Fits a model one output variable at a time """
    sim_data_dict = dict()
    for v in flux_vars:
        # TODO: Might eventually want to update this to run multivariate-out models
        # There isn't much point right now, because there is almost no data where all variables are available.
        flux_train_v = flux_train[[v]]

        if hasattr(model, 'partial_data_ok'):
            # model accepts partial data
            print("Training {v} using all (possibly incomplete) data.".format(v=v))
            model.fit(X=met_train, y=flux_train_v)
        else:
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
    """Fits a model multiple outputs variable at once"""

    if hasattr(model, 'partial_data_ok'):
            # model accepts partial data
        print("Training {v} using all (possibly incomplete) data.".format(v=flux_vars))
        model.fit(X=met_train, y=flux_train)
    else:
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


def PLUMBER_fit_predict(model, name, site, multivariate=False, fix_closure=True):
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
        get_train_test_sets(site, met_vars, flux_vars, use_names, fix_closure=True)

    print_good("Running {n} at {s}".format(n=name, s=site))

    print('Fitting and running {f} using {m}'.format(f=flux_vars, m=met_vars))
    t_start = dt.now()
    if multivariate:
        sim_data = fit_predict_multivariate(model, flux_vars, met_train, met_test, met_test_xr, flux_train)
    else:
        sim_data = fit_predict_univariate(model, flux_vars, met_train, met_test, met_test_xr, flux_train)
    run_time = str(dt.now() - t_start).split('.')[0]

    sim_data.attrs.update(
        model_name=name,
        site=site,
        forcing_vars=met_vars,
        fit_predict_time=run_time
    )

    return sim_data


def main_run(model, name, site, multivariate=False, overwrite=False, fix_closure=True):
    """Main function for fitting and running a model.

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at (or 'all', or 'debug')
    """
    sim_dir = 'model_data/{n}'.format(n=name)
    os.makedirs(sim_dir, exist_ok=True)

    nc_file = '{d}/{n}_{s}.nc'.format(d=sim_dir, n=name, s=site)

    if os.path.isfile(nc_file) and not overwrite:
        print_warn("Sim netcdf already exists for {n} at {s}, use --overwrite to re-run."
                   .format(n=name, s=site))
        return
    else:
        sim_data = PLUMBER_fit_predict(model, name, site,
                                       multivariate=multivariate, fix_closure=fix_closure)

        if os.path.isfile(nc_file):
            print_warn("Overwriting sim file at {f}".format(f=nc_file))
        else:
            print_good("Writing sim file at {f}".format(f=nc_file))

        if site != 'debug':
            sim_data.to_netcdf(nc_file)

        return


def main_run_mp(name, site, no_mp=False, multivariate=False, overwrite=False, fix_closure=True):
    """Multi-processor run handling."""

    model = get_model(name)

    if site in ['all', 'PLUMBER_ext', 'PLUMBER']:
        print_good('Running {n} at {s} sites'.format(n=name, s=site))
        datasets = get_sites(site)
        if no_mp:
            for s in datasets:
                main_run(model, name, s, multivariate, overwrite, fix_closure)
        else:
            f_args = [(model, name, s, multivariate, overwrite, fix_closure) for s in datasets]
            ncores = min(os.cpu_count(), 1 + int(os.cpu_count() * 0.5))
            with Pool(ncores) as p:
                p.starmap(main_run, f_args)
    else:
        main_run(model, name, site, multivariate, overwrite, fix_closure)

    return


def main(args):
    name = args['<name>']
    site = args['<site>']

    main_run_mp(name, site,
                no_mp=args['--no-mp'],
                multivariate=args['--multivariate'],
                overwite=args['--overwrite'],
                fix_closure=not args['--no-fix-closure'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
