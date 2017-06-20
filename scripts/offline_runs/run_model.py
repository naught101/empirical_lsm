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
import math
import psutil
import pickle
from datetime import datetime as dt

from multiprocessing import Pool

from ubermodel.transforms import LagWrapper, LagAverageWrapper
from ubermodel.models import get_model
from ubermodel.data import get_sites, sim_dict_to_xr, get_train_test_data
from ubermodel.utils import print_good, print_warn, print_bad
from ubermodel.checks import model_sanity_check, run_var_checks

from pals_utils.data import set_config, get_config


set_config(['vars', 'flux'], ['NEE', 'Qle', 'Qh'])


def get_suitable_ncores():
    return min(os.cpu_count(), 1 + int(os.cpu_count() * 0.5))


def bytes_human_readable(n):
    if (n == 0):
        return '0B'
    div = 10 ** 3
    exp = int(math.log(n, div))
    suffix = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB'][exp]
    signif = '%.1f' % (n / (div ** exp))

    return(signif + suffix)


def fit_predict_univariate(model, flux_vars, train_test_data):
    """Fits a model one output variable at a time """
    sim_data_dict = dict()
    for v in flux_vars:
        # TODO: Might eventually want to update this to run multivariate-out models
        # There isn't much point right now, because there is almost no data where all variables are available.
        flux_train_v = train_test_data["flux_train"][[v]]

        if hasattr(model, 'partial_data_ok'):
            # model accepts partial data
            print("Training {v} using all (possibly incomplete) data.".format(v=v))
            model.fit(X=train_test_data["met_train"], y=flux_train_v)
        else:
            # Ditch all of the incomplete data
            qc_index = (~pd.concat([train_test_data["met_train"], flux_train_v], axis=1).isnull()).apply(all, axis=1)
            if qc_index.sum() > 0:
                print("Training {v} using {count} complete samples out of {total}"
                      .format(v=v, count=qc_index.sum(), total=train_test_data["met_train"].shape[0]))
            else:
                print("No training data, skipping variable %s" % v)
                continue

            model.fit(X=train_test_data["met_train"][qc_index], y=flux_train_v[qc_index])
        print("Fitting complete.")

        sim_data_dict[v] = model.predict(train_test_data["met_test"])
        print("Prediction complete.")

        if run_var_checks(sim_data_dict[v]):
            name = "%s_%s_%s" % (model.name, v, train_test_data["site"])
            save_model_structure(model, name)

    if len(sim_data_dict) < 1:
        print("No fluxes successfully fitted, quitting")
        sys.exit()

    sim_data = sim_dict_to_xr(sim_data_dict, train_test_data["met_test_xr"])

    return sim_data


def fit_predict_multivariate(model, flux_vars, train_test_data):
    """Fits a model multiple outputs variable at once"""

    if hasattr(model, 'partial_data_ok'):
            # model accepts partial data
        print("Training {v} using all (possibly incomplete) data.".format(v=flux_vars))
        model.fit(X=train_test_data["met_train"], y=train_test_data["flux_train"])
    else:
        # Ditch all of the incomplete data
        qc_index = (~pd.concat([train_test_data["met_train"], train_test_data["flux_train"]], axis=1).isnull()).apply(all, axis=1)
        if qc_index.sum() > 0:
            print("Training {v} using {count} complete samples out of {total}"
                  .format(v=flux_vars, count=qc_index.sum(), total=train_test_data["met_train"].shape[0]))
        else:
            print("No training data, failing")
            return

        model.fit(X=train_test_data["met_train"][qc_index], y=train_test_data["flux_train"][qc_index])
    print("Fitting complete.")

    sim_data = model.predict(train_test_data["met_test"])
    print("Prediction complete.")

    # TODO: some models return arrays... convert to dicts in that case.
    if isinstance(sim_data, np.ndarray):
        sim_data = {v: sim_data[:, i] for i, v in enumerate(flux_vars)}

    sim_data = sim_dict_to_xr(sim_data, train_test_data["met_test_xr"])

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
        met_vars = get_config(['vars', 'met'])

    use_names = isinstance(model, (LagWrapper, LagAverageWrapper))

    train_test_data = get_train_test_data(site, met_vars, get_config(['vars', 'flux']), use_names, fix_closure=True)

    print_good("Running {n} at {s}".format(n=name, s=site))

    print('Fitting and running {f} using {m}'.format(f=get_config(['vars', 'flux']), m=met_vars))
    t_start = dt.now()
    if multivariate:
        sim_data = fit_predict_multivariate(model, get_config(['vars', 'flux']), train_test_data)
    else:
        sim_data = fit_predict_univariate(model, get_config(['vars', 'flux']), train_test_data)
    run_time = str(dt.now() - t_start).split('.')[0]

    process = psutil.Process(os.getpid())
    mem_usage = bytes_human_readable(process.memory_info().rss)

    print("Model fit and run in %s, using %s memory." % (run_time, mem_usage))

    sim_data.attrs.update({
        "Model_name": name,
        "Model_description": str(model),
        "PALS_site": site,
        "Forcing_vars": ', '.join(met_vars),
        "Fit_predict_time": run_time,
        "Fit_predict_mem_usage": mem_usage,
        "Production_time": str(dt.now()),
        "Production_source": "Ubermodel offline run scripts"
    })

    return sim_data


def save_model_structure(model, name='None'):
    if name is None:
        name = model.name
    log_dir = 'logs/models/' + name
    os.makedirs(log_dir, exist_ok=True)

    now = dt.now().strftime('%Y%m%d_%H%M%S')
    pickle_file = "%s/%s-%s.pickle" % (log_dir, name, now)
    with open(pickle_file, 'wb') as f:
        pickle.dump(model, f)
    print_warn('Model structure saved to ' + pickle_file)

    return


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

    for i in range(3):
        # We attempt to run the model up to 3 times, incase of numerical problems
        sim_data = PLUMBER_fit_predict(model, name, site,
                                       multivariate=multivariate, fix_closure=fix_closure)

        try:
            model_sanity_check(sim_data, name, site)
        except RuntimeError as e:
            print_warn(str(e))

            if i < 2:
                print_warn('Attempting a %s run.' % ['2nd', '3rd'][i])
                continue
            else:
                print_bad('Giving up after 3 failed runs. Check your model structres or met data.')
                sim_data.attrs.update({'Warning': 'model failed after 3 attempts, saved anyway'})
        else:
            # model run successful, presumably
            break

    if os.path.isfile(nc_file):
        print_warn("Overwriting sim file at {f}".format(f=nc_file))
    else:
        print_good("Writing sim file at {f}".format(f=nc_file))

    # if site != 'debug':
    sim_data.to_netcdf(nc_file)

    return


def run_model_site_tuples_mp(tuples_list, no_mp=False, multivariate=False, overwrite=False, fix_closure=True):
    """Run (model, site) pairs"""

    f_args = [(get_model(t[0]), t[0], t[1], multivariate, overwrite, fix_closure) for t in tuples_list]

    if no_mp:
        [main_run(*args) for args in f_args]
    else:  # multiprocess
        ncores = get_suitable_ncores()
        # TODO: Deal with memory requirement?
        with Pool(ncores) as p:
            p.starmap(main_run, f_args)


def main_run_mp(name, site, no_mp=False, multivariate=False, overwrite=False, fix_closure=True):
    """Multi-processor run handling."""
    # TODO: refactor to work with above caller.

    model = get_model(name)

    if site in ['all', 'PLUMBER_ext', 'PLUMBER']:
        print_good('Running {n} at {s} sites'.format(n=name, s=site))
        datasets = get_sites(site)
        if no_mp:
            for s in datasets:
                main_run(model, name, s, multivariate, overwrite, fix_closure)
        else:
            f_args = [(model, name, s, multivariate, overwrite, fix_closure) for s in datasets]
            ncores = get_suitable_ncores()
            if site is not 'debug' and hasattr(model, 'memory_requirement'):
                ncores = max(1, int((psutil.virtual_memory().total / 2) // model.memory_requirement))
            print("Running on %d core(s)" % ncores)

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
