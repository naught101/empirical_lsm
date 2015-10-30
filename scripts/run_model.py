#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: Fits and runs a basic model and produces rst output with diagnostics

Usage:
    run_model.py run <name> <site>
    run_model.py eval <name> <site> [<file>]

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt

from datetime import datetime as dt

import pandas as pd
import sys
import os
import xray
from matplotlib.cbook import dedent

from pals_utils.constants import DATASETS, MET_VARS, FLUX_VARS
from pals_utils.data import get_site_data, pals_xray_to_df, xray_list_to_df

from ubermodel.models import get_model
from ubermodel.evaluate import evaluate_simulation
from ubermodel.plots import diagnostic_plots
from ubermodel.utils import print_good, dataframe_to_rst
from ubermodel.data import sim_dict_to_xray, get_sim_nc_path


def PLUMBER_fit_predict(model, name, site):
    """Fit and predict a model

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at
    :returns: xray dataset of simulation

    """
    print("Loading all data... ", end='')
    met_data = get_site_data(DATASETS, 'met')
    flux_data = get_site_data(DATASETS, 'flux')

    met_vars = MET_VARS

    flux_vars = FLUX_VARS

    # TODO: fix dirty hack for loading names when required.
    use_names = 'lag' in name

    print("Converting... ", end='')
    met_train = xray_list_to_df([ds for s, ds in met_data.items() if s != site],
                                variables=met_vars, qc=True, name=use_names)

    # We use gap-filled data for the testing period, or the model fails.
    met_test = pals_xray_to_df(met_data[site], variables=met_vars)

    flux_train = xray_list_to_df([ds for s, ds in flux_data.items() if s != site],
                                 variables=flux_vars, qc=True, name=use_names)

    print('Fitting and running {f} using {m}'.format(f=flux_vars, m=met_vars))
    sim_data_dict = dict()
    for v in flux_vars:
        # Might eventually want to update this to run multivariate-out models
        flux_train_v = flux_train[v]

        # Ditch all of the incomplete data
        qc_index = (~pd.concat([met_train, flux_train_v], axis=1).isnull()).apply(all, axis=1)
        if qc_index.sum() > 0:
            print("Training {v} using {count} complete samples out of {total}"
                  .format(v=v, count=qc_index.sum(), total=met_train.shape[0]))
        else:
            print("No training data, skipping variable %s" % v)
            continue

        model.fit(X=met_train[qc_index], y=flux_train_v[qc_index])

        sim_data_dict[v] = model.predict(met_test)

    if len(sim_data_dict) < 1:
        print("No fluxes successfully fitted, quitting")
        sys.exit()

    sim_data = sim_dict_to_xray(sim_data_dict, met_data[site])

    return sim_data


def PLUMBER_fit_predict_eval(model, name, site):
    nc_path = get_sim_nc_path(name, site)
    if os.path.exists(nc_path):
        print_good('{n} already run at {s} - loading from {p}'.format(n=name, s=site, p=nc_path))
        sim_data = xray.open_dataset(nc_path)
    else:
        sim_data = PLUMBER_fit_predict(model, name, site)
        sim_data.to_netcdf(nc_path)

    flux_data = get_site_data([site], 'flux')[site]

    print_good('Evaluating data for {n} at {s}'.format(n=name, s=site))
    eval_results = evaluate_simulation(sim_data, flux_data, name)

    files = diagnostic_plots(sim_data, flux_data, name)

    return eval_results, files


def model_site_rst_format(model, name, site, eval_text, files):
    """format all the datas into an rst!
    """

    date = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    plots = '\n\n'.join([
        ".. image :: {file}".format(file=f) for f in files])

    template = dedent("""
    {name} at {site}
    ====================

    date: :code:`{date}`

    Model details:
    --------------

    :code:`{model}`

    Evaluation results:
    -------------------

    {eval_text}

    Plots:
    ------

    {plots}
    """)

    output = (template.format(model=model,
                              name=name,
                              site=site,
                              plots=plots,
                              date=date,
                              eval_text=eval_text))

    return output


def model_site_rst_write(model, name, site, eval_results, files):
    """run a model and generate an rst file.

    This is useful for importing.

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at
    """
    model_site_rst_file = 'source/models/{n}/{n}_{s}.rst'.format(n=name, s=site)

    print_good("Generating rst file for {n} at {s}.".format(n=name, s=site))

    eval_text = dataframe_to_rst(eval_results)

    output = model_site_rst_format(model, name, site, eval_text, files)

    with open(model_site_rst_file, 'w') as f:
        f.write(output)

    return


def main_run(model, name, site):
    """Main function for fitting, running, and evaluating a model.

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at
    """
    eval_results, files = PLUMBER_fit_predict_eval(model, name, site)

    model_site_rst_write(model, name, site, eval_results, files)

    return


def main_eval(name, site, sim_file=None):
    """Main function for evaluating an existing simulation.

    Copies simulation data to source directory.

    :name: name of the model
    :site: PALS site name to run the model at
    :sim_file: Path to simulation netcdf
    """
    nc_path = get_sim_nc_path(name, site)
    if sim_file is not None:
        sim_data = xray.open_dataset(sim_file)
        # WARNING! over writes existing sim!
        sim_data.to_netcdf(nc_path)
    else:
        sim_data = xray.open_dataset(nc_path)

    flux_data = get_site_data([site], 'flux')[site]

    print_good('Evaluating data for {n} at {s}'.format(n=name, s=site))
    eval_results = evaluate_simulation(sim_data, flux_data, name)

    files = diagnostic_plots(sim_data, flux_data, name)

    model_site_rst_write("Not generated", name, site, eval_results, files)

    return


def main(args):
    # print(args)
    # sys.exit()

    name = args['<name>']
    site = args['<site>']

    if args['run']:
        model = get_model(name)
        if site == 'all':
            for s in DATASETS:
                main_run(model, name, s)
        else:
            main_run(model, name, site)

    elif args['eval']:
        sim_file = args['<file>']
        if site == 'all':
            # will only work if simulations are already run.
            for s in DATASETS:
                main_eval(name, s)
        else:
            main_eval(name, site, sim_file)

    return


if (__name__ == '__main__'):
    args = docopt(__doc__)

    main(args)
