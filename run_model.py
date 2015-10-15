#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: Fits and runs a basic model and produces rst output with diagnostics

Usage:
    run_model.py <name> <site>

Options:
    -h, --help  Show this screen and exit.
    --baud=<n>  Baudrate [default: 9600]
"""

from docopt import docopt

from datetime import datetime as dt

import pandas as pd
import sys
import os
import xray
from matplotlib.cbook import dedent

from pals_utils.constants import DATASETS, MET_VARS, FLUX_VARS
from pals_utils.data import get_site_data, pals_xray_to_df, xray_list_to_df, copy_data

from ubermodel.models import get_model
from ubermodel.evaluate import evaluate_simulation
from ubermodel.plots import diagnostic_plots


def sim_dict_to_xray(sim_dict, old_ds):
    """Converts a dictionary of arrays into a xray dataset with the same geo data as old_ds

    :sim_dict: TODO
    :old_ds: TODO
    :returns: TODO

    """
    sim_data = copy_data(old_ds)

    for v in sim_dict:
        sim_var = sim_dict[v]
        sim_var.shape = (sim_var.shape[0], 1, 1)
        sim_array = xray.DataArray(sim_var, dims=['time', 'y', 'x'],
                                   coords=dict(time=old_ds.coords['time'], y=[1.0], x=[1.0]))
        sim_data[v] = sim_array

    return sim_data


def PLUMBER_fit_predict(model, name, site):
    """Fit and predict a model

    :model: TODO
    :name: TODO
    :site: TODO
    :returns: TODO

    """
    print("Loading all data... ", end='')
    met_data = get_site_data(DATASETS, 'met')
    flux_data = get_site_data(DATASETS, 'flux')

    met_vars = MET_VARS.copy()
    met_vars.remove('LWdown')
    met_vars.remove('PSurf')

    flux_vars = FLUX_VARS

    print("Converting... ", end='')
    met_train = xray_list_to_df([ds for s, ds in met_data.items() if s != site],
                                variables=met_vars, qc=True)

    # We use gap-filled data for the testing period, or the model fails.
    met_test = pals_xray_to_df(met_data[site], variables=met_vars)

    flux_train = xray_list_to_df([ds for s, ds in flux_data.items() if s != site],
                                 variables=flux_vars, qc=True)

    print('Fitting and running {0} using {1}'.format(flux_vars, met_vars))
    sim_data_dict = dict()
    for v in flux_vars:
        flux_train_v = flux_train[[v]]

        # Ditch all of the incomplete data
        qc_index = (~pd.concat([met_train, flux_train_v], axis=1).isnull()).apply(all, axis=1)
        if qc_index.sum() > 0:
            print("Training %s using %d complete samples out of %d" %
                  (v, qc_index.sum(), met_train.shape[0]))
        else:
            print("No training data, skipping variable %s" % v)
            continue

        model.fit(X=met_train[qc_index], y=flux_train_v[qc_index])

        sim_data_dict[v] = model.predict(met_test)

        # flux_test = pals_xray_to_df(flux_data[site], variables=[v])
        # evaluate_simulation(sim_data, flux_test, name)

    if len(sim_data_dict) < 1:
        print("No fluxes successfully fitted, quitting")
        sys.exit()

    sim_data = sim_dict_to_xray(sim_data_dict, met_data[site])

    return sim_data


def PLUMBER_fit_predict_eval(model, name, site):
    model_path = 'source/models/{0}/'.format(name)
    nc_path = '{0}{1}.nc'.format(model_path, name)
    if os.path.exists(nc_path):
        sim_data = xray.open_dataset(nc_path)
    else:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        sim_data = PLUMBER_fit_predict(model, name, site)
        sim_data.to_netcdf(nc_path)

    flux_data = get_site_data([site], 'flux')[site]

    files = diagnostic_plots(sim_data, flux_data, name)

    return files


def rst_output(model, name, site, files):

    date = dt.isoformat(dt.now(), sep=' ')

    plots = '\n\n'.join([
        ".. image :: {file}".format(file=f) for f in files])

    template = dedent("""
    {name} at {site}
    ====================

    date: {date}

    Model details:
    --------------

    {model}


    Plots:
    ------

    {plots}
    """)

    output = (template.format(model=model,
                              name=name,
                              site=site,
                              plots=plots,
                              date=date))

    return output


def rst_gen(model, name, site):
    """run a model and generate an rst file.

    This is useful for importing.

    :model: TODO
    :name: TODO
    :site: TODO
    :returns: TODO

    """
    rst_file = 'source/models/{name}/{site}.rst'.format(
        name=name,
        site=site)

    print("Generating rst file for {0} at {1}.".format(name, site))

    files = PLUMBER_fit_predict_eval(model, name, site)

    output = rst_output(model, name, site, files)

    with open(rst_file, 'w') as f:
        f.write(output)


def main(args):
    name = args['<name>']
    model = get_model(name)

    site = args['<site>']

    rst_gen(model, name, site)

    return


if (__name__ == '__main__'):
    args = docopt(__doc__)

    main(args)
