#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: run_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: Fits and runs a basic model and produces rst output with diagnostics

Usage:
    run_model.py run <name> <site>

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt


import pandas as pd
import sys

from pals_utils.constants import DATASETS, MET_VARS, FLUX_VARS
from pals_utils.data import get_site_data, pals_xray_to_df, xray_list_to_df

from ubermodel.models import get_model
from ubermodel.data import sim_dict_to_xray


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


def main_run(model, name, site):
    """Main function for fitting and running a model.

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at
    """
    PLUMBER_fit_predict(model, name, site)

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
    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
