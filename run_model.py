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

from pals_utils.constants import DATASETS, MET_VARS, FLUX_VARS
from pals_utils.data import get_site_data, pals_xray_to_df, xray_list_to_df, copy_data

from ubermodel.models import get_model
from ubermodel.evaluate import evaluate_simulation
from ubermodel.plots import diagnostic_plots


def PLUMBER_fit_predict_eval(model, name, site):
    met_data = get_site_data(DATASETS, 'met')
    flux_data = get_site_data(DATASETS, 'flux')

    met_vars = MET_VARS.copy()
    met_vars.remove('LWdown')
    met_vars.remove('PSurf')

    met_train = xray_list_to_df([ds for s, ds in met_data.items() if s != site],
                                variables=met_vars, qc=True)
    met_test = pals_xray_to_df(met_data[site], variables=met_vars, qc=True)

    for v in FLUX_VARS:
        flux_train = xray_list_to_df([ds for s, ds in flux_data.items()
                                      if s != site and v in list(ds.data_vars)],
                                     variables=[v], qc=True)
        flux_test = pals_xray_to_df(flux_data[site], variables=[v], qc=True)

        qc_index = (~pd.concat([met_train, flux_train]).isnull()).apply(all, axis=1)
        if qc_index.sum() > 0:
            print("Training using %d complete samples out of %d" % (qc_index.sum(), met_train.shape[0]))
        else:
            print("No training data, skipping variable %s" % v)
            continue

        model.fit(X=met_train[qc_index], y=flux_train[qc_index])

        sim_data = copy_data(met_test)

        sim_data[v] = model.predict(met_test)

        # evaluate_simulation(sim_data, flux_test, name)

    files = diagnostic_plots(sim_data, flux_data, name)

    return files


def rst_output(model, name, site, files):
    print(dt.isoformat(dt.now(), sep=' '))

    plots = '\n\n'.join([
        ".. image :: {file}".format(file=f) for f in files])

    template = """
    {name} at {site}
    ====================

    Model details:
    --------------

    {model}


    Plots:
    ------

    {plots}
    """

    output = (template.format(model=model,
                              name=name,
                              site=site,
                              plots=plots))

    return output


def main(args):
    name = args['<name>']
    model = get_model(name)

    site = args['<site>']

    files = PLUMBER_fit_predict_eval(model, name, site)

    rst_file = 'source/{model}/{site}.rst'.format(
        model=model,
        site=site)

    output = rst_output(model, name, site, files)

    with open(rst_file, 'w') as f:
        f.write(output)

    return


if (__name__ == '__main__'):
    args = docopt(__doc__)

    main(args)
