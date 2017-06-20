#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: check_sanity.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Checks existing model output for sanity

Usage:
    check_sanity.py data (all|<model>...) [--sites=<sites>] [--re-run] [--re-eval]
    check_sanity.py metrics (all|<model>...) [--sites=<sites>] [--re-run] [--re-eval]
    check_sanity.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
"""

from docopt import docopt

import os
import pandas as pd
import xarray as xr
from glob import glob

from ubermodel.utils import print_bad
from ubermodel.data import get_sites
from ubermodel.checks import model_sanity_check

from ubermodel.offline_simulation import run_model_site_tuples_mp
from ubermodel.offline_eval import eval_simulation


def check_model_data(models, sites):
    """Checks all models

    :models: list of model names
    """
    bad_simulations = []

    print("Checking {nm} models at {ns} sites.".format(nm=len(models), ns=len(sites)))
    for model in models:
        print('Checking {m}:'.format(m=model))
        for site in sites:
            file_path = 'model_data/{m}/{m}_{s}.nc'.format(m=model, s=site)
            if not os.path.exists(file_path):
                # print('\nmissing model run: {m} at {s}'.format(m=model, s=site))
                print('x', end='', flush=True)
                continue
            with xr.open_dataset(file_path) as ds:
                try:
                    model_sanity_check(ds, model, site)
                except RuntimeError as e:
                    print_bad('\n' + str(e))
                    bad_simulations.append((model, site))
                else:
                    print('.', end='', flush=True)
        print('')

    return bad_simulations


def check_metrics(models, sites):
    """Checks metrics to see if they're bullshit

    :models: TODO
    :sites: TODO
    :returns: TODO

    """
    # Glob all metric filenames

    bad_simulations = []

    for model in models:
        for site in sites:
            csv_path = 'source/models/{m}/metrics/{m}_{s}_metrics.csv'.format(m=model, s=site)
            if not os.path.exists(csv_path):
                continue

            metrics = pd.read_csv(csv_path, index_col=0)
            if ((metrics > 500).any().any() or
                    (metrics.loc['corr'] > 1).any() or
                    (metrics.loc['corr'] < -1).any() or
                    (metrics.loc['overlap'] > 1).any() or
                    (metrics.loc['overlap'] < 0).any()):
                print_bad("Crazy value for {m} at {s}".format(m=model, s=site))
                bad_simulations.append((model, site))

    return bad_simulations


def main(args):

    if args['--sites'] is None:
        sites = get_sites('PLUMBER_ext')
    else:
        sites = args['--sites'].split(',')

    if args['all']:
        if args['data']:
            models = [os.path.basename(f) for f in glob('model_data/*')]
        else:
            models = [os.path.basename(f) for f in glob('source/models/*')]
    else:
        models = args['<model>']

    if args['data']:
        bad_sims = check_model_data(models, sites)
        summary = "%d model with bad data out of %d models checked" % (len(bad_sims), len(models))
    if args['metrics']:
        bad_sims = check_metrics(models, sites)
        summary = "%d model with bad metrics out of %d models checked" % (len(bad_sims), len(models))

    if args['--re-run'] and len(bad_sims) > 0:
        run_model_site_tuples_mp(bad_sims)
        summary += " and re-run"

    if args['--re-eval'] and len(bad_sims) > 0:
        [eval_simulation(t[0], t[1]) for t in bad_sims]
        summary += " and re-evaluated"

    print(summary + ".")

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
