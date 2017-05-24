#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: check_sanity.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Checks existing model output for sanity

Usage:
    check_sanity.py (all|<model>...) [--sites=<sites>]
    check_sanity.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
"""

from docopt import docopt

import os
import xarray as xr
from glob import glob

from ubermodel.utils import print_bad
from ubermodel.data import get_sites
from ubermodel.checks import model_sanity_check


def check_models(models, sites):
    """Checks all models

    :models: list of model names
    """
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
                else:
                    print('.', end='', flush=True)
        print('')

    return


def main(args):

    if args['--sites'] is None:
        sites = get_sites('PLUMBER_ext')
    else:
        sites = args['--sites'].split(',')

    if args['all']:
        models = [os.path.basename(f) for f in glob('model_data/*')]
    else:
        models = args['<model>']

    check_models(models, sites)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
