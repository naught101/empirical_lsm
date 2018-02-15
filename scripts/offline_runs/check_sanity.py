#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: check_sanity.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/empirical_lsm
Description: Checks existing model output for sanity

Usage:
    check_sanity.py (data|metrics) (all|<model>...) [--sites=<sites>]
                    [--re-run] [--re-eval]
                    [--delete-sims] [--delete-eval]
    check_sanity.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
"""

from docopt import docopt

import os
from glob import glob

from empirical_lsm.data import get_sites
from empirical_lsm.checks import check_model_data, check_metrics

from empirical_lsm.offline_simulation import run_model_site_tuples_mp
from empirical_lsm.offline_eval import eval_simulation

from pals_utils.logging import setup_logger
logger = setup_logger(None, 'logs/check_sanity.log')


def main(args):
    if args['--sites'] is None:
        sites = get_sites('PLUMBER_ext')
    else:
        try:
            sites = get_sites(args['--sites'])
        except:
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

    if len(bad_sims) > 0:
        if args['--delete-sims']:
            for m, s in bad_sims:
                os.remove('model_data/{m}/{m}_{s}.nc'.format(m=m, s=s))
            summary += " and deleted sims"

        if args['--delete-eval']:
            for m, s in bad_sims:
                os.remove('source/models/{m}/metrics/{m}_{s}_metrics.csv'.format(m=m, s=s))
            summary += " and deleted evals"

        if args['--re-run']:
            run_model_site_tuples_mp(bad_sims)
            summary += " and re-run"

        if args['--re-eval']:
            [eval_simulation(t[0], t[1]) for t in bad_sims]
            summary += " and re-evaluated"

    logger.info(summary + ".")

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
