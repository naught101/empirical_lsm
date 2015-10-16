#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: evaluation.py
Author: ned haughton
Email: ned@nedhaughton.com
Github:
Description: PALS-style model evaluation
"""

import pandas as pd

from pals_utils.data import pals_site_name, FLUX_VARS
from pals_utils.stats import run_metrics


# Evaluate

def evaluate_simulation(sim_data, flux_data, name):
    """Top-level simulation evaluator.

    Compares sim_data to flux_data, using standard metrics.

    TODO: Maybe get model model_name from sim_data directly (this is a PITA at
          the moment, most models don't report it).
    """

    eval_vars = list(set(FLUX_VARS).intersection(sim_data.data_vars)
                                   .intersection(flux_data.data_vars))

    metric_data = pd.DataFrame()
    metric_data.index.name = 'metric'
    for v in eval_vars:
        sim_v = sim_data[v].values.ravel()
        obs_v = flux_data[v].values.ravel()

        for m, val in run_metrics(sim_v, obs_v).items():
            metric_data.ix[m, v] = val

    site = pals_site_name(flux_data)

    eval_path = 'source/models/{0}/{1}.csv'.format(name, site)
    metric_data.to_csv(eval_path)

    return metric_data
