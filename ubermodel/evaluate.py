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
from datetime import datetime

from pals_utils.data import pals_site_name
from pals_utils.stats import run_metrics
from pals_utils.helpers import short_hash


# Evaluate

def evaluate_simulation(sim_data, flux_data, model_name, flux_vars, cache):
    """Top-level simulation evaluator.

    Compares sim_data to flux_data, using standard metrics. Stores the results in an easily accessible format.

    TODO: Maybe get model model_name from sim_data directly (this is a PITA at
          the moment, most models don't report it).
    """

    eval_hash = short_hash((sim_data, flux_data))[0:7]
    eval_time = datetime.now().isoformat()

    # TODO: This currently returns "TumbaFluxnet" - would be nice to return the proper name.
    #       Probably should be fixed in PALS.
    site = pals_site_name(flux_data)

    index = {"eval_hash": "%s_%s" % (eval_hash, flux_vars[0])}

    if "metric_data" in cache:
        metric_data = cache.metric_data
    else:
        metric_data = pd.DataFrame([index])
        metric_data = metric_data.set_index(list(index.keys()))

    for y_var in flux_vars:
        Y_sim = sim_data[y_var].values.ravel()
        Y_obs = flux_data[y_var].values.ravel()

        row_id = "%s_%s" % (eval_hash, y_var)
        metric_data.ix[row_id, "eval_hash"] = eval_hash
        metric_data.ix[row_id, "model_name"] = model_name
        metric_data.ix[row_id, "sim_hash"] = short_hash(sim_data)
        metric_data.ix[row_id, "site"] = site
        metric_data.ix[row_id, "var"] = y_var
        metric_data.ix[row_id, "eval_time"] = eval_time

        for k, v in run_metrics(Y_sim, Y_obs).items():
            metric_data.ix[row_id, k] = v

    cache["metric_data"] = metric_data
    cache.flush()

    return metric_data.query("eval_hash == '%s'" % eval_hash)
