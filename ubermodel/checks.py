#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: checks.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: model sanity checks.
"""

import numpy as np


flux_vars = ['NEE', 'Qh', 'Qle']


def check_var_too_low(data):
    if (data.min() < -1500):
        print("data too low!")
        return True
    else:
        return False


def check_var_too_high(data):
    if (data.max() > 5000):
        print("data too high!")
        return True
    else:
        return False


def check_var_change_too_fast(data):
    if (np.abs(np.diff(data)) > 1500):
        print("data changing too fast!")
        return True
    else:
        return False


def run_var_checks(data):
    return (check_var_too_low(data) or
            check_var_too_high(data) or
            check_var_change_too_fast(data))


def model_sanity_check(sim_data, name, site):
    """Checks a model's output for clearly incorrect values, warns the user,
    and saves debug output

    :sim_data: xarray dataset with flux output
    :name: model name
    :site: site name
    """
    warning = ""
    for v in flux_vars:
        if v not in sim_data:
            warning = v + " missing"

    if warning == "":
        for v in flux_vars:
            # Check output sanity
            if check_var_too_low(sim_data[v].values):
                warning = v + " too low"
                break
            if check_var_too_high(sim_data[v].values):
                warning = v + " too high"
                break

    if warning == "":
        sim_diff = sim_data.diff('time')
        for v in flux_vars:
            # Check rate-of-change sanity
            if (abs(sim_diff[v].values).max() > 500):
                warning = v + " changing rapidly"
                break

    if warning != "":
        warning = "Probable bad model output: {w} at {s} for {n}".format(
            w=warning, s=site, n=name)
        raise RuntimeError(warning)

    return
