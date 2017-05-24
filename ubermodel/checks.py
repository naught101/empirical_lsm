#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: checks.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: model sanity checks.
"""


flux_vars = ['NEE', 'Qh', 'Qle']


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
            if (sim_data[v].values.min() < -300):
                warning = v + " too low"
                break
            if (sim_data[v].values.max() > 1000):
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
