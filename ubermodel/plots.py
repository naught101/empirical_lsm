#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: plots.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description: diagnostic plots for evaluating models
"""

import matplotlib.pyplot as pl
import seaborn as sns
import pandas as pd
import os

from pals_utils.data import pals_site_name, pals_xray_to_df


def diagnostic_plots(sim_data, flux_data, name):
    """Plot standard diagnostic plots

    :sim_data: TODO
    :flux_data: TODO
    :name: TODO
    :returns: TODO

    """
    site = pals_site_name(flux_data)

    base_path = os.path.join('source', name, site, 'figures')
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    files = []

    # TODO: For variables
    for var in ['Qh']:
        plot_monthly_timeseries(sim_data, flux_data, var, name)
        filename = '{0}_{1}_monthly_timeseries.png'.format(name, site)
        plot_path = os.path.join(base_path, filename)
        files.append(plot_path)
        pl.savefig(plot_path)

    return files


def plot_monthly_timeseries(sim_data, flux_data, var, name):
    data = pd.concat([pals_xray_to_df(sim_data, [var]), pals_xray_to_df(flux_data, [var])], axis=1)
    data.columns = ['modelled', 'observed']

    return data.plot()
