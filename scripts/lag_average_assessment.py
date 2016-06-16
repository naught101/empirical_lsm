#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: lag_average_assessment.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: assess lagg structure of flux/met data

Usage:
    lag_average_assessment.py
    lag_average_assessment.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
"""

from docopt import docopt

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re

import pals_utils.data as pud


def rolling_window(a, rows):
    """from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html"""
    shape = a.shape[:-1] + (a.shape[-1] - rows + 1, rows)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_to_rows(window, datafreq=0.5):
    """calculate number of rows for window

    :window: window of the format "30min", "3h"
    :datafreq: data frequency in hours
    :returns: number of rows

    """
    n, freq = re.match('(\d*)([a-zA-Z]*)', window).groups()
    n = int(n)
    if freq == "min":
        rows = n / (60 * datafreq)
    elif freq == "h":
        rows = n / datafreq
    elif freq == "d":
        rows = n * 24 / datafreq
    else:
        raise 'Unknown frequency "%s"' % freq

    assert rows == int(rows), "window doesn't match data frequency - not integral result"

    return int(rows)


def rolling_mean(data, window, datafreq=0.5):
    """calculate rolling mean for an array

    :data: ndarray
    :window: time span, e.g. "30min", "2h"
    :datafreq: data frequency in hours
    :returns: data in the same shape as the original, with leading NaNs
    """
    rows = window_to_rows(window, datafreq)

    result = np.full_like(data, np.nan)

    np.mean(rolling_window(data.T, rows), -1, out=result[(rows - 1):].T)
    return result


def main(args):

    sites = ['Tumba']

    met_vars = ['Rainf']
    # flux_vars = ['Qle', 'Qh']

    met_data = pud.get_met_df(sites, met_vars, qc=True)
    # flux_data = pud.get_flux_df(sites, flux_vars, qc=True)

    lags = [('30min'),
            ('1h'),
            ('2h'),
            ('3h'),
            ('4h'),
            ('5h'),
            ('6h'),
            ('12h'),
            ('1d'),
            ('2d'),
            ('3d'),
            ('5d'),
            ('7d'),
            ('14d'),
            ('30d'),
            ('60d'),
            ('90d'),
            ('180d'),
            ('365d')][:-2]

    lagged_data = pd.DataFrame(np.concatenate([rolling_mean(met_data[['Rainf']].values, l) for l in lags], axis=1), columns=lags)

    fig = plt.imshow(lagged_data.corr(), interpolation='none')
    fig.axes.set_xticks(range(len(lagged_data.columns)))
    fig.axes.set_yticks(range(len(lagged_data.columns)))
    fig.axes.set_xticklabels(lagged_data.columns)
    fig.axes.set_yticklabels(lagged_data.columns)
    plt.colorbar()
    plt.title("Rainfall lagged averages' correlation")

    plt.savefig('plots/Rainf_lagged_avg_correlations.png')

if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
