#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: plot_annual_cycle.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: TODO: File description

Usage:
    plot_annual_cycle.py <filename> <variable>
    plot_annual_cycle.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
    --option=<n>  Option description [default: 3]
"""

from docopt import docopt

import xarray as xr
import cartopy as cp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_range(x):
    x = x.where(x > -10e8).values

    return np.quantile(x, [0, 95])


def plot_annual_cycle(filename, variable):
    """TODO: Docstring for .
    :returns: TODO

    """

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
    # Assumes 3-hourly
    month_starts = 4 * np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    fig = plt.figure()

    # load file
    with xr.open_dataset(filename) as ds:

        crange = get_range(ds['Qh'])
        for i, m in enumerate(months):
            ax = fig.add_subplot(3, 4, i + 1)
            # monthly averages
            data = ds[variable].isel(time=slice(month_starts[i], month_starts[i + 1])).mean('time')
            data.where(data > -10e8).T.plot(ax=ax, vmin=crange[0], vmax=crange[1])

    # plot per month
    # arrange/save plots


def main(args):

    plot_annual_cycle(args['<filename'], args['<variable>'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
