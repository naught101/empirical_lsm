#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: plot_dataset.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: TODO: File description

Usage:
    plot_dataset.py year <filename> <variable> <year>
    plot_dataset.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
    --option=<n>  Option description [default: 3]
"""

from docopt import docopt

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import basemap


def get_range(x):
    return np.quantile(x, [0, 100])


def plot_array(da, ax=None, shift=True):
    """plots an array of lat by lon on a coastline map"""

    m = basemap.Basemap()
    m.drawcoastlines()
    m.pcolormesh(m.pcolormesh(x=da.lon, y=da.lat, data=da.T, latlon=True))

    return m


def get_filename(benchmark, variable, year):
    template = "data/gridded_benchmarks/{b}/{b}_{v}_{y}.nc"
    filename = template.format(b=benchmark, v=variable, y=year)

    return filename


def plot_all(benchmark, variable, year):
    """TODO: Docstring for .
    :returns: TODO

    """

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']

    filename = get_filename(benchmark, variable)

    # load file
    with xr.open_dataset(filename) as ds:
        da = ds[variable]
        # da.where((-1e8 < da) & (da < 1e8))

        # monthly averages
        monthly_mean = da.groupby('time.month').mean(dim='time')
        monthly_std = da.groupby('time.month').std(dim='time')
        annual_std = da.std(dim='time')

    annual_mean = monthly_mean.mean(dim='month')

    crange = [-50, 150]  # estimate with some leeway

    fig, axes = plt.subplots(nrows=3, ncols=4)
    for i, m in enumerate(months):
        # mean plot per month
        ax = axes.flat[i]
        plot_array(monthly_mean.sel(month=i + 1), ax=ax)
        plt.title(m)
    plt.suptitle("{b} - {y} monthly means".format(b=benchmark, y=year))
    plt.savefig("plots/montly_means/{y}/{b}".format(b=benchmark, y=year))
    plt.colorbar(axes.flat)

    fig, axes = plt.subplots(nrows=3, ncols=4)
    for i, m in enumerate(months):
        # stddev plot per month
        ax = axes.flat[i]
        plot_array(monthly_std.sel(month=i + 1), ax=ax)
        plt.title(m)
    plt.suptitle("{b} - {y} monthly std devs")
    plt.savefig("plots/montly_stds/{y}/{b}".format(b=benchmark, y=year))

    # arrange/save plots


def main(args):

    plot_all(args['<filename'], args['<variable>'], args['<year>'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
