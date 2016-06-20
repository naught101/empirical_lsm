#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: plot_dataset.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: TODO: File description

Usage:
    plot_dataset.py year <benchmark> <variable> <year>
    plot_dataset.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
"""

from docopt import docopt

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import basemap
from mpl_toolkits.axes_grid1 import ImageGrid


def get_range(x):
    return np.quantile(x, [0, 100])


def plot_array(da, ax=None, shift=True):
    """plots an array of lat by lon on a coastline map"""

    m = basemap.Basemap()
    m.drawcoastlines()
    m.pcolormesh(da.lon, y=da.lat, data=da.T, latlon=True)

    return m


def get_filename(benchmark, variable, year):
    template = "data/gridded_benchmarks/{b}/{b}_{v}_{y}.nc"
    filename = template.format(b=benchmark, v=variable, y=year)

    return filename


def plot_year(benchmark, variable, year):
    """Plots annual and monthly means and std devs.
    """

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']

    filename = get_filename(benchmark, variable, year)

    # load file
    with xr.open_dataset(filename) as ds:
        da = ds[variable]
        # da.where((-1e8 < da) & (da < 1e8))

        # monthly averages
        monthly_mean = da.groupby('time.month').mean(dim='time').copy()
        monthly_std = da.groupby('time.month').std(dim='time').copy()
        annual_std = da.std(dim='time').copy()

    annual_mean = monthly_mean.mean(dim='month')

    # crange = [-50, 150]  # estimate with some leeway

    print("Plotting monthly mean grid")
    fig = plt.figure(0, (14, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 4), axes_pad=0.3,
                     cbar_mode='single', cbar_location="bottom",)
    for i, m in enumerate(months):
        plt.sca(grid[i])
        plot_array(monthly_mean.sel(month=i + 1))
        plt.title(m)
    plt.suptitle("{b} - {y} monthly means".format(b=benchmark, y=year))
    # plt.colorbar()
    plt.tight_layout()

    os.makedirs("plots/monthly_means/{y}".format(y=year), exist_ok=True)
    plt.savefig("plots/monthly_means/{y}/{b}_{y}.png".format(b=benchmark, y=year))
    plt.close()

    print("Plotting monthly std dev grid")
    fig = plt.figure(0, (14, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 4), axes_pad=0.3,
                     cbar_mode='single', cbar_location="bottom",)
    for i, m in enumerate(months):
        plt.sca(grid[i])
        plot_array(monthly_std.sel(month=i + 1))
        plt.title(m)
    plt.suptitle("{b} - {y} monthly std devs".format(b=benchmark, y=year))
    plt.tight_layout()

    os.makedirs("plots/monthly_stds/{y}".format(y=year), exist_ok=True)
    plt.savefig("plots/monthly_stds/{y}/{b}_{y}.png".format(b=benchmark, y=year))
    plt.close()

    print("Plotting annual Mean")
    plot_array(annual_mean)
    plt.title("{b} - {y} annual mean".format(b=benchmark, y=year))
    plt.tight_layout()
    plt.colorbar()

    os.makedirs("plots/annual_mean/{y}".format(y=year), exist_ok=True)
    plt.savefig("plots/annual_mean/{y}/{b}_{y}.png".format(b=benchmark, y=year))
    plt.close()

    print("Plotting annual Std dev")
    plot_array(annual_std)
    plt.title("{b} - {y} annual std dev".format(b=benchmark, y=year))
    plt.tight_layout()
    plt.colorbar()

    os.makedirs("plots/annual_std/{y}".format(y=year), exist_ok=True)
    plt.savefig("plots/annual_std/{y}/{b}_{y}.png".format(b=benchmark, y=year))
    plt.close()

    # arrange/save plots


def main(args):

    if args['year']:
        plot_year(args['<benchmark>'], args['<variable>'], args['<year>'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
