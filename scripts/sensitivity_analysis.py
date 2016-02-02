#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: sensitivity_analysis.py
Author: naught101
Email: naught101@email.com
Description: Investigation into dependence between met and flux in the PLUMBER datasets

Usage:
    sensitivity_analysis.py command <opt1> [--timeout=<seconds>]
    sensitivity_analysis.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
    --option=<n>  Option description [default: 3]
"""

from docopt import docopt

import numpy as np
import pandas as pd
import pals_utils as pu
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.stats as ss

from mutual_info.mutual_info import mutual_information_2d


def entropy(x, bins='auto'):
    """Calculate entropy of a variable based on a histrogram
    """
    if bins == 'auto':
        # Rule of thumb based on http://stats.stackexchange.com/a/181195/9007
        b = np.floor(np.sqrt(len(x) / 5))
        bins = [b, b]

    counts = np.histogram(x, bins, density=True)[0]
    probs = counts / np.sum(counts)
    entropy = - (probs * np.ma.log2(probs)).sum()

    return entropy


def mutual_info(x, y, bins='auto'):
    """Calculate mutual information based on 2D histrograms
    """
    if bins == 'auto':
        # Rule of thumb based on http://stats.stackexchange.com/a/181195/9007
        b = np.floor(np.sqrt(len(x) / 5))
        bins = [b, b]

    hist = np.histogram2d(x, y, bins, normed=True)[0]
    joint_prob = hist / np.sum(hist)

    hist = np.histogram(x, bins[0], density=True)[0]
    probs_x = hist / np.sum(hist)

    hist = np.histogram(y, bins[1], density=True)[0]
    probs_y = hist / np.sum(hist)

    probs = joint_prob / (np.reshape(probs_x, [-1, 1]) * probs_y)

    # use masked array to avoid NaNs
    info = (joint_prob * np.ma.log2(probs)).sum()

    return info


def get_df_MI_matrix(df):
    """Gets the mutual information of each pair of variables in a dataframe
    """
    MI_matrix = dict()
    for var1 in df.columns:
        MI_matrix[var1] = dict()
        for var2 in df.columns:
            MI_matrix[var1][var2] = mutual_information_2d(df[var1], df[var2])
    return pd.DataFrame(MI_matrix)


def get_measures(flux_df, met_df):
    """Gets mutual information and covariance for multiple lags of a dataset.
    """
    results = dict()
    for flux_var in flux_df.columns:
        # lag by half hours, up to 24 hours
        for lag in range(49):
            met_lag = met_df.shift(lag)
            for met_var in met_df.columns:
                results[flux_var][met_var] = dict(mutual_info=dict(), corr=dict())
                results[flux_var][met_var]['mutual_info']['lag_' + lag] = \
                    mutual_information_2d(flux_df[flux_var], met_lag[flux_var])
                results[flux_var][met_var]['corr']['lag_' + lag] = \
                    np.corrcoef(flux_df[flux_var], flux_df[flux_var])[0, 1]

        # Lag by day, up to 10 days
        met_roll_24 = pd.rolling_mean(met_df, 48)
        for lag in range(10):
            met_lag = met_roll_24.shift(lag * 48)
            for met_var in met_df.columns:
                results[flux_var][met_var] = dict(mutual_info=dict(), corr=dict())
                results[flux_var][met_var]['mutual_info']['lag_day_' + lag] = \
                    mutual_information_2d(flux_df[flux_var], met_lag[flux_var])
                results[flux_var][met_var]['corr']['lag_day_' + lag] = \
                    np.corrcoef(flux_df[flux_var], flux_df[flux_var])[0, 1]

    return results


def plot_matrix(mat, names):
    """plot a square covariance matrix
    """
    im = plt.imshow(mat, interpolation='nearest')
    im.axes.set_xticks(range(mat.shape[0]))
    im.axes.set_yticks(range(mat.shape[1]))
    im.axes.set_xticklabels(names, rotation=90)
    im.axes.set_yticklabels(names)

    return im


def plot_lin_reg_CI(x, y, ax=None):
    """Plot linear regression with confidence limits

    Modified from https://tomholderness.wordpress.com/2013/01/10/confidence_intervals/

    References:
    - Statistics in Geography by David Ebdon (ISBN: 978-0631136880)
    - Reliability Engineering Resource Website:
    - http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
    - University of Glascow, Department of Statistics:
    - http://www.stats.gla.ac.uk/steps/glossary/confidence_intervals.html#conflim
    """

    # fit a curve to the data using a least squares 1st order polynomial fit
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    # get the coordinates for the fit curve
    c_x = np.percentile(x, [1, 99])
    c_y = p(c_x)

    # predict y values of origional data using the fit
    p_y = z[0] * x + z[1]

    # calculate the y-error (residuals)
    y_err = y - p_y

    # R-squared for plots 1-SSresid/SStot
    r2 = 1 - np.sum(y_err**2) / np.sum((y - np.mean(y))**2)

    # create series of new test x-values to predict for
    p_x = np.linspace(c_x[0], c_x[1], 100)

    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x)                 # mean of x
    n = len(x)                          # number of samples in origional fit
    t = ss.t.ppf(0.99975, n - 1)          # appropriate t value (where n=9, two tailed 95%)
    s_err = np.sum(np.power(y_err, 2))  # sum of the squares of the residuals

    confs = t * np.sqrt((s_err / (n - 2)) *
                        (1.0 / n + (np.power(p_x - mean_x, 2) /
                         (np.sum(np.power(x, 2)) - n * np.power(mean_x, 2)))
                         ))

    # now predict y based on test x-values
    p_y = z[0] * p_x + z[1]

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)

    # import ipdb; ipdb.set_trace()

    # plot line of best fit
    ax.plot(c_x, c_y, 'r-', label='Regression line')

    # plot confidence limits
    ax.plot(p_x, lower, 'r--', label='Lower confidence limit (95%)')
    ax.plot(p_x, upper, 'r--', label='Upper confidence limit (95%)')
    plt.annotate('R^2 = {0:.2f}'.format(r2), xy=(0.9, 0.9), xycoords='axes fraction')


def hexplot_matrix(df):
    """Plot matrix of variables in the dataframe against each other
    """
    dim = df.shape[1]
    nbins = np.floor(np.sqrt(df.shape[0] / 40))

    fig, axes = plt.subplots(dim, dim, gridspec_kw=dict(wspace=0.05, hspace=0.05))

    for i in range(dim):
        icol = df.columns[i]

        for j in range(dim):
            jcol = df.columns[j]
            ax = axes[i, j]
            # Diagonals: histograms
            if i == j:
                ax.hist(df[icol], bins=nbins)

            # Below diagonal: scatter plots
            if j < i:
                ax.scatter(df[jcol], df[icol], alpha=0.05, s=5, marker='.')

            # Above diagonal: hexbin plots
            if j > i:
                if j == dim - 1:
                    ax.yaxis.set_ticks_position('right')

                ax.hexbin(df[jcol], df[icol], gridsize=nbins, bins='log',
                          cmap=plt.get_cmap('Blues'))
                ax.set_xticklabels([])

            # linear regressions on each plot
            if j != i:
                plot_lin_reg_CI(df[jcol], df[icol], ax=axes[i, j])

            # Add labels to rows and columns
            if j == 0:
                ax.set_ylabel(icol)
                ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
            if i == dim - 1:
                ax.set_xlabel(jcol)
                ax.xaxis.set_major_locator(MaxNLocator(prune='both'))

            # remove stuff from all but the side rows:
            if j > 0:
                ax.set_yticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            if i != dim - 1:
                ax.set_xticklabels([])


def plot_MI_cov_matrices(df):
    var_names = df.columns

    # Calculate pairwise measures
    MI_matrix = get_df_MI_matrix(df)

    cov_matrix = df.cov()

    # run a sensitivity study on:
    #    past 10(?) days,
    #    past 24 hours averagess
    #    autocorrelation?
    #    other?

    # plot results, decide on thresholds, based on limiting the number of input variabels?
    plt.subplot(1, 2, 1)
    plot_matrix(MI_matrix, var_names)
    plt.axvline(4.5, color="w")
    plt.axhline(4.5, color="w")
    plt.colorbar(shrink=0.7)
    plt.title("Mutual Information matrix")

    plt.subplot(1, 2, 2)
    plot_matrix(cov_matrix, var_names)
    plt.axvline(4.5, color="w")
    plt.axhline(4.5, color="w")
    plt.colorbar(shrink=0.7)
    plt.title("Covariance matrix")


def main(args):

    # Load all data
    # (split by site/site type?)
    sites = ['Tumba']
    met_df = pu.data.get_met_df(sites)
    flux_df = pu.data.get_flux_df(sites)

    all_df = pd.concat([flux_df, met_df], axis=1)
    normed_df = all_df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    plot_MI_cov_matrices(normed_df)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
