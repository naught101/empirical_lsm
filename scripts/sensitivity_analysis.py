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
            MI_matrix[var1][var2] = mutual_info(df[var1], df[var2])
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
                    mutual_info(flux_df[flux_var], met_lag[flux_var])
                results[flux_var][met_var]['corr']['lag_' + lag] = \
                    np.corrcoef(flux_df[flux_var], flux_df[flux_var])[0, 1]

        # Lag by day, up to 10 days
        met_roll_24 = pd.rolling_mean(met_df, 48)
        for lag in range(10):
            met_lag = met_roll_24.shift(lag * 48)
            for met_var in met_df.columns:
                results[flux_var][met_var] = dict(mutual_info=dict(), corr=dict())
                results[flux_var][met_var]['mutual_info']['lag_day_' + lag] = \
                    mutual_info(flux_df[flux_var], met_lag[flux_var])
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


def hexplot_matrix(df):
    """Plot matrix of variables in the dataframe against each other
    """
    dim = df.shape[1]
    nbins = np.floor(np.sqrt(df.shape[0] / 40))

    fig, axes = plt.subplots(dim, dim, gridspec_kw=dict(wspace=0.01, hspace=0.01))

    for i in range(dim):
        icol = df.columns[i]
        ax = axes[i, i]
        ax.hist(df[icol], bins=nbins)
        ax.set_yticklabels([])
        if i == 0:
            ax.set_ylabel(icol)

        # Below diagonal: scatter plots
        for j in range(i):
            jcol = df.columns[j]
            ax = axes[i, j]
            ax.scatter(df[icol], df[jcol], alpha=0.05, s=5, marker='.')
            if i != dim - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            if j == 0:
                ax.set_ylabel(icol)
            if i == dim - 1:
                ax.set_xlabel(jcol)

        # Above diagonal: hexbin plots
        for j in range(i + 1, dim):
            jcol = df.columns[j]
            ax = axes[i, j]
            ax.hexbin(df[icol], df[jcol], gridsize=nbins, bins='log',
                      cmap=plt.get_cmap('Blues'))  # , norm=mc.LogNorm(vmin=1, vmax=10))
            ax.set_xticklabels([])
            if j == dim - 1:
                ax.yaxis.set_ticks_position('right')
            else:
                ax.set_yticklabels([])


def main(args):

    # Load all data
    # (split by site/site type?)
    sites = ['Tumba']
    met_df = pu.data.get_met_df(sites)
    flux_df = pu.data.get_flux_df(sites)

    all_df = pd.concat([flux_df, met_df], axis=1)
    normed_df = all_df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    var_names = all_df.columns

    # Calculate pairwise measures
    MI_matrix = get_df_MI_matrix(normed_df)

    cov_matrix = normed_df.cov()

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

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
