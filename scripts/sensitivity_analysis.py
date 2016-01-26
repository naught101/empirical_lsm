#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: sensitivity_analysis.py
Author: naught101
Email: naught101@email.com
Description: TODO: File description

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


def entropy(x, bins=10000):
    """Calculate entropy of a variable based on a histrogram

    :x: TODO
    :returns: TODO

    """
    counts = np.histogram(x, bins, density=True)[0]
    probs = counts / np.sum(counts)
    entropy = - (probs * np.ma.log2(probs)).sum()

    return entropy


def mutual_info(x, y, bins=[10000, 10000]):
    """Calculate mutual information based on 2D histrograms

    :x: TODO
    :y: TODO
    :returns: TODO
    """
    hist = np.histogram2d(x, y, bins, normed=True)[0]
    joint_prob = hist / np.sum(hist)

    hist = np.histogram(x, bins[0], density=True)[0]
    probs_x = hist / np.sum(hist)

    hist = np.histogram(y, bins[1], density=True)[0]
    probs_y = hist / np.sum(hist)

    probs = np.divide(np.divide(joint_prob,
                                np.reshape(probs_x, [-1, 1])),
                      probs_y)

    info = (joint_prob * np.ma.log2(probs)).sum()

    return info


def get_df_MI_matrix(df):
    """TODO: Docstring for get_self_measures.

    :df: TODO
    :returns: TODO

    """
    MI_matrix = dict()
    for var1 in df.columns:
        MI_matrix[var1] = dict()
        for var2 in df.columns:
            MI_matrix[var1][var2] = mutual_info(df[var1], df[var2])
    return MI_matrix


def get_measures(flux_df, met_df):
    """TODO: Docstring for get_measures.

    :flux_df: TODO
    :met_df: TODO
    :returns: TODO

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


def main(args):

    # Load all data
    # (split by site/site type?)
    sites = ['Tumba']
    met_df = pu.data.get_met_df(sites)
    flux_df = pu.data.get_flux_df(sites)
    all_df = pd.concat([flux_df, met_df], axis=1)

    MI_matrix = get_df_MI_matrix(all_df)

    cov_matrix = all_df.cov()

    # run a sensitivity study on:
    #    past 10(?) days,
    #    past 24 hours averagess
    #    autocorrelation?
    #    other?


    # plot results, decide on thresholds, based on limiting the number of input variabels?

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
