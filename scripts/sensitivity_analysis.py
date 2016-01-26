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
import pals_utils as pu


def entropy(x, bins=1000):
    """Calculate entropy of a variable based on a histrogram

    :x: TODO
    :returns: TODO

    """
    counts = np.histogram(x, bins, density=True)[0]
    probs = counts / np.sum(counts)
    entropy = - (probs * np.ma.log2(probs)).sum()

    return entropy


def mutual_info(x, y, bins=[1000, 1000]):
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


def main(args):

    # Load all data
    # (split by site/site type?)
    sites = ['Tumba']
    met_df = pu.data.get_met_data(sites)
    flux_df = pu.data.get_flux_data(sites)

    # run a sensitivity study on:
    #    past 10(?) days,
    #    past 24 hours averagess
    #    autocorrelation?
    #    other?

    results = dict()
    for flux_var in flux_df.columns:

        results[flux_var] = dict()
        results[flux_var]['entropy'] = entropy(flux_df[flux_var])
        # Autocorrelation:

        for met_var in met_df.columns:
            results[flux_var][met_var] = dict()
            results[flux_var][met_var]['mutual_info'] = mutual_info(flux_df[flux_var], flux_df[flux_var])

    # plot results, decide on thresholds, based on limiting the number of input variabels?

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
