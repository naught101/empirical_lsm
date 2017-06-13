#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: inspect_cluster_regression.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Inspects a cluster+regression model for weird values

Usage:
    inspect_cluster_regression.py <model_path>
    inspect_cluster_regression.py (-h | --help | --version)

Options:
    -h, --help    Show this screen and exit.
    --option=<n>  Option description [default: 3]
"""

from docopt import docopt

import pickle
import numpy as np
import ubermodel


def get_wrapper_vars(wrapper):
    return [v + '_' + l for v, lags in wrapper.var_lags.items() for l in lags]


def get_cluster_regression_model(wrapper):
    """Unwraps a model"""
    if isinstance(wrapper, ubermodel.clusterregression.ModelByCluster):
        return wrapper
    else:
        return get_cluster_regression_model(wrapper.model)


def get_cluster_regression_centers(model):
    return(model.clusterer_.cluster_centers_)


def get_cluster_regression_counts(model):
    return(model.clusterer_.counts_)


def get_cluster_regression_n_iter(model):
    return(model.clusterer_.n_iter_)


def get_cluster_regression_inertia(model):
    return(model.clusterer_.inertia_)


def get_regression_parameters(model):
    return np.concatenate(model.coef_, model.intercept_)


def get_cluster_regression_parameters(model):
    # TODO: This will only work for single-output models
    return np.array([get_regression_parameters(reg) for reg in model.estimators_])


def main(args):

    with open(args['<model_path>'], 'rb') as f:
        wrapper = pickle.load(f)

    model = get_cluster_regression_model(wrapper)

    counts = get_cluster_regression_counts(model)
    count_threshold = 100
    if (counts < count_threshold).any():
        bad_clusters = np.where(counts < count_threshold)[0]
        print("low counts at clusters", bad_clusters.tolist())



    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
