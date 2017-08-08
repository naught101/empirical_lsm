#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scikit-Learn Model-by-Cluster wrapper.

Original code by jnorthman: https://gist.github.com/jnothman/566ebde618ec18f2bea6
"""

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.utils import safe_mask


class ModelByCluster(BaseEstimator):
    """Cluster data, then run a regression independently on each cluster.

    Parameters:

    :clusterer: scikit-learn style clustering model

    :regression: scikit-learn style regression model
    """

    def __init__(self, clusterer, estimator):
        self.clusterer = clusterer
        self.estimator = estimator

    def fit(self, X, y):
        self.clusterer_ = clone(self.clusterer)
        clusters = self.clusterer_.fit_predict(X)
        cluster_ids = np.unique(clusters)

        assert (len(cluster_ids) == self.clusterer_.n_clusters), \
            "MBC: Some clusters have no data. Probably too little data available: " + \
            "Only {n} data points for {k} clusters.".format(
                n=X.shape[0], k=self.clusterer_.n_clusters)

        self.estimators_ = {}
        for c in cluster_ids:
            mask = clusters == c
            est = clone(self.estimator)
            est.fit(X[safe_mask(X, mask)], y[safe_mask(y, mask)])
            self.estimators_[c] = est

        return self

    def predict(self, X):
        # this returns -1 if any of the values squared are too large
        # models with numerical instability will fail.
        clusters = self.clusterer_.predict(X)

        y_tmp = []
        idx = []
        for c, est in self.estimators_.items():
            mask = clusters == c
            if mask.any():
                idx.append(np.flatnonzero(mask))
                y_tmp.append(est.predict(X[safe_mask(X, mask)]))

        y_tmp = np.concatenate(y_tmp)
        idx = np.concatenate(idx)
        y = np.full([X.shape[0], y_tmp.shape[1]], np.nan)
        y[idx] = y_tmp

        return y
