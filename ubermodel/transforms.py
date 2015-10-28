#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: transforms.py
Author: naughton101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Transformations for ubermodel
"""

import pandas as pd

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


class LagTransform(BaseEstimator, TransformerMixin):

    """Docstring for LagTransform. """

    def __init__(self, lag=1, freq='30min'):
        """Lags a dataset.

        Lags all features.
        Missing data is dropped for fitting, and replaced with the mean for transform.

        :lag: Number of timesteps to lag by
        """
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

        self._lag = lag
        self._freq = freq

    def fit(self, X, y=None):
        """Fit the model with X

        compute number of output features
        """
        n_features = check_array(X).shape[1]
        self.n_input_features_ = n_features
        self.n_output_features_ = 2 * n_features

        self.X_mean = X.mean()

        return self

    def transform(self, X):
        """Add lagged features to X

        :X: TODO
        :returns: TODO

        """
        check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        X_lag = pd.concat([X, X.shift(1, self._freq)], axis=1)

        return X_lag
