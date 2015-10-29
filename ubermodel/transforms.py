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
import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


def lag_dataframe(df, periods, freq):
    """Helper for lagging a dataframe

    :df: TODO
    :periods: TODO
    :freq: TODO
    :returns: TODO

    """
    # TODO: problem: remove trailing entries. For now assume constant spacing, 1 lag
    shifted = df.select_dtypes(include=[np.number]).shift(periods, freq)
    shifted.columns = [c + '_lag' for c in shifted.columns]
    new_df = pd.merge(df, shifted, how='left', left_index=True, right_index=True)

    return new_df


class LagTransform(BaseEstimator, TransformerMixin):

    """Docstring for LagTransform. """

    def __init__(self, periods=1, freq='30min'):
        """Lags a dataset.

        Lags all features.
        Missing data is dropped for fitting, and replaced with the mean for transform.

        :periods: Number of timesteps to lag by
        """
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

        self._periods = periods
        self._freq = freq

    def fit(self, X, y=None):
        """Fit the model with X

        compute number of output features
        """
        n_features = X.select_dtypes(include=[np.number]).shape[1]
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

        # if n_features != self.n_input_features_:
        #     raise ValueError("X shape does not match training shape")

        if 'site' in X.index.names:
            X_lag = (X.reset_index('site')
                      .groupby('site')
                      .apply(lag_dataframe, periods=self._periods, freq=self._freq))

        elif 'site' in X.columns:
            X_lag = (X.groupby('site')
                      .apply(lag_dataframe, periods=self._periods, freq=self._freq))
        # TODO: if predict transform, fill NAs with mean, if fit transform, drop NAs.

        return X_lag
