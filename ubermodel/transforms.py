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


class LagWrapper(BaseEstimator, TransformerMixin):

    """Wraps a scikit-learn pipeline, lags the data, and deals with NAs."""

    def __init__(self, pipeline, periods=1, freq='30min'):
        """Lags a dataset.

        Lags all features.
        Missing data is dropped for fitting, and replaced with the mean for predict.

        :periods: Number of timesteps to lag by
        """
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

        self._periods = periods
        self._freq = freq

        self._pipeline = pipeline

    def fit(self, X, y=None):
        """Fit the model with X

        compute number of output features

        :X: pandas dataframe
        :y: Pandas series or vector
        """
        if 'site' in X.columns:
            raise ValueError("site should be an index, not a column")

        n_features = X.select_dtypes(include=[np.number]).shape[1]
        self.n_input_features_ = n_features
        self.n_output_features_ = 2 * n_features

        self._X_mean = X.mean()
        self._X_cols = X.columns

        X_lag = self.transform(X, dropna=True)

        self._pipeline.fit(X_lag, y)

        return self

    def transform(self, X, dropna=False):
        """Add lagged features to X

        :X: TODO
        :returns: TODO

        """
        check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        if 'site' in X.index.names:
            X_lag = (X.groupby(level='site')
                      .apply(lag_dataframe, periods=self._periods, freq=self._freq))
        else:
            X_lag = X.apply(lag_dataframe, periods=self._periods, freq=self._freq)

        if dropna:
            return X_lag.dropna()
        else:
            return X_lag

    def predict(self, X):
        """Predicts with a pipeline using lagged X

        :X: TODO
        :returns: TODO

        """

        X_lag = self.transform(X)

        # Replace NAs with mean values from fitting step
        replace = {c + '_lag': {np.nan: self._X_mean[c]} for c in self._X_cols}

        X_lag.replace(replace, inplace=True)

        return self._pipeline.predict(X)


class PandasCleaner(BaseEstimator, TransformerMixin):
    """Removes rows with NAs from both X and y, and converts to an array and back"""

    def __init__(self, remove_NA=True):
        """:remove_NA: Whether to remove NA rows from the data

        :remove_NA: TODO

        """
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

        self._remove_NA = remove_NA

    def fit(self, X, y=None):
        """Gather pandas metadata and store it.

        :X: TODO
        :y: TODO
        :returns: TODO

        """
        if 'site' in X.columns:
            self.X_sites_ = X.pop('site')
        else:
            self.X_sites_ = None
        self.X_columns_ = X.columns
        self.X_index_ = X.index
        self.X_col_types_ = [(c, X[c].dtype) for c in X.columns]

        if y is not None:
            if 'site' in y.columns:
                self.y_sites_ = y.pop('site')
            else:
                self.y_sites_ = None
            self.y_columns_ = y.columns
            self.y_index_ = y.index
            self.y_col_types_ = [(c, y[c].dtype) for c in y.columns]

    def transform(self, X):
        """Transforms

        :X: TODO
        :returns: TODO

        """
        pass
