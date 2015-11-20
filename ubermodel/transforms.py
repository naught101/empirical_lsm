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

        self.periods = periods
        self.freq = freq

        self.pipeline = pipeline

    def fit(self, X, y=None):
        """Fit the model with X

        compute number of output features

        :X: pandas dataframe
        :y: Pandas series or vector
        """
        if 'site' in X.columns:
            raise ValueError("site should be an index, not a column")

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        self.X_mean = X.mean()
        self.X_cols = X.columns

        X_lag = self.transform(X, nans='drop')

        self.pipeline.fit(X_lag, y)

        return self

    def lag_dataframe(self, df):
        """Helper for lagging a dataframe

        :df: Pandas dataframe with a time index
        :returns: Dataframe with all columns copied and lagged

        """
        if not all(df.dtypes == 'float64'):
            raise ValueError('One or more columns are non-numeric.')
        shifted = df.shift(self.periods, self.freq)
        shifted.columns = [c + '_lag' for c in shifted.columns]
        new_df = pd.merge(df, shifted, how='left', left_index=True, right_index=True)

        return new_df

    def fix_nans(self, lagged_df, nans=None):
        """Remove NAs, replace with mean, or do nothing

        :df: Pandas dataframe
        :nans: 'drop' to drop NAs, 'fill' to fill with mean values, None to leave NAs in place.
        :returns: TODO

        """
        if nans == 'drop':
            return lagged_df.dropna()
        elif nans == 'fill':
            # Replace NAs in lagged columns with mean values from fitting step
            replace = {c + '_lag': {np.nan: self.X_mean[c]} for c in self.X_cols}
            replace.update({c + '_lag': {np.nan: self.y_mean[c]} for c in self.y_cols})
            lagged_df.replace(replace, inplace=True)
            return lagged_df
        else:
            # return with NAs
            return lagged_df

    def transform(self, X, nans=None):
        """Add lagged features to X

        :X: TODO
        :nans: 'drop' to drop NAs, 'fill' to fill with mean values, None to leave NAs in place.
        :returns: TODO

        """
        check_is_fitted(self, ['_n_features', '_n_outputs'])

        n_samples, n_features = X.shape

        if n_features != self.n_features:
            raise ValueError("X shape does not match training shape")

        if 'site' in X.index.names:
            X_lag = X.groupby(level='site').apply(self.lag_dataframe)
        else:
            X_lag = X.apply(self.lag_dataframe)

        return self.fix_nans(X_lag, nans)

    def predict(self, X):
        """Predicts with a pipeline using lagged X

        :X: TODO
        :returns: TODO

        """

        X_lag = self.transform(X, nans='fill')

        return self.pipeline.predict(X_lag)


class PandasCleaner(BaseEstimator, TransformerMixin):
    """Removes rows with NAs from both X and y, and converts to an array and back"""

    def __init__(self, remove_NA=True):
        """:remove_NA: Whether to remove NA rows from the data

        :remove_NA: TODO

        """
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

        self.remove_NA = remove_NA

    def fit(self, X, y=None):
        """Gather pandas metadata and store it.

        :X: TODO
        :y: TODO
        :returns: TODO

        """
        if 'site' in X.columns:
            self.X_sites = X.pop('site')
        else:
            self.X_sites = None
        self.X_columns = X.columns
        self.X_index = X.index
        self.X_col_types = X.dtypes

        if y is not None:
            if 'site' in y.columns:
                self.y_sites = y.pop('site')
            else:
                self.y_sites = None
            self.y_columns = y.columns
            self.y_index = y.index
            self.y_col_types = [(c, y[c].dtype) for c in y.columns]

    def transform(self, X):
        """Transforms

        :X: TODO
        :returns: TODO

        """
        pass
