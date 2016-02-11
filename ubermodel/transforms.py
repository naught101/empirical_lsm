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

    """Wraps a scikit-learn model, lags the data, and deals with NAs."""

    def __init__(self, model, periods=1, freq='30min'):
        """Lags a dataset.

        Lags all features.
        Missing data is dropped for fitting, and replaced with the mean for predict.

        :periods: Number of timesteps to lag by
        """
        assert isinstance(model, BaseEstimator), "`model` isn't a scikit-learn model"

        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

        self.periods = periods
        self.freq = freq

        self.model = model

    def fit(self, X, y=None):
        """Fit the model with X

        compute number of output features

        :X: pandas dataframe
        :y: Pandas dataframe or series
        """
        if 'site' in X.columns:
            raise ValueError("site should be an index, not a column")

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        self.X_mean = X.mean()
        self.X_cols = X.columns

        X_lag = self.transform(X, nans='drop')

        print("Data lagged, now fitting.")

        self.model.fit(X_lag, y.ix[X_lag.index])

        return self

    def lag_dataframe(self, df, grouping=None, lagged_only=False, variables=None):
        """Helper for lagging a dataframe

        :df: Pandas dataframe with a time index
        :returns: Dataframe with all columns copied and lagged

        """
        df = pd.DataFrame(df)
        if not all(df.dtypes == 'float64'):
            raise ValueError('One or more columns are non-numeric.')

        if variables is None:
            shifted = df
        else:
            shifted = df[variables]

        if grouping is not None:
            shifted = (shifted.reset_index(grouping)
                              .groupby(grouping)
                              .shift(self.periods, self.freq)
                              .reset_index(grouping, drop=True)
                              .set_index(grouping, append=True))
            # it would be nice to do this, but https://github.com/pydata/pandas/issues/114524
            # shifted = df.groupby(level=grouping).shift(self.periods, self.freq)
        else:
            shifted = shifted.shift(self.periods, self.freq)
        shifted.columns = [c + '_lag' for c in shifted.columns]

        if lagged_only:
            return shifted.ix[df.index]
        else:
            return df.join(shifted)

    def fix_nans(self, lagged_df, nans=None):
        """Remove NAs, replace with mean, or do nothing

        :df: Pandas dataframe
        :nans: 'drop' to drop NAs, 'fill' to fill with mean values, None to leave NAs in place.
        :returns: Dataframe with NANs modified

        """
        if nans == 'drop':
            return lagged_df.dropna()
        elif nans == 'fill':
            # Replace NAs in lagged columns with mean values from fitting step
            replace = {c + '_lag': {np.nan: self.X_mean[c]} for c in self.X_cols}
            if hasattr(self, 'y_cols'):
                replace.update({c + '_lag': {np.nan: self.y_mean[c]} for c in self.y_cols})
            lagged_df.replace(replace, inplace=True)
            return lagged_df
        else:
            # return with NAs
            return lagged_df

    def transform(self, X, nans=None):
        """Add lagged features to X

        :X: Dataframe matching the fit frame.
        :nans: 'drop' to drop NAs, 'fill' to fill with mean values, None to leave NAs in place.
        :returns: Dataframe with lagged duplicate columns

        """
        check_is_fitted(self, ['n_features', 'n_outputs'])

        n_samples, n_features = X.shape

        if n_features != self.n_features:
            raise ValueError("X shape does not match training shape")

        if 'site' in X.index.names:
            X_lag = self.lag_dataframe(X, grouping='site')
        else:
            X_lag = self.lag_dataframe(X)

        return self.fix_nans(X_lag, nans)

    def predict(self, X):
        """Predicts with a model using lagged X

        :X: Dataframe matching the fit dataframe
        :returns: prediction based on X
        """

        X_lag = self.transform(X, nans='fill')

        print("Data lagged, now predicting.")

        return self.model.predict(X_lag)


class MarkovWrapper(LagWrapper):

    """Wraps a scikit-learn model, Markov-lags the data (includes y values), and deals with NAs."""

    def fit(self, X, y):
        """Fit the model with X

        compute number of output features

        :X: pandas dataframe
        :y: Pandas dataframe or series
        """
        if 'site' in X.columns:
            raise ValueError("site should be an index, not a column")

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        self.X_mean = X.mean()
        self.X_cols = X.columns

        self.y_mean = y.mean()
        self.y_cols = y.columns

        X_lag = self.transform(X, y, nans='drop')

        print("Data lagged, now fitting.")

        self.model.fit(X_lag, y.ix[X_lag.index])

        return self

    def transform(self, X, y=None, nans=None):
        """Add lagged features of X and y to X

        :X: features dataframe
        :y: outputs dataframe
        :nans: 'drop' to drop NAs, 'fill' to fill with mean values, None to leave NAs in place.
        :returns: X dataframe with X and y lagged columns

        """
        check_is_fitted(self, ['n_features', 'n_outputs'])

        n_samples, n_features = X.shape

        if n_features != self.n_features:
            raise ValueError("X shape does not match training shape")

        if 'site' in X.index.names:
            grouping = 'site'
        else:
            grouping = None

        X_lag = self.lag_dataframe(X, grouping=grouping)
        if y is not None:
            y_lag = self.lag_dataframe(y, grouping=grouping, lagged_only=True)
            X_lag = pd.merge(X_lag, y_lag, how='left', left_index=True, right_index=True)

        return self.fix_nans(X_lag, nans)

    def predict(self, X):
        """Predicts with a model using lagged X

        :X: Dataframe matching the fit frame
        :returns: Dataframe of predictions
        """

        X_lag = self.transform(X, nans='fill')

        print("Data lagged, now predicting, step by step.")

        # initialise with mean flux values
        init = pd.concat([X_lag.iloc[0], self.y_mean]).reshape(1, -1)
        results = []
        results.append(self.model.predict(init).ravel())
        n_steps = X_lag.shape[0]
        print('Predicting, step 0 of {n}'.format(n=n_steps), end='\r')

        for i in range(1, n_steps):
            if i % 100 == 0:
                print('Predicting, step {i} of {n}'.format(i=i, n=n_steps), end="\r")
            x = np.concatenate([X_lag.iloc[i], results[i - 1]]).reshape(1, -1)
            results.append(self.model.predict(x).ravel())
        print('Predicting, step {i} of {n}'.format(i=n_steps, n=n_steps))

        # results = pd.DataFrame.from_records(results, index=X_lag.index, columns=self.y_cols)
        # Scikit-learn models produce numpy arrays, not pandas dataframes
        results = np.concatenate(results)

        return results


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
