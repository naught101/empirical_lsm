#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: test_transforms.py
Author: naughton101
Email: naught101@email.com
Github: https://github.com/naught101/empirical_lsm
Description: Tests for empirical_lsm
"""

import unittest
import numpy.testing as npt
import pandas as pd
import numpy as np
import numpy.testing as nt
from sklearn.linear_model import LinearRegression

from pals_utils.data import get_site_data, xr_list_to_df
from empirical_lsm.transforms import LagWrapper, MarkovWrapper, rolling_mean


class TestLagWrapper(unittest.TestCase):
    """Test LagWrapper"""

    def setUp(self):
        X = [ds.isel(time=slice(0, 48)) for ds in
             get_site_data(['Amplero', 'Tumba'], 'met').values()]

        self.X = xr_list_to_df(X, ['SWdown', 'Tair'], qc=True, name=True)

        self.y = pd.DataFrame(dict(test=list(range(96))), index=self.X.index)

    def tearDown(self):
        pass

    def test_shift_30(self):

        lag_transform = LagWrapper(LinearRegression(), 1, '30min')

        transformed = lag_transform.fit_transform(self.X, self.y)

        npt.assert_array_equal(transformed.ix[0, ['SWdown', 'Tair']],
                               transformed.ix[1, ['SWdown_lag', 'Tair_lag']])

        self.assertEqual(self.X.shape[0], transformed.shape[0])
        self.assertEqual(self.X.shape[1] + 2, transformed.shape[1])

    def test_shift_2H(self):

        lag_transform = LagWrapper(LinearRegression(), 2, 'H')

        transformed = lag_transform.fit_transform(self.X, self.y)

        npt.assert_array_equal(transformed.ix[0, ['SWdown', 'Tair']],
                               transformed.ix[4, ['SWdown_lag', 'Tair_lag']])

        # Check first two hours at each site are empty
        npt.assert_array_equal(
            transformed.ix[[0, 1, 2, 3, 48, 49, 50, 51], ['SWdown_lag', 'Tair_lag']],
            np.nan)

        self.assertEqual(self.X.shape[0], transformed.shape[0])
        self.assertEqual(self.X.shape[1] + 2, transformed.shape[1])


class TestMarkovWrapper(unittest.TestCase):
    """Test MarkovWrapper"""

    def setUp(self):
        index = pd.date_range('2000-01', periods=5, freq='1D')
        df = pd.DataFrame(dict(A=[1, 2, 3, 4, 5], B=[1, 3, 4, 5, 6]),
                          index=index)
        df2 = df.copy()
        df['site'] = 'site1'
        df2['site'] = 'site2'
        self.X = pd.concat([df.set_index('site', append=True),
                            df2.set_index('site', append=True)])

        self.y = (2 * self.X[['B', 'A']]).rename({'B': 'r1', 'A': 'r2'})
        print(self.y)

    def tearDown(self):
        pass

    def test_shift_30(self):

        lag_transform = MarkovWrapper(LinearRegression(), 1, '30min')

        transformed = lag_transform.fit_transform(self.X, self.y)

        npt.assert_array_equal(transformed.ix[0, ['SWdown', 'Tair']],
                               transformed.ix[1, ['SWdown_lag', 'Tair_lag']])

        self.assertEqual(self.X.shape[0], transformed.shape[0])
        self.assertEqual(self.X.shape[1] + 2, transformed.shape[1])

    def test_shift_2H(self):

        lag_transform = LagWrapper(LinearRegression(), 2, 'H')

        transformed = lag_transform.fit_transform(self.X, self.y)

        npt.assert_array_equal(transformed.ix[0, ['SWdown', 'Tair']],
                               transformed.ix[4, ['SWdown_lag', 'Tair_lag']])

        # Check first two hours at each site are empty
        npt.assert_array_equal(
            transformed.ix[[0, 1, 2, 3, 48, 49, 50, 51], ['SWdown_lag', 'Tair_lag']],
            np.nan)

        self.assertEqual(self.X.shape[0], transformed.shape[0])
        self.assertEqual(self.X.shape[1] + 2, transformed.shape[1])


class test_rolling_mean(unittest.TestCase):
    """Tests rolling lagged averages"""

    def setUp(self):
        self.df = pd.DataFrame(dict(var1=[1.0, 0, 0, 0] * 12))  # 48 steps

    def tearDown(self):
        pass

    def test_30m_inclusive(self):
        df_lagged = rolling_mean(self.df.values, '30min')
        nt.assert_array_equal(df_lagged, self.df.values, "arrays should match, but don't")

    def test_30m_exclusive(self):
        df_lagged = rolling_mean(self.df.values, '30min', shift=1)
        nt.assert_array_equal(df_lagged[:1, ], np.nan, "leading lags should be NAN")
        nt.assert_array_equal(df_lagged[1:, ], self.df.values[:-1, ], "lagged array should match, but doesn't")

    def test_2h_inclusive(self):
        df_lagged = rolling_mean(self.df.values, '2h')
        nt.assert_array_equal(df_lagged[:3, ], np.nan, "leading lags should be NAN")
        nt.assert_array_equal(df_lagged[3:, ], 0.25, "Moving average should be constant 1/4")

    def test_2h_exclusive(self):
        df_lagged = rolling_mean(self.df.values, '2h', shift=1)
        nt.assert_array_equal(df_lagged[:4, ], np.nan, "leading lags should be NAN")
        nt.assert_array_equal(df_lagged[4:, ], 0.25, "Moving average should be constant 1/4")

    def test_1d_6hourly_inclusive(self):
        """Should be the same as test_2h_inclusive"""
        df_lagged = rolling_mean(self.df.values, '1d', datafreq=6)
        nt.assert_array_equal(df_lagged[:3, ], np.nan, "leading lags should be NAN")
        nt.assert_array_equal(df_lagged[3:, ], 0.25, "Moving average should be constant 1/4")
