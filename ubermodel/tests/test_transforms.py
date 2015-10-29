#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: test_transforms.py
Author: naughton101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: TODO: File description
"""

import unittest
import numpy.testing as npt

from pals_utils.data import get_site_data, xray_list_to_df
from ubermodel.transforms import LagTransform


class TestLagTransform(unittest.TestCase):
    """Test LagTransform"""

    def setUp(self):
        met_data = [ds.isel(time=slice(0, 48)) for ds in
                    get_site_data(['Amplero', 'Tumba'], 'met').values()]

        self.met_data = xray_list_to_df(met_data, ['SWdown', 'Tair'], qc=True, name=True)

    def tearDown(self):
        pass

    def test_shift_30(self):

        lag_transform = LagTransform(1, '30min')

        transformed = lag_transform.fit_transform(self.met_data)

        npt.assert_array_equal(transformed.ix[0, ['SWdown', 'Tair']],
                               transformed.ix[1, ['SWdown_lag', 'Tair_lag']])

        self.assertEqual(self.met_data.shape[0], transformed.shape[0])
        self.assertEqual(self.met_data.shape[1] + 2, transformed.shape[1])

    def test_shift_2H(self):

        lag_transform = LagTransform(2, 'H')

        transformed = lag_transform.fit_transform(self.met_data)

        npt.assert_array_equal(transformed.ix[0, ['SWdown', 'Tair']],
                               transformed.ix[4, ['SWdown_lag', 'Tair_lag']])

        self.assertEqual(self.met_data.shape[0], transformed.shape[0])
        self.assertEqual(self.met_data.shape[1] + 2, transformed.shape[1])


