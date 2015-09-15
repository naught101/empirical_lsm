#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: land_data.py
Author: ned haughton
Email: ned@nedhaughton.com
Github:
Description: Land surface data handler for PALS data.
"""

import numpy as np
import xray
# import pandas as pd
# import joblib
# import time


MET_VARS = ["SWdown", "Tair", "LWdown", "Wind", "Rainf", "PSurf", "Qair"]
FLUX_VARS = ["Qh", "Qle", "Rnet", "NEE"]


class LandData(object):
    """Base class for land surface storage classes"""

    def __init__(self):
        pass


class SiteData():
    """Land surface data storage mechanism for a single site.

    Stores met data, flux data, and domain data.

    TODO: This could be used to store model output too...

    :param name: arbitrary name for the dataset (useful for plotting, etc)
    :param domain_type: "site" TODO: or "grid" or "set" (multiple sites)
    :param geo: geographic data (dict for "site", TODO: dict of dicts for "set", array for "grid")
    :param met: dataframe of met data
    :param flux: dataframe of flux data
    :param veg: dict of vegetation data (constant)
    :param soil: dict of soil data (constant)
    """

    def __init__(self, name, domain_type, geo, met=None, flux=None, veg=None, soil=None, metadata=None):
        super(SiteData, self).__init__()
        self.name = name
        self.domain_type = domain_type
        self.geo = geo
        self.met = met
        self.flux = flux
        self.veg = veg
        self.soil = soil

    def copy_data(self, met=None, flux=None):
        """Return a copy of the land dataset.

        met and flux components optional: use self.met, self.flux
        """
        return LandData(self.name, self.domain_type, self.geo, met, flux, self.veg, self.soil, metadata)

    def time_split(self, ratio):
        """Split data along the time axis, by ratio"""
        first_len = np.floor(ratio * len(self.met))
        first = self.copy_data()
        second = self.copy_data()
        if self.met is not None:
            first.met = self.met[:first_len]
            second.met = self.met[:first_len]
        if self.flux is not None:
            first.flux = self.flux[:first_len]
            second.flux = self.flux[:first_len]
            pen
        return first, second

    def load_ncdf(self, filename):
        """Load data from a PALS-style netCDF file
        
        TODO: deal with quality control flags (return NaN where qc=0?)

        :param filename: path to file to open
        """
        data = xray.open_dataset(filename)
        data_vars = list(data.vars.keys())
        met_vars = list(set(MET_VARS).union(data_vars))
        flux_vars = list(set(FLUX_VARS).union(data_vars))
        self.met = data.to_dataframe().reset_index(["x", "y", "z"]).ix[:, met_vars]
        self.flux = data.to_dataframe().reset_index(["x", "y", "z"]).ix[:, flux_vars]

        self.geo = dict(
            latitude=data.latitude.values.flatten()[0],
            longitude=data.longitude.values.flatten()[0],
            elevation=data.elevation.values.flatten()[0],
            reference_height=data.reference_height.values.flatten()[0],
        )

        self.metadata = data.attrs

        return(self)

    def save_ncdf(self, filename):
        """Save land data to a PALS-style netcdf file

        :param filename: path to file to open
        """

        data = xray.Dataset(
            variables=dict(
                latitude=(['x', 'y'], [[self.geo["latitude"]]]),
                longitude=(['x', 'y'], [[self.geo["longitude"]]]),
                elevation=(['x', 'y'], [[self.geo["elevation"]]]),
                reference_height=(['x', 'y'], [[self.geo["reference_height"]]])
            ),
            coords=dict(
                x=(['x'], [1]),
                y=(['y'], [1])
            )
        )

        if self.met is not None:
            data.merge(xray.Dataset.from_dataframe(self.met))
        if self.flux is not None:
            data.merge(xray.Dataset.from_dataframe(self.flux))

        data.attrs.update(self.metadata)
