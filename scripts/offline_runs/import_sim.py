#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: import_sim.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: imports a benchmark from a PALS dataset

Usage:
    import_sim.py benchmark <name> [<site>...]
    import_sim.py sim <name> <site> <file>

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt
import numpy as np
import xarray as xr

from pals_utils.data import get_flux_data
from ubermodel.utils import print_good
from ubermodel.data import get_sites, get_sim_nc_path


def fix_benchmark(site_data, name, site):
    """Performs checks on broken benchmarks, and fixes them inplace

    :name: benchmark name
    :site_data: xarray dataset (will be modified inplace
    """

    if name in ['Manabe_Bucket.2', 'Penman_Monteith.1']:
        lon = site_data['longitude']
        lat = site_data['latitude']
        del site_data['longitude'], site_data['latitude'], lon['latitude'], lat['longitude']
        site_data['longitude'] = lon
        site_data['latitude'] = lat

        defaults = dict(
            Production_time='2013-01-01 00:00:00',
            Production_source='Martin Best; PLUMBER',
            PALS_dataset_name=site + 'Fluxnet',
            PALS_dataset_version='1.4',
            Contact='palshelp@gmail.com')

        for k in defaults:
            if k not in site_data.attrs:
                site_data.attrs[k] = defaults[k]

    return


def main_import_benchmark(name, site):
    """import a PLUMBER benchmark for all sites

    :name: PLUMBER benchmark name
    :site: plumber site name
    """
    # Hacky solution just for PLUMBER benchmarks
    print_good('Importing {n} data for: '.format(n=name))

    if len(site) == 0:
        datasets = get_sites('PLUMBER')
    else:
        datasets = site

    for s in datasets:
        print(s, end=', ', flush=True)
        s_file = 'data/PALS/benchmarks/{n}/{n}_{s}Fluxnet.1.4.nc'.format(n=name, s=s)
        nc_path = get_sim_nc_path(name, s)

        sim_data = xr.open_dataset(s_file)

        fix_benchmark(sim_data, name, s)

        # WARNING! over writes existing sim!
        sim_data.to_netcdf(nc_path)

        sim_data.close()

    return


def main_import_sim(name, site, sim_file):
    """import a PLUMBER benchmark for all sites

    :name: PLUMBER benchmark name
    :site: plumber site name
    """
    # Hacky solution just for PLUMBER benchmarks
    print_good('Importing {n} data for: {s}'.format(n=name, s=site))

    nc_path = get_sim_nc_path(name, site)

    data_vars = ['Qh', 'Qle', 'NEE']
    with xr.open_dataset(sim_file) as ds:
        d_vars = [v for v in data_vars if v in ds]
        sim_data = ds[d_vars].copy(deep=True)

    if name == 'CHTESSEL':
        print("Inverting CHTESSEL and adding data")
        # Fucking inverted, and
        sim_data = - sim_data

        # missing most of the last fucking day, and
        tsteps = ds.dims['time']
        complete_tsteps = 48 * int(np.ceil(tsteps / 48))
        missing_tsteps = complete_tsteps - tsteps

        new_data = (sim_data.isel(time=slice(- missing_tsteps, None))
                            .copy(deep=True))

        # fucking off-set by an hour.
        with get_flux_data([site])[site] as ds:
            site_time = ds.time.values.flat.copy()

        sim_data['time'] = site_time[:tsteps]
        new_data['time'] = site_time[tsteps:]

        sim_data = xr.concat([sim_data, new_data], dim='time')

    if name == 'ORCHIDEE.trunk_r1401':
        sim_data.rename(dict(time_counter='time'), inplace=True)

        # Stores data for all vegetation types
        # NEE_veget_index = np.where(sim_data.NEE
        #                                    .isel(time_counter=0, lat=0, lon=0)
        #                                    .values.flat > 0)[0][0]
        # sim_data['NEE'] = sim_data['NEE'].isel(veget=NEE_veget_index)
        print("Flattening veg in ORCHIDEE")
        sim_data['NEE'] = sim_data['NEE'].sum(axis=1)  # Sum over veget axis

        if site == 'Espirra':
            print("Deleting a year of data at Espirra for Orchidee")
            sim_data = sim_data.isel(time=slice(70128))

    # WARNING! over writes existing sim!
    print('Writing to', nc_path)
    sim_data.to_netcdf(nc_path)

    sim_data.close()

    return


def main(args):
    name = args['<name>']

    if args['benchmark']:
        site = args['<site>']
        main_import_benchmark(name, site)

    if args['sim']:
        site = args['<site>'][0]
        sim_file = args['<file>']
        main_import_sim(name, site, sim_file)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
