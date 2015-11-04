#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: import_benchmark.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: imports a benchmark from a PALS dataset

Usage:
    import-benchmark.py <name> [<site>...]

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt
import xray

from pals_utils.constants import DATASETS

from ubermodel.utils import print_good
from ubermodel.data import get_sim_nc_path


def fix_benchmark(name, site_data):
    """Performs checks on broken benchmarks, and fixes them inplace

    :name: benchmark name
    :site_data: xray dataset (will be modified inplace
    """

    if name in ['Manabe_Bucket.2', 'Penman_Monteith.1']:
        lon = site_data['longitude']
        lat = site_data['latitude']
        del site_data['longitude'], site_data['latitude'], lon['latitude'], lat['longitude']
        site_data['longitude'] = lon
        site_data['latitude'] = lat

    return


def main_import_benchmark(name, site):
    """import a PLUMBER benchmark for all sites

    :name: PLUMBER benchmark name
    :site: plumber site name
    """
    # Hacky solution just for PLUMBER benchmarks
    print_good('Importing {n} data for: '.format(n=name))

    if len(site) == 0:
        datasets = DATASETS
    else:
        datasets = site

    for s in datasets:
        print(s, end=', ', flush=True)
        s_file = 'data/PALS/benchmarks/{n}/{n}_{s}Fluxnet.1.4.nc'.format(n=name, s=s)
        nc_path = get_sim_nc_path(name, s)

        sim_data = xray.open_dataset(s_file)

        fix_benchmark(name, sim_data)

        # WARNING! over writes existing sim!
        sim_data.to_netcdf(nc_path)

        sim_data.close()

    return


def main(args):
    name = args['<name>']
    site = args['<site>']

    main_import_benchmark(name, site)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
