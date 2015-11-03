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


def main_import_benchmark(name, site):
    """import a PLUMBER benchmark for all sites

    :name: PLUMBER benchmark name
    :site: plumber site name
    """
    # Hacky solution just for PLUMBER benchmarks
    print_good('Importing {n} data for: '.format(n=name), end='')

    if site is None:
        datasets = DATASETS
    else:
        datasets = site

    for s in datasets:
        print(s, end=', ')
        s_file = 'data/PALS/benchmarks/{n}/{n}_{s}Fluxnet.1.4.nc'.format(n=name, s=s)
        nc_path = get_sim_nc_path(name, s)

        sim_data = xray.open_dataset(s_file)
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
