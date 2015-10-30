#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model_search_indexes.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description: create indexes for the model search

Usage:
    model_search_indexes.py
"""

import glob
import os
from docopt import docopt
from matplotlib.cbook import dedent
from datetime import datetime as dt
from ubermodel.plots import get_PLUMBER_plot
from ubermodel.utils import print_good


def model_site_index_rst(model_dir):
    """list site simulations for a model and print global model plots

    :model_dir: TODO
    :returns: TODO

    """
    time = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')
    name = model_dir.replace('source/models/', '')

    print_good('Generating index for {n}.'. format(n=name))

    model_run_files = sorted(glob.glob(model_dir + '/*.rst'))

    sim_pages = [m.replace('source/models/', '').replace('.rst', '') for m in model_run_files]

    sim_links = '\n'.join(['    %s' % m for m in sim_pages])

    plots = get_PLUMBER_plot(model_dir)

    template = dedent("""
    {name} simulations
    ==============================

    {time}

    Plots
    -----

    {plots}

    Simulations
    -----------

    .. toctree::
        :maxdepth: 1

    {links}
    """)

    rst = template.format(time=time, links=sim_links, name=name, plots=plots)

    with open(model_dir + '.rst', 'w') as f:
        f.write(rst)

    return


def model_search_index_rst():
    """mail model search index
    """
    time = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    print_good('Generating models index')

    model_dirs = [d for d in sorted(glob.glob('source/models/*')) if os.path.isdir(d)]

    model_pages = [m.replace('source/', '') for m in model_dirs]

    model_links = '\n'.join(['    %s' % m for m in model_pages])

    template = dedent("""
    Model Search
    =============

    {time}

    .. toctree::
        :maxdepth: 1

    {links}
    """)

    with open('source/model_search.rst', 'w') as f:
        f.write(template.format(time=time, links=model_links))

    return


def main(args):

    model_search_index_rst()

    model_dirs = [d for d in sorted(glob.glob('source/models/*')) if os.path.isdir(d)]

    for model_dir in model_dirs:
        model_site_index_rst(model_dir)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
