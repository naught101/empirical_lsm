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
from docopt import docopt
from matplotlib.cbook import dedent
from datetime import datetime as dt


def model_search_index_rst():
    """mail model search index
    """
    time = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    model_run_files = glob.glob('source/models/*/*.rst')

    model_pages = [m.lstrip('source/').rstrip('.rst') for m in model_run_files]

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

    return


if (__name__ == '__main__'):
    args = docopt(__doc__)

    main(args)
