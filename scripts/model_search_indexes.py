#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model_search_indexes.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description: create indexes for the model search

Usage:
    model_search_indexes.py all
    model_search_indexes.py model <name>
    model_search_indexes.py summary
"""

from docopt import docopt

import glob
import os
import pandas as pd

from matplotlib.cbook import dedent
from datetime import datetime as dt
from ubermodel.plots import get_PLUMBER_plot
from ubermodel.utils import print_good, dataframe_to_rst


def model_site_index_rst(model_dir):
    """list site simulations for a model and print global model plots

    :model_dir: Directory to create a model index in
    """
    time = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')
    name = model_dir.replace('source/models/', '')

    print_good('Generating index for {n}.'. format(n=name))

    model_run_files = sorted(glob.glob(model_dir + '/*.rst'))

    sim_pages = [m.replace('source/models/', '').replace('.rst', '') for m in model_run_files]

    sim_links = '\n'.join(['    %s' % m for m in sim_pages])

    plots = get_PLUMBER_plot(model_dir)

    title = '{name} simulations'.format(name=name)
    title += '\n' + '=' * len(title)

    template = dedent("""
    {title}

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

    rst = template.format(time=time, links=sim_links, title=title, plots=plots)

    with open(model_dir + '.rst', 'w') as f:
        f.write(rst)

    return


def get_metric_data(model_dirs):
    """Get a dataframe of metric means for each model

    :model_dirs: List of model directories to grab data from
    :returns: All metric data as a pandas dataframe

    """
    import re
    import pandas as pd

    pat = re.compile('[^\W_]+(?=_metrics.csv$)')

    data = []

    for md in model_dirs:
        name = md.replace('source/models/', '')
        csv_files = glob.glob(md + '/metrics/*csv')
        if len(csv_files) == 0:
            continue
        model_dfs = []
        for csvf in csv_files:
            site = re.search(pat, csvf).group(0)
            df = pd.DataFrame.from_csv(csvf)
            df['site'] = site
            model_dfs.append(df)
        model_df = pd.concat(model_dfs)
        model_df['name'] = name
        data. append(model_df)
    data = pd.concat(data)

    return data


def get_metric_tables(model_dirs):
    """Get data, average, and return as table.

    :model_dirs: list of model directories to search for metric data
    :returns: metric summary tables as rst source.

    """
    data = get_metric_data(model_dirs).reset_index()

    data = (pd.melt(data, id_vars=['name', 'metric', 'site'])
              .pivot_table(index=['variable', 'name', 'site'],
                           columns='metric', values='value')
              .reset_index())

    summary = data.groupby(['variable', 'name']).mean()

    table_text = ''
    for v in data.variable.unique():
        table = dataframe_to_rst(summary.filter(like=v, axis=0)
                                        .reset_index('variable', drop=True))
        table_text += dedent("""
            {variable}
            -------------

            .. rst-class:: tablesorter

            {table}
            """).format(variable=v, table=table)
        table_text += "\n\n"

    return table_text


def model_search_index_rst():
    """mail model search index
    """

    time = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    print_good('Generating models index')

    model_dirs = [d for d in sorted(glob.glob('source/models/*')) if os.path.isdir(d)]

    table_text = get_metric_tables(model_dirs)

    model_pages = [m.replace('source/', '') for m in model_dirs]
    model_links = '\n'.join(['    %s' % m for m in model_pages])

    template = dedent("""
    Model Search
    =============

    {time}

    Summary tables
    ==============

    {tables}

    Model pages
    ===========

    .. toctree::
        :maxdepth: 1

    {links}
    """)

    with open('source/model_search.rst', 'w') as f:
        f.write(template.format(time=time, links=model_links, tables=table_text))

    return


def newest_file(path):
    """find the most recently modified file in a path

    :path: TODO
    :returns: TODO

    """
    walk = os.walk(path)
    max_age = 0
    for d in walk:
        if len(d[2]) == 0:
            continue
        local_max = max([os.path.getmtime("%s/%s" % (d[0], f)) for f in d[2]])
        max_age = max(max_age, local_max)

    return max_age


def main(args):

    if args['model']:
        if args['<name>'] == 'all':
            model_dirs = [d for d in sorted(glob.glob('source/models/*')) if os.path.isdir(d)]
        else:
            model_dirs = ['source/models/' + args['<name>']]

        for model_dir in model_dirs:
            model_site_index_rst(model_dir)

    if args['all']:
        model_dirs = [d for d in sorted(glob.glob('source/models/*')) if os.path.isdir(d)]
        for model_dir in model_dirs:
            # for 'all' only regenerate pages that need it.
            newest = newest_file(model_dir)
            if newest > os.path.getmtime(model_dir + ".rst"):
                model_site_index_rst(model_dir)
            else:
                print("skipping %s - rst is up to date" % model_dir)

    # Over-all index
    if args['all'] or args['summary']:
        model_search_index_rst()

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
