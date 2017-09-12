import glob
import os
import pandas as pd

from multiprocessing import Pool

from matplotlib.cbook import dedent
from datetime import datetime as dt
from empirical_lsm.evaluate import get_metric_data
from empirical_lsm.plots import get_PLUMBER_plot
from empirical_lsm.utils import print_good, print_warn, dataframe_to_rst
from empirical_lsm.models import get_model


def model_site_index_rst(name):
    """list site simulations for a model and print global model plots

    :model_dir: Directory to create a model index in
    """
    time = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    model_dir = 'source/models/' + name

    try:
        model = get_model(name)
        try:
            description = model.description
        except:
            description = "description missing"
        description = dedent("""
            {desc}

            .. code:: python

              `{model}`""").format(desc=description, model=model)
    except:
        description = "Description missing - unknown model"

    print_good('Generating index for {n}.'. format(n=name))

    model_run_files = sorted(glob.glob('{d}/{n}*.rst'.format(d=model_dir, n=name)))

    sim_pages = [os.path.splitext(os.path.basename(m))[0] for m in model_run_files]

    sim_links = '\n'.join(['    {0}'.format(m) for m in sorted(sim_pages)])

    try:
        get_PLUMBER_plot(model_dir)
        plot_files = glob.glob('{d}/figures/*.png'.format(d=model_dir))
        rel_paths = ['figures/' + os.path.basename(p) for p in plot_files]
        plots = '\n\n'.join([
            ".. image :: {file}\n    :width: 300px".format(file=f) for f in sorted(rel_paths)])
    except AttributeError as e:
        print_warn('No plots found, skipping {n}: {e}'.format(n=name, e=e))
        return

    title = '{name} simulations'.format(name=name)
    title += '\n' + '=' * len(title)

    template = dedent("""
    {title}

    {description}

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

    rst = template.format(time=time,
                          description=description,
                          links=sim_links,
                          title=title,
                          plots=plots)

    rst_index = model_dir + '/index.rst'

    with open(rst_index, 'w') as f:
        f.write(rst)

    return


def get_metric_tables(names):
    """Get data, average, and return as table.

    :names: list of models to search for metric data
    :returns: metric summary tables as rst source.

    """
    data = get_metric_data(names).reset_index()

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
    """main model search index
    """

    time = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    print_good('Generating models index')

    names = get_available_models()

    table_text = get_metric_tables(names)

    model_pages = ['models/' + n + '/index' for n in names]
    model_links = '\n'.join(['    %s' % m for m in sorted(model_pages)])

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

    :path: directory under which to search
    :returns: newest file mtime

    """
    walk = os.walk(path)
    max_mtime = 0
    for d in walk:
        if len(d[2]) == 0:
            continue
        local_max = max([os.path.getmtime("%s/%s" % (d[0], f)) for f in d[2]])
        max_mtime = max(max_mtime, local_max)

    return max_mtime


def model_site_index_as_needed(name, rebuild=False):
    """build index only if necessary

    :model_dir: TODO
    :returns: TODO

    """
    model_dir = "source/models/%s" % name
    rst_path = "{d}/index.rst".format(d=model_dir)

    if rebuild:
        # Path needs (re)building
        model_site_index_rst(name)
    else:
        if not os.path.exists(rst_path):
            model_site_index_rst(name)
        else:
            # for 'all' only regenerate pages that need it.
            newest = newest_file(model_dir)
            # TODO: this doesn't seem to work..
            if newest > os.path.getmtime(rst_path):
                model_site_index_rst(name)
            else:
                print("skipping %s - rst is up to date" % model_dir)


def get_available_model_dirs():
    """Searches for existing model output, and returns model paths."""
    return [d for d in sorted(glob.glob('source/models/*')) if os.path.isdir(d)]


def get_available_models():
    """Searches for existing model output, and returns model names."""
    return [d.replace('source/models/', '') for d in get_available_model_dirs()]


def model_site_index_rst_mp(names, rebuild=False, no_mp=False):
    """(Re)build site index rst files as needed, with multiple processors"""
    if no_mp:
        for n in names:
            model_site_index_as_needed(n, rebuild)
    else:
        ncores = min(os.cpu_count(), 1 + int(os.cpu_count() * 0.5))
        f_args = zip(names, [rebuild] * len(names))
        with Pool(ncores) as p:
            p.starmap(model_site_index_as_needed, f_args)
