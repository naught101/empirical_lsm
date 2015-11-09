#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: drought_workshop.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: Evaluates a model (sim or set of sims) and produces rst output with diagnostics

For CABLE workshop

Usage:
    drought_workshop.py eval <name> <site> [<file>]
    drought_workshop.py rst-gen <name> <site>

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt

import xray
import os
import glob
import pandas as pd

from matplotlib.cbook import dedent
from datetime import datetime as dt

from pals_utils.constants import DATASETS
from pals_utils.data import get_flux_data, get_met_data

from ubermodel.plots import plot_drydown, plot_drydown_daily_cycles, save_plot
from ubermodel.utils import print_good, print_bad, dataframe_to_rst
from ubermodel.data import get_sim_nc_path


def main_eval(name, site, sim_file=None):
    """Main function for evaluating an existing simulation.

    Copies simulation data to source directory.

    TODO: skip running if cached, for easier page regeneration

    :name: name of the model
    :site: PALS site name to run the model at
    :sim_file: Path to simulation netcdf
    """
    try: 
        date_range = (pd.DataFrame.from_csv('data/Ukkola_drought_days_clean.csv')
                              .ix[site, ['start_date', 'end_date']]
                              .values)
    except KeyError as e:
        print_bad("No drydown period found for {n} at {s}. {e}".format(n=name, s=site, e=e))

    print_good("Generating dry-down plots for {n} at {s}.".format(n=name, s=site))

    nc_path = get_sim_nc_path(name, site)
    if sim_file is not None:
        sim_data = xray.open_dataset(sim_file)
        # WARNING! over writes existing sim!
        sim_data.to_netcdf(nc_path)
    else:
        sim_data = xray.open_dataset(nc_path)

    flux_data = get_flux_data(site)
    met_data = get_met_data(site)

    base_path = 'source/models/{n}'.format(n=name)
    rel_path = 'figures/{s}'.format(s=site)
    fig_path = os.path.join(base_path, rel_path)
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    filename = plot_drydown(sim_data, flux_data, met_data, name, date_range)
    save_plot(base_path, rel_path, filename)

    filename = plot_drydown_daily_cycles(sim_data, flux_data, met_data, name, date_range)
    save_plot(base_path, rel_path, filename)

    return


def get_model_drydown_plots(name):
    """Load all drydown plots saved in the evaluation step.
    """
    plot_dir = 'source/models/{n}/figures'.format(n=name)

    plots = {}

    groups = ['timeseries', 'daily_cycles']

    for g in groups:
        matches = glob.glob('{d}/*/{n}_*_drydown_{g}_plot.png'.format(d=plot_dir, n=name, g=g))
        plots[g] = sorted([m.replace(plot_dir, 'figures') for m in matches])

    return plots


def model_drydown_rst_format(model, name, site, eval_text, plot_files):
    """format all the datas into an rst!
    """

    date = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    plots_text = ''
    for group in sorted(plot_files):
        plots_text += "{g}\n".format(g=group)
        plots_text += "^" * len(group) + "\n\n"
        plots_text += '\n\n'.join([
            ".. image :: {file}\n    :width: 200px".format(file=f) for f in plot_files[group]])
        plots_text += '\n\n'

    title = '{name} at {site}'.format(name=name, site=site)
    title += '\n' + '=' * len(title)

    template = dedent("""
    {title}

    date: :code:`{date}`

    Model details:
    --------------

    .. code:: python

      `{model}`

    Evaluation results:
    -------------------

    .. rst-class:: tablesorter

    {eval_text}

    Plots:
    ------

    {plots}
    """)

    output = (template.format(model=model,
                              title=title,
                              plots=plots_text,
                              date=date,
                              eval_text=eval_text))

    return output


def model_drydown_rst_write(model, name, site, eval_results, plot_files):
    """run a model and generate an rst file.

    This is useful for importing.

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at
    """
    model_drydown_rst_file = 'source/models/{n}/{n}_{s}.rst'.format(n=name, s=site)

    print_good("Generating rst file for {n} at {s}.".format(n=name, s=site))

    eval_text = dataframe_to_rst(eval_results)

    output = model_drydown_rst_format(model, name, site, eval_text, plot_files)

    with open(model_drydown_rst_file, 'w') as f:
        f.write(output)

    return


def main_drydown_rst_gen(name, site):
    """Main function for formatting existing simulation evaluations and plots

    Copies simulation data to source directory.

    :name: name of the model
    :site: PALS site name to run the model at
    """

    plot_files = get_model_drydown_plots(name, site)

    model_drydown_rst_write("Not generated", name, site, plot_files)

    return


def main(args):
    name = args['<name>']
    site = args['<site>']
    sim_file = args['<file>']

    if args['eval']:
        if site == 'all':
            # will only work if simulations are already run.
            for s in DATASETS:
                main_eval(name, s)
        else:
            main_eval(name, site, sim_file)

    if args['rst-gen']:
        if site == 'all':
            # will only work if simulations are already evaluated.
            for s in DATASETS:
                main_drydown_rst_gen(name, s)
        else:
            main_drydown_rst_gen(name, site)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
