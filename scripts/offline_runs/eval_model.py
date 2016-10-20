#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: eval_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: Evaluates a model (sim or set of sims) and produces rst output with diagnostics

Usage:
    eval_model.py eval <name> <site> [<file>] [--no-mp] [--no-plots]
    eval_model.py rst-gen <name> <site> [--no-mp]

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt

import xarray as xr
import os
import glob

from multiprocessing import Pool

from matplotlib.cbook import dedent
from datetime import datetime as dt

from pals_utils.constants import FLUX_VARS
from pals_utils.data import get_site_data

from ubermodel.evaluate import evaluate_simulation, load_sim_evaluation
from ubermodel.plots import diagnostic_plots
from ubermodel.utils import print_good, print_bad, dataframe_to_rst
from ubermodel.data import get_sites, get_sim_nc_path
from ubermodel.models import get_model


def model_site_rst_format(name, site, eval_text, plot_files):
    """format all the datas into an rst!
    """

    date = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    plots_text = ''
    for group in sorted(plot_files):
        plots_text += "{g}\n".format(g=group)
        plots_text += "^" * len(group) + "\n\n"
        plots_text += '\n\n'.join([
            ".. image :: {file}\n    :width: 200px".format(file=f) for f in sorted(plot_files[group])])
        plots_text += '\n\n'

    title = '{name} at {site}'.format(name=name, site=site)
    title += '\n' + '=' * len(title)

    try:
        model = get_model(name)
    except:
        model = None

    if model is not None:
        try:
            description = model.description
        except:
            description = "description missing"
        description = dedent("""
            {desc}

            .. code:: python

              `{model}`""").format(desc=description, model=model)
    else:
        description = "Description missing - unknown model"

    template = dedent("""
    {title}

    date: :code:`{date}`

    Model details:
    --------------

    {desc}

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
                              desc=description,
                              plots=plots_text,
                              date=date,
                              eval_text=eval_text))

    return output


def model_site_rst_write(name, site, eval_results, plot_files):
    """run a model and generate an rst file.

    This is useful for importing.

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at
    """
    model_site_rst_file = 'source/models/{n}/{n}_{s}.rst'.format(n=name, s=site)

    print_good("Generating rst file for {n} at {s}.".format(n=name, s=site))

    eval_text = dataframe_to_rst(eval_results)

    output = model_site_rst_format(name, site, eval_text, plot_files)

    with open(model_site_rst_file, 'w') as f:
        f.write(output)

    return


def main_eval(name, site, sim_file=None, plots=True):
    """Main function for evaluating an existing simulation.

    Copies simulation data to source directory.

    TODO: skip running if cached, for easier page regeneration

    :name: name of the model
    :site: PALS site name to run the model at
    :sim_file: Path to simulation netcdf
    """
    nc_path = get_sim_nc_path(name, site)

    if sim_file is None:
        filename = nc_path
    else:
        filename = sim_file

    try:
        sim_data = xr.open_dataset(filename)
    except OSError as e:
        print_bad("Sim file ({f}) doesn't exist. What are you doing? {e}".format(f=filename, e=e))
        return

    if sim_file is not None:
        # WARNING! over writes existing sim!
        sim_data.to_netcdf(nc_path)

    flux_data = get_site_data([site], 'flux')[site]

    evaluate_simulation(sim_data, flux_data, name)

    if plots:
        diagnostic_plots(sim_data, flux_data, name)

    return


def get_existing_plots(name, site):
    """Load all plots saved in the evaluation step.

    :name: model name
    :site: site name
    :returns: list of existing plots

    """
    plot_dir = 'source/models/{n}/figures'.format(n=name)

    plots = {}

    plot_name = '{d}/{n}_all_PLUMBER_plot_all_metrics.png'.format(d=plot_dir, n=name, s=site)
    if os.path.exists(plot_name):
        plots['All variables'] = ['figures/{p}'.format(p=os.path.basename(plot_name))]

    for v in FLUX_VARS:
        plots[v] = ['figures/{s}/{p}'.format(s=site, p=os.path.basename(p)) for p in
                    sorted(glob.glob('{d}/{s}/{n}_{v}_{s}_*.png'.format(d=plot_dir, n=name, v=v, s=site)))]

    return plots


def main_rst_gen(name, site):
    """Main function for formatting existing simulation evaluations and plots

    Copies simulation data to source directory.

    :name: name of the model
    :site: PALS site name to run the model at
    """

    try:
        eval_results = load_sim_evaluation(name, site)
        plot_files = get_existing_plots(name, site)
    except OSError as e:
        print_bad('one or more files missing for {n} at {s}: {e}'.format(
            n=name, s=site, e=e))
        return

    model_site_rst_write(name, site, eval_results, plot_files)

    return


def main_eval_mp(name, site, sim_file=None, plots=True, no_mp=False):
    """Evaluate using multiple processes if necessary"""
    if site in ['all', 'PLUMBER_ext']:
        # will only work if simulations are already run.
        datasets = get_sites('PLUMBER_ext')

        if no_mp:
            for s in datasets:
                main_eval(name, s, plots=plots)
        else:
            f_args = [[name, s, None, plots] for s in datasets]
            ncores = min(os.cpu_count(), 1 + int(os.cpu_count() * 0.5))
            with Pool(ncores) as p:
                p.starmap(main_eval, f_args)

    else:
        main_eval(name, site, sim_file, plots)


def main_rst_gen_mp(name, site, sim_file=None, no_mp=False):
    """Generate rst files using multiple processes if necessary

    :name: TODO
    :site: TODO
    :sim_file: TODO
    :returns: TODO

    """
    if site in ['all', 'PLUMBER_ext']:
        # will only work if simulations are already evaluated.
        datasets = get_sites('PLUMBER_ext')

        if no_mp:
            for s in datasets:
                main_rst_gen(name, s)
        else:
            f_args = [[name, s] for s in datasets]
            ncores = min(os.cpu_count(), 2 + int(os.cpu_count() * 0.25))
            with Pool(ncores) as p:
                p.starmap(main_rst_gen, f_args)

    else:
        main_rst_gen(name, site)


def main(args):
    name = args['<name>']
    site = args['<site>']
    sim_file = args['<file>']
    plots = not args['--no-plots']

    if args['eval']:
        main_eval_mp(name, site, sim_file, plots, args['--no-mp'])
    if args['rst-gen']:
        main_rst_gen_mp(name, site, sim_file, args['--no-mp'])

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
