#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: offline_eval.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Evaluation functions for offline runs
"""

import xarray as xr
import os
import glob

from multiprocessing import Pool

from matplotlib.cbook import dedent
from datetime import datetime as dt

from pals_utils.data import get_sites, get_flux_data

from empirical_lsm.evaluate import evaluate_simulation, load_sim_evaluation
from empirical_lsm.plots import diagnostic_plots
from empirical_lsm.utils import print_good, print_bad, dataframe_to_rst
from empirical_lsm.data import get_sim_nc_path
from empirical_lsm.models import get_model


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

    if hasattr(model, "_var_lags"):
        forcing_vars = model._var_lags
    elif hasattr(model, "forcing_vars"):
        forcing_vars = model.forcing_vars
    else:
        forcing_vars = None

    if model is not None:
        try:
            description = model.description
        except:
            description = "description missing"
        description = dedent("""
            {desc}

            .. code:: python

              {model}
              {fvs}""").format(desc=description, model=model, fvs=forcing_vars)
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


def eval_simulation(name, site, sim_file=None, plots=False, fix_closure=True, qc=True):
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
    except (OSError, RuntimeError) as e:
        print_bad("Sim file ({f}) doesn't exist. What are you doing? {e}".format(f=filename, e=e))
        return

    if sim_file is not None:
        # WARNING! over writes existing sim!
        sim_data.to_netcdf(nc_path)

    flux_data = get_flux_data([site], fix_closure=fix_closure)[site]

    evaluate_simulation(sim_data, flux_data, name, qc=qc)

    if plots:
        diagnostic_plots(sim_data, flux_data, name)

    sim_data.close()

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

    flux_vars = ['Qle', 'Qh', 'NEE']

    for v in flux_vars:
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
    except (OSError, RuntimeError) as e:
        print_bad('one or more files missing for {n} at {s}: {e}'.format(
            n=name, s=site, e=e))
        return

    model_site_rst_write(name, site, eval_results, plot_files)

    return


def eval_simulation_mp(name, site, sim_file=None, plots=False, no_mp=False,
                       fix_closure=True, qc=True):
    """Evaluate using multiple processes if necessary"""
    if site in ['all', 'PLUMBER_ext', 'PLUMBER']:
        # will only work if simulations are already run.
        datasets = get_sites(site)

        if no_mp:
            for s in datasets:
                eval_simulation(name, s, plots=plots, fix_closure=fix_closure, qc=qc)
        else:
            f_args = [[name, s, None, plots, fix_closure, qc] for s in datasets]
            ncores = min(os.cpu_count(), 1 + int(os.cpu_count() * 0.5))
            with Pool(ncores) as p:
                p.starmap(eval_simulation, f_args)

    else:
        eval_simulation(name, site, sim_file, plots=plots, fix_closure=fix_closure, qc=qc)


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
