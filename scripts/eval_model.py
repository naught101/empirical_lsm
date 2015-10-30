#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: eval_model.py
Author: ned haughton
Email: ned@nedhaughton.com
Description: Evaluates a model (sim or set of sims) and produces rst output with diagnostics

Usage:
    eval_model.py <name> <site> [<file>]

Options:
    -h, --help  Show this screen and exit.
"""

from docopt import docopt

from datetime import datetime as dt

import xray
from matplotlib.cbook import dedent

from pals_utils.constants import DATASETS
from pals_utils.data import get_site_data

from ubermodel.evaluate import evaluate_simulation
from ubermodel.plots import diagnostic_plots
from ubermodel.utils import print_good, dataframe_to_rst
from ubermodel.data import get_sim_nc_path


def model_site_rst_format(model, name, site, eval_text, files):
    """format all the datas into an rst!
    """

    date = dt.isoformat(dt.now().replace(microsecond=0), sep=' ')

    plots = '\n\n'.join([
        ".. image :: {file}".format(file=f) for f in files])

    template = dedent("""
    {name} at {site}
    ====================

    date: :code:`{date}`

    Model details:
    --------------

    :code:`{model}`

    Evaluation results:
    -------------------

    {eval_text}

    Plots:
    ------

    {plots}
    """)

    output = (template.format(model=model,
                              name=name,
                              site=site,
                              plots=plots,
                              date=date,
                              eval_text=eval_text))

    return output


def model_site_rst_write(model, name, site, eval_results, files):
    """run a model and generate an rst file.

    This is useful for importing.

    :model: sklearn-style model or pipeline (regression estimator)
    :name: name of the model
    :site: PALS site name to run the model at
    """
    model_site_rst_file = 'source/models/{n}/{n}_{s}.rst'.format(n=name, s=site)

    print_good("Generating rst file for {n} at {s}.".format(n=name, s=site))

    eval_text = dataframe_to_rst(eval_results)

    output = model_site_rst_format(model, name, site, eval_text, files)

    with open(model_site_rst_file, 'w') as f:
        f.write(output)

    return


def main_eval(name, site, sim_file=None):
    """Main function for evaluating an existing simulation.

    Copies simulation data to source directory.

    :name: name of the model
    :site: PALS site name to run the model at
    :sim_file: Path to simulation netcdf
    """
    nc_path = get_sim_nc_path(name, site)
    if sim_file is not None:
        sim_data = xray.open_dataset(sim_file)
        # WARNING! over writes existing sim!
        sim_data.to_netcdf(nc_path)
    else:
        sim_data = xray.open_dataset(nc_path)

    flux_data = get_site_data([site], 'flux')[site]

    print_good('Evaluating data for {n} at {s}'.format(n=name, s=site))
    eval_results = evaluate_simulation(sim_data, flux_data, name)

    files = diagnostic_plots(sim_data, flux_data, name)

    model_site_rst_write("Not generated", name, site, eval_results, files)

    return


def main(args):
    name = args['<name>']
    site = args['<site>']
    sim_file = args['<file>']

    if site == 'all':
        # will only work if simulations are already run.
        for s in DATASETS:
            main_eval(name, s)
    else:
        main_eval(name, site, sim_file)

    return


if __name__ == '__main__':
    args = docopt(__doc__)

    main(args)
