
## layouts

layouts are model structural definitions without set parameters

layout
: Scikit-learn estimator/pipeline or object with similar fit/predict methods
layout_hash
: short_hash of layout
layout_path
: path to pickled layout


## datasets

A met and/or flux dataset. Either an xray dataset, or a list of xray datasets

dataset
: either an xray dataset or a list of datasets
site_id
: name_version, e.g. "TumbaFluxnet_1.4", or an arbitrary unique name (e.g. PLUMBER_leave_one_out)
site_path
: path to dataset (may not exist if list?)

## fits

a fit is a model layout with parameters fit to a given met/flux dataset.

fit
: scikit-learn estimator/pipline like object, with set parameters
fit_id
: unique identifier: layouthash_siteid
layout_hash
: can calculate the layout path from this.
site_id
: can reference the site data from this
fit_hash
: shorthash of the fit (unique to layout/data combo)
fit_path
: path to pickled fit

## simulations

a simulation is an xray dataset

sim_id
: unique identifier: fithash_siteid
fit_hash
: can look up the fit metadata with this, or load the fit from a calculated path
site_id
: can reference the site_data from this
sim_hash
: shorthash of the xray dataset
sim_path
: path to netcdf file where sim data is stored.


## Evaluations

Evaluation of a variable of a dataset (or multiple variables?)

ev_id
: simhash_var - unique identifier
sim_hash
: can look up simulatio data with this
site_id
: can look up site_data with this
fit_hash
: can loop up model structure/parameters with this
variable
: variable(s) being assessed
<metrics>
: multiple columns, one for each metric
other metatdata?
: what else might we need?

