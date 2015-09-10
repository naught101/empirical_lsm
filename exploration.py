# coding: utf-8

# # Model exploration
#
# ## Todo
#
# - add more metrics
#     - mutual info score
# - multi variate output
# - table of results
# - Rhys: Compare the functional form of empirical models to that of LSMs, see where they differ
#     - multivariate functional form
# -
#

# In[ ]:

import numpy as np
import pylab as pl
import xray
import pandas as pd
import os
import joblib
import pickle
import time

from numbers import Number
from collections import OrderedDict


# In[ ]:

import pals_utils as pu
from pals_utils.stats import metrics


# In[ ]:

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


# In[ ]:

from sklearn.linear_model import LinearRegression, Perceptron, SGDRegressor, LogisticRegression, PassiveAggressiveRegressor
from sklearn.svm import SVR, NuSVR  #, LinearSVR
# from sklearn.neural_network import MultilayerPerceptronRegressor # This is from a pull request: https://github.com/scikit-learn/scikit-learn/pull/3939
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:

met_vars = ["SWdown", "Tair", "LWdown", "Wind", "Rainf", "PSurf", "Qair"]
met_data = xray.open_dataset("/home/naught101/phd/data/PALS/datasets/met/TumbaFluxnet.1.4_met.nc")
met_df = met_data.to_dataframe().reset_index(["x", "y", "z"]).ix[:, met_vars]

flux_vars = ["Qh", "Qle", "Rnet", "NEE"]
flux_data = xray.open_dataset("/home/naught101/phd/data/PALS/datasets/flux/TumbaFluxnet.1.4_flux.nc")
flux_df = flux_data.to_dataframe().reset_index(["x", "y"]).ix[:, flux_vars]


# In[ ]:

if not os.path.exists("cache/"):
    os.mkdir("cache")
cache = pd.HDFStore("cache/cache.hdf5")


# ### functions

# In[ ]:


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        # print(f.__name__, "took: {:2.4f} sec".format(te-ts))
        return (result, te - ts)
    return timed


# In[ ]:

@timeit
def fit_pipeline(pipe, X, Y):
    pipe.fit(X, Y)


@timeit
def get_pipeline_prediction(pipe, X):
    return(pipe.predict(X))


def get_pipeline_name(pipe, suffix=None):
    if suffix is not None:
        return ", ".join(list(pipe.named_steps.keys()) + [suffix])
    else:
        return ", ".join(pipe.named_steps.keys())


# In[ ]:

def plot_test_data(Y_pred, Y_validate, y_var):
    # Sample plot
    plot_data = pd.DataFrame({y_var + "_obs": Y_validate, y_var + "_pred": Y_pred})

    # week 7 raw
    pl.plot(plot_data[(70 * 48):(77 * 48)])
    pl.legend(plot_data.columns)
    pl.show()

    # fornightly rolling mean
    pl.plot(pd.rolling_mean(plot_data, window=14 * 48))
    pl.legend(plot_data.columns)
    pl.show()

    # daily cycle
    pl.plot(plot_data.groupby(np.mod(plot_data.index, 48)).mean())
    pl.legend(plot_data.columns)
    pl.show()


# In[ ]:

def run_metrics(Y_pred, Y_validate, metrics):
    metric_data = OrderedDict()
    for (n, m) in metrics.items():
        metric_data[n] = m(Y_pred, Y_validate)
    return metric_data


# In[ ]:

def test_pipeline(pipe, X=met_df, Y=flux_df, y_var=["Qh"], name=None, plot=False, cache=cache, clear_cache=False):
    """Top-level pipeline fitter and tester.

    Fits and predicts with a model, runs metrics, optionally runs some diagnostic plots.
    """

    if name is None:
        name = get_pipeline_name(pipe)

    if "metric_data" in cache and not clear_cache:
        if name in cache.metric_data.index:
            print("Metrics already calculated for %s, skipping." % name)
            return
        metric_data = cache.metric_data
    else:
        metric_data = pd.DataFrame()

    Y = np.array(Y[y_var])

    train_len = (7 * len(X) // 10)

    # X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, train_size=0.7, random_state=0)
    X_train = X[:train_len]
    X_validate = X[train_len:]
    Y_train = Y[:train_len]
    Y_validate = Y[train_len:]

    if "predictions/" + name in cache and not clear_cache:
        print("prediction already run for %s, skipping fit and predict" % name)
        Y_pred = np.array(cache["predictions"][name])
    else:
        # Fit model
        metric_data.ix[name, "t_fit"] = fit_pipeline(pipe, X_train, Y_train)[1]

        # Run model
        (Y_pred, metric_data.ix[name, "t_pred"]) = get_pipeline_prediction(pipe, X_validate)
        # Some sklearn models return vector (n,) inputs as 2D arrays (n,1)
        if len(Y_pred.shape) > 1:
            Y_pred = Y_pred[:, 0]

        cache.put("predictions/" + name, pd.DataFrame(Y_pred))

    for k, v in run_metrics(Y_pred, Y_validate, metrics).items():
        metric_data.ix[name, k] = v
    cache["metric_data"] = metric_data
    cache.flush()

    # Plotting
    if plot:
        [print("{:>10}:".format(k), "{:.3f}".format(v) if isinstance(v, Number) else v)
         for (k, v) in metric_data.items()]

        plot_test_data(Y_pred, Y_validate, y_var)


# ########################
# ## fit - run - assess
# ########################

def get_model_fit_path(hash):
    return "cache/model_fits/%s.pickle" % hash


def fit_exists(pipe, land_data, cache):
    model_fit_id = joblib.hash((pipe, land_data))

    if "model_fits/" + model_fit_id in cache:
        return cache["model_fits"][model_fit_id]["model_fit_hash"]


def fit_model_pipeline(pipe, land_data, name=None, cache=cache, clear_cache=False):
    """Top-level pipeline fitter.

    Fits a model, stores model and metadata.

    TODO: store domain metadata

    returns (pipe, model_hash)
    """

    if name is None:
        name = get_pipeline_name(pipe)

    model_fit_id = joblib.hash((pipe, land_data))

    model_hash = fit_exists(pipe, land_data, cache)
    if model_hash is not None and not clear_cache:
        print("Model %s already fitted for %s, loading from file." % name, land_data.name)
        with open(get_model_fit_path(model_hash), "rb") as f:
            pipe = pickle.load(f)
    else:
        if land_data.met is None or land_data.flux is None:
            raise KeyError("missing met or flux data")
        fit_time = fit_pipeline(pipe, land_data.met, land_data.flux)[1]
        model_hash = joblib.hash(pipe)
        cache["model_fits"][model_fit_id]["model_fit_hash"] = model_hash
        cache["model_fits"][model_fit_id]["model_fit_time"] = fit_time
        cache.flush()
        with open(get_model_fit_path(model_hash), "wb") as f:
            pickle.dump(pipe, f)

    return pipe, model_hash


# Simulate

def get_sim_path(hash):
    return "cache/simulations/%s.pickle" % hash


def sim_exists(pipe, land_data, cache):
    model_sim_id = joblib.hash((pipe, land_data))

    return "simulations/" + model_sim_id in cache


def simulate_model_pipeline(pipe, land_data, name=None, cache=cache, clear_cache=False):
    """Top-level pipeline predictor.

    runs model, caches model simulation.

    returns (sim_data, sim_hash)
    """

    if name is None:
        name = get_pipeline_name(pipe)

    model_sim_id = joblib.hash((pipe, land_data))

    sim_hash = sim_exists(pipe, land_data, cache)
    if sim_hash and not clear_cache:
        print("Model %s already simulated for %s, loading from file." % name, land_data.name)
        with open(get_sim_path(sim_hash), "rb") as f:
            sim_data = pickle.load(f)
    else:
        if land_data.met is None or land_data.flux is None:
            raise KeyError("missing met or flux data")
        sim_data = LandData("%s_%s" % (get_pipeline_name(pipe), model_sim_id),
                            land_data.domain_type, land_data.geo)
        (sim_data.flux, fit_time) = get_pipeline_prediction(pipe, land_data.met, land_data.flux)
        # TODO: If a simulation can produce more than one output for a given input, this won"t be unique. Is that ok?
        sim_hash = joblib.hash(sim_data)
        cache["simulations"][model_sim_id]["sim_hash"] = sim_hash
        cache["simulations"][model_sim_id]["model_predict_time"] = fit_time
        cache.flush()
        with open(get_model_fit_path(sim_hash), "wb") as f:
            pickle.dump(sim_data, f)

    return sim_data, sim_hash


# Evaluate

# TODO: Could make a wrapper around this so that you can just pass a hash, or a fit model, and auto-load the data.

def evaluation_exists(sim_data, land_data, cache):
    eval_hash = joblib.hash((sim_data, land_data))

    if ("metric_data" in cache) and (eval_hash in cache.metric_data.index[0]):
        return eval_hash


def evaluate_simulation(sim_data, land_data, y_vars, name, cache=cache, clear_cache=False):
    """Top-level simulation evaluator.

    Compares sim_data to land_data, using standard metrics. Stores the results in an easily accessible format.
    """

    eval_hash = joblib.hash((sim_data, land_data))

    index = {"eval_hash": eval_hash,
             "name": name,
             "site": land_data.name,
             "var": y_vars[0]}

    if "metric_data" in cache and not clear_cache:
        if eval_hash in cache.metric_data.index[0]:
            print("Metrics already calculated for %s, skipping." % name)
            return cache.metric_data
        metric_data = cache.metric_data
    else:
        metric_data = pd.DataFrame([index])
        metric_data = metric_data.set_index(list(index.keys()))

    for y_var in y_vars:
        Y_sim = np.array(sim_data.flux[y_var])
        Y_obs = np.array(land_data.flux[y_var])

        row_id = tuple(list(index.values())[0:3] + [y_var])
        metric_data.ix[row_id, "name"] = "%s_%s" % (y_var)
        metric_data.ix[row_id, "sim_id"] = joblib.hash(sim_data)
        metric_data.ix[row_id, "site"] = land_data.name
        metric_data.ix[row_id, "var"] = y_var

        for k, v in run_metrics(Y_sim, Y_obs, metrics).items():
            metric_data.ix[row_id, k] = v

    cache["metric_data"] = metric_data
    cache.flush()

    return metric_data.loc[eval_hash]


# In[ ]:

def test_pipeline(pipe, land_data, y_var=["Qh"], name=None, plot=False, cache=cache, clear_cache=False):
    """Top-level pipeline fitter and tester.

    Fits and predicts with a model, runs metrics, optionally runs some diagnostic plots.
    """

    (train_data, test_data) = land_data.time_split(0.7)

    fit_hash = joblib.hash(pipe, train_data)




# ## Site setup

# In[ ]:

class LandData():
    """Land data storage mechanism.

    Stores met data, flux data, and domain data.

    TODO: This could be used to store model output too...
    """

    def __init__(self, name, domain_type, geo, met=None, flux=None, veg=None, soil=None):
        self.name = name
        self.domain_type = domain_type
        self.geo = geo
        self.met = met
        self.flux = flux
        self.veg = veg
        self.soil = soil

    def copy_data(self, met=None, flux=None):
        """Return a copy of the land dataset.

        met and flux components optional: use self.met, self.flux
        """
        return LandData(self.name, self.domain_type, self.geo, met, flux, veg, soil)

    def time_split(self, first_len):
        first = self.copy_metadata()
        second = self.copy_metadata()
        if self.met is not None:
            first.met = self.met[:frist_len]
            first.met = self.met[:first_len]




# In[ ]:

site_data = LandData("Tumbarumba", "site", {"lat": 151, "long": -34})
site_data


# ## Linear regression
#
# - insensitive to scaling or PCA

# In[ ]:

pipe = make_pipeline(LinearRegression())
test_model_pipeline(pipe, clear_cache=True)


# In[ ]:

pipe = make_pipeline(LinearRegression())
test_pipeline(pipe, clear_cache=True)


# In[ ]:

# pipe = make_pipeline(StandardScaler(), LinearRegression())
# test_pipeline(pipe)


# In[ ]:

# pipe = make_pipeline(PCA(), LinearRegression())
# test_pipeline(pipe)


# In[ ]:

# pipe = make_pipeline(StandardScaler(), PCA(), LinearRegression())
# test_pipeline(pipe)


# ## Polynomial regression
#
# - Only a slight improvement
#     - because non-linearities are localised?

# In[ ]:

pipe = make_pipeline(PolynomialFeatures(2), LinearRegression())
test_pipeline(pipe, name=get_pipeline_name(pipe, "poly2"))


# In[ ]:

pipe = make_pipeline(PolynomialFeatures(5), LinearRegression())
test_pipeline(pipe, name=get_pipeline_name(pipe, "poly5"))


# In[ ]:

met_df_with_lag = pd.concat([met_df, met_df.diff()], axis=1).dropna()
met_df_with_lag.shape


# In[ ]:

np.linalg.matrix_rank(np.array(met_df_with_lag[:40000]))


# In[ ]:

flux_df.shape


# In[ ]:

flux_df[1:40001].shape


# In[ ]:

pipe = make_pipeline(LinearRegression())
test_pipeline(pipe, X=met_df_with_lag[:40000], Y=flux_df[1:40001], name=get_pipeline_name(pipe, "lag1"))


# ## SGD
#
# - very sensitive to scaling. Not sensitive to PCA

# In[ ]:

# pipe = make_pipeline(SGDRegressor())
# test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), SGDRegressor())
test_pipeline(pipe)


# In[ ]:

# pipe = make_pipeline(PCA(), SGDRegressor())
# test_pipeline(pipe)


# In[ ]:

# pipe = make_pipeline(StandardScaler(), PCA(), SGDRegressor())
# test_pipeline(pipe)


# In[ ]:

# test_model("LogisticRegression", LogisticRegression())


# In[ ]:

# test_model("PassiveAggressiveRegressor", PassiveAggressiveRegressor())


# ## Support Vector Machines
#
# - Sensitive to scaling, not to PCA

# In[ ]:

# pipe = make_pipeline(SVR())
# test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), SVR())
test_pipeline(pipe)


# In[ ]:

# pipe = make_pipeline(StandardScaler(), PCA(), SVR())
# test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), SVR(kernel="poly"))
#
test_pipeline(pipe, name=get_pipeline_name(pipe, "polykernel"))


# ## Multilayer Perceptron

# In[ ]:

pipe = make_pipeline(MultilayerPerceptronRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(PCA(), MultilayerPerceptronRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), PCA(), MultilayerPerceptronRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(activation="logistic"))
test_pipeline(pipe, get_pipeline_name(pipe, "logisitic"))


# In[ ]:

pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,20,)))
test_pipeline(pipe, get_pipeline_name(pipe, "[20,20,20]"))


# In[ ]:

pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(10,10,)))
test_pipeline(pipe, get_pipeline_name(pipe, "[10,10]"))


# In[ ]:

pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(10,30,)))
test_pipeline(pipe, get_pipeline_name(pipe, "[10,30]"))


# In[ ]:

pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,)))
test_pipeline(pipe, get_pipeline_name(pipe, "[20,20]"))


# ## K-nearest neighbours
#
# - Not sensitive to scaling or PCA

# In[ ]:

pipe = make_pipeline(KNeighborsRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(PCA(), KNeighborsRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1000))
test_pipeline(pipe, get_pipeline_name(pipe, "1000 neighbours"))


# ## Decision Trees

# In[ ]:

pipe = make_pipeline(DecisionTreeRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(ExtraTreesRegressor())
test_pipeline(pipe)


# In[ ]:

pipe = make_pipeline(StandardScaler(), PCA(), ExtraTreesRegressor())
test_pipeline(pipe)


# # Metric results

# In[ ]:

cache.metric_data


# In[ ]:

normed_metrics = cache.metric_data - cache.metric_data.min()
normed_metrics /= normed_metrics.max()


# In[ ]:

normed_metrics.columns


# In[ ]:

normed_metrics[["corr", "nme", "mbe", "sd_diff"]].plot(kind="bar")


# In[ ]:

normed_metrics[["extreme_5", "extreme_95"]].plot(kind="bar")


# In[ ]:

cache


# In[ ]:

get_ipython().magic("pinfo cache")


# In[ ]:

pipe


# In[ ]:

get_ipython().magic("pinfo pd.DataFrame.values")


# In[ ]:

get_ipython().magic("pinfo base_repr")


# In[ ]:

np.unsignedinteger(1)


# In[ ]:

hash(KNeighborsRegressor(n_neighbors=1000))


# In[ ]:

hash(object())


# In[ ]:

import hashlib


# In[ ]:

joblib.hash(str(pipe.get_params()))


# In[ ]:

l=pipe.named_steps["linearregression"]
l.score()


# In[ ]:




# In[ ]:




# In[ ]:

a = np.array([321,12.3,1,1.4,1])


# In[ ]:

hash(a)


# In[ ]:

cache


# In[ ]:



