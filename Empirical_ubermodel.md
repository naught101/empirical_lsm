---
title: "Empirical Ubermodel: Estimating the maximum available information in met data"
author:
  - N. Haughton
  - G. Abramowitz
  - A. J. Pitman
date: \today{}
geometry: margin=2cm
header-includes:
    - \usepackage{lineno}
    - \linenumbers
abstract: >
    TODO
---

*For submission to Journal of Hydrometeorology.*

Introduction
============

- Land surface models aren't performing well [@PLUMBER, @Haughton]
    - Partly due to their complexity - at least that's a main reason why we can't figure out the problems.
- We don't know how well they aren't performing - @PLUMBER put a lower bound on the performance improvement we should expect, but we don't know how much available information there is in the met forcings...
    - empirical models can help us quantify the maximum available information
        - But they don't let us make inferences about the physics.
- LSMs are partially empirical, and complex enough to suffer from equifinality - it is not certain that what they are producing is physical, and it is not certain that they will accurately represent out-of-sample behaviour.
    - Empirical models *will* accurately represent out-of-sample data, as long as:
        - They are trained on representative data (e.g. behaviours that occur in the test set also occur in the training set).

Aim
===

1. To produce a model/data transformation that maximises the use of available data to predict climatological fluxes.
By any means necessary.
2. To test run the model in an AMIP/GSWP context, and assess the results.

Methodology
===========

*for full lists see possibilities.yaml*

Inputs
------

Propose to use all of the available Met data: (SWdown, Tair, LWdown, Wind, Rainf, PSurf, Qair)
Any other data we can find (e.g. satellite derived stuff)

Data transformations
--------------------

Sometimes it makes sense to transform the data, and some models require particular transformations.
Likely transformations include normalisation (scaling/shifting), dimensionality reduction, dimensionality production (e.g. support vector machines), log/exp/power transforms, lagging, window-averaging

Outputs
-------

Major climatological fluxes: (Qh, Qle, Rnet, Qg, NEE, GPP, Runoff, Soil moisture)

Model selection
---------------

Any relevant empirical model (regressions, classification + regression, neural networks (back propagation))

TODO: Mine scikit-learn and other packages for model ideas.

### Candidates:

- Linear regression
    - Polynomial regression
- Cluster + piecewise regression
    - some calibration for optimal cluster size?
- Support vector machines
- Neural networks - multilayer perceptron
    - Requires calibration
- Decision trees/forests
    - volatile

### Testing

Start with PLUMBER-style cross-validation - leave one out.

For models that require a testing set as well as a validation set (e.g. neural nets trained with backpropagation), use one of the training sites as a testing set.

Just to make a point, also do train and test using *just* the most extreme sites, one at a time (train and test within that site, if required).

Results
=======


Implementation in a GCM
=======================

Methods
-------

Take which ever model(s) are selected as the "best" models, with their optimal global calibration, and implement those offline over a gridded dataset (e.g. GSWP3).

If that works, implement the model as a land surface scheme in a coupled model (e.g. ACCESS)

Results
-------

Discussion
==========

Bibliography
============

