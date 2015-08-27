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



Model selection
===============

Methods
-------

Propose to use all of the available Met data: (SWdown, Tair, LWdown, Wind, Rainf, PSurf, Qair)

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

### Data transformations

- 

Results
-------

Implementation in a GCM
=======================

Methods
-------

Results
-------

Discussion
==========

Bibliography
============

