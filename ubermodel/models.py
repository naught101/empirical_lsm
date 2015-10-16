#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: models.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description:
"""

import yaml

from sklearn.pipeline import make_pipeline


#################
# Model loaders
#################

def get_model(name):
    """return a model as defines in model_search.yaml

    :returns: TODO

    """
    with open('model_search.yaml') as f:
        model_dict = yaml.load(f)[name]

    pipe_list = []

    if 'transforms' in model_dict:
        transforms = model_dict['transforms'].copy()
        if 'scaler' in transforms:
            scaler = transforms.pop('scaler')
            pipe_list.append(get_scaler(scaler))
        if 'pca' in transforms:
            transforms.pop('pca')
            pipe_list.append(get_pca())
        if len(transforms) > 0:
            raise Exception("unknown transforms: %s" % repr(transforms))

    if 'args' in model_dict:
        model = get_model_class(model_dict['class'], model_dict['args'])
    else:
        model = get_model_class(model_dict['class'])
    pipe_list.append(model)

    return make_pipeline(*pipe_list)


def get_scaler(scaler):
    """get a sklearn scaler from a scaler name

    :scaler: TODO
    :returns: TODO

    """
    if scaler == 'standard':
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    if scaler == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()


def get_pca():
    """get a PCA decomposition
    :returns: TODO

    """
    from sklearn.decomposition import PCA
    return PCA()


def get_model_class(class_name, kwargs={}):
    """return a scikit-learn model class, and the required arguments

    :class_name: name of the model class
    :returns: model class object
    """
    # , Perceptron, PassiveAggressiveRegressor
    # , NuSVR, LinearSVR

    if class_name == 'LinearRegression':
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**kwargs)

    if class_name == 'SGDRegressor':
        from sklearn.linear_model import SGDRegressor
        return SGDRegressor(**kwargs)

    if class_name == 'SVR':
        from sklearn.svm import SVR
        return SVR(**kwargs)

    if class_name == 'DecisionTreeRegressor':
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(**kwargs)

    if class_name == 'ExtraTreesRegressor':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**kwargs)

    if class_name == 'KNeighborsRegressor':
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**kwargs)

    if class_name == 'MultilayerPerceptronRegressor':
        from sklearn.neural_network import MultilayerPerceptronRegressor
        # This is from a pull request: https://github.com/scikit-learn/scikit-learn/pull/3939
        return MultilayerPerceptronRegressor(**kwargs)

    raise Exception("Unknown Model class")
