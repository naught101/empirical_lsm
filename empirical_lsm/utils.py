#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: utils.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description: empirical_lsm helper functions
"""

from tabulate import tabulate


def print_good(string, **kwargs):
    """print string in green
    """
    okgreen = '\033[92m'
    reset = '\033[39m'

    print(okgreen + string + reset, **kwargs)


def print_warn(string, **kwargs):
    """print string in yellow
    """
    warnyellow = '\033[93m'
    reset = '\033[39m'

    print(warnyellow + string + reset, **kwargs)


def print_bad(string, **kwargs):
    """print string in red
    """
    badred = '\033[91m'
    reset = '\033[39m'

    print(badred + string + reset, **kwargs)


def dataframe_to_rst(dataframe):
    """Format eval results in rst format
    """
    return tabulate(dataframe.round(4), headers='keys', tablefmt='rst')
