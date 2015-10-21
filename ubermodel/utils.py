#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: utils.py
Author: ned haughton
Email: ned@nedhaughton.com
Github: https://github.com/naught101
Description: ubermodel helper functions
"""


def print_good(string):
    """print string in green

    :string: TODO
    :returns: TODO

    """

    okgreen = '\033[92m'
    reset = '\033[39m'

    print(okgreen + string + reset)
