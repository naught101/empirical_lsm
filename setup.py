#!/usr/bin/env python

from distutils.core import setup

setup(name='empirical_lsm',
      version='0.1',
      description='Empirical land surface model',
      author='ned haughton',
      author_email='ned@nedhaughton.com',
      url='https://bitbucket.org/naught101/empirical_lsm',
      packages=['empirical_lsm'],
      package_data={'': ['data/*.yaml']},
      )
