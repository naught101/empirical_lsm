#!/usr/bin/env python

from distutils.core import setup

setup(name='ubermodel',
      version='0.1',
      description='Empirical land surface model',
      author='ned haughton',
      author_email='ned@nedhaughton.com',
      url='https://bitbucket.org/naught101/empirical_ubermodel',
      packages=['ubermodel'],
      package_data={'': ['data/*.yaml']},
      )
