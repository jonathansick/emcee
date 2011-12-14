#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
import numpy.distutils.misc_util

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()

algorithms_ext = Extension('pyest.mixtures._algorithms',
                ['pyest/mixtures/_algorithms.c'],
                libraries=['m'])

setup(name='PyEST',
        version='0.1',
        description='pyest',
        author='Daniel Foreman-Mackey',
        author_email='dan@danfm.ca',
        packages=['pyest'],
        ext_modules = [algorithms_ext],
        include_dirs=include_dirs
        )

