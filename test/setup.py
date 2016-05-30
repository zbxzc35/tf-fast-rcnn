#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(cmdclass = {'build_ext': build_ext}, ext_modules=[Extension("roi_pool_c", sources=["roi_pool_sing.pyx", "roi_pool.c"], include_dirs=[numpy.get_include()])])
