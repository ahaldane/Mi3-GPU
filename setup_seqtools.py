#!/usr/bin/env python2
from numpy.distutils.core import setup, Extension

# compile me with
# python2 ./setup.py build_ext --inplace

module1 = Extension('seqtools',
                    sources = ['seqtools.c'],
                    extra_compile_args = ['-O3 -Wall'])
                    #extra_compile_args = ['-g -Wall'])

setup (name = 'seqtools',
       version = '1.0',
       description = 'helper functions for sequence analysis',
       ext_modules = [module1])
