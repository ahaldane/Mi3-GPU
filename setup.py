#!/usr/bin/env python3
import setuptools
import numpy

seqtools_module = setuptools.Extension('mi3gpu.utils.seqtools',
                    sources = ['mi3gpu/utils/seqtools.c'],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args = ['-O3', '-Wall'])


setuptools.setup(
    ext_modules = [seqtools_module],
    #setup_requires=['setuptools-git-ver'],
)
