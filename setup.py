#!/usr/bin/env python3
import setuptools
import numpy

seqtools_module = setuptools.Extension('mi3gpu.utils.seqtools',
                    sources = ['mi3gpu/utils/seqtools.c'],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args = ['-O3', '-Wall'])

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mi3gpu",
    version_config=True,
    author="Allan Haldane",
    author_email="allan.haldane@temple.edu",
    description="Monte Carlo Inverse Ising Inference on GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahaldane/Mi3-GPU",
    packages=setuptools.find_packages(),
    scripts=['mi3gpu/Mi3.py',
             'mi3gpu/utils/getMarginals.py',
             'mi3gpu/utils/pseudocount.py',
             'mi3gpu/utils/phyloWeights.py',
             'mi3gpu/utils/getSeqEnergies.py',
             'mi3gpu/utils/changeGauge.py',
             'mi3gpu/utils/getXij.py',
             'mi3gpu/utils/exploreParam.py',
             'mi3gpu/utils/alphabet_reduction.py',
             'mi3gpu/utils/apply_alphamap.py',
             'mi3gpu/utils/pre_regularize.py'],
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
        "Operating System :: OS Independent",
    ],
    ext_modules = [seqtools_module],
    python_requires='>=3.6',
    install_requires=[
       'numpy>=1.14',
       'scipy>=1.0.0',
       'pyopencl',
       'configargparse',
    ],
    setup_requires=['setuptools-git-ver'],
)
