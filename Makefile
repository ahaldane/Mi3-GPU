all: seqtools mcmc_helpers

seqtools:
	python2 ./setup_seqtools.py build_ext --inplace

mcmc_helpers:
	python2 ./setup_mcmc_helpers.py build_ext --inplace

mcmcCPUgen: mcmcCPUgenThreaded.c common/epsilons.c common/epsilons.h
	gcc -O3 -lm -pthread mcmcCPUgenThreaded.c common/epsilons.c -o mcmcCPUgen
