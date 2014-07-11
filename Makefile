
all: mcmcCPUgen

mcmcCPUgen: mcmcCPUgen.c common/epsilons.c common/epsilons.h
	gcc -O3 -lm mcmcCPUgen.c common/epsilons.c -o mcmcCPUgen
