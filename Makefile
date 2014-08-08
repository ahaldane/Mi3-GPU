
all: mcmcCPUgen mcmcCPUbin

mcmcCPUbin: mcmcCPUbin.c common/stdPanic.c
	gcc -O3 -lm mcmcCPUbin.c common/stdPanic.c -o mcmcCPUbin

mcmcCPUgen: mcmcCPUgen.c common/epsilons.c common/epsilons.h
	gcc -O3 -lm mcmcCPUgen.c common/epsilons.c -o mcmcCPUgen
