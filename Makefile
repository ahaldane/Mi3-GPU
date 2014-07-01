
mcmcGPU: mcmcGPU.c common/epsilons.c common/epsilons.h clErrors.h
	module load cuda
	module load gcc
	gcc -O3 -lm -lOpenCL mcmcGPU.c common/epsilons.c -o mcmcGPU

mcmcCPU: mcmcCPU.c common/epsilons.c common/epsilons.h
	gcc -O3 -lm mcmcCPU.c -o mcmcCPU
