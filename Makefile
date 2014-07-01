
mcmcGPU: mcmcGPU.c common/epsilons.c common/epsilons.h clErrors.h
	@echo -e "Remember to do 'module load cuda' and 'module load gcc'\n"
	gcc -O3 -lm -lOpenCL mcmcGPU.c common/epsilons.c -o mcmcGPU

mcmcCPU: mcmcCPU.c common/epsilons.c common/epsilons.h
	gcc -O3 -lm mcmcCPU.c -o mcmcCPU common/epsilons.c -o mcmcCPU
