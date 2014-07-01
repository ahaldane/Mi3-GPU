
mcmcGPU: mcmcGPU.c common/epsilons.c common/epsilons.h clErrors.h makeSeqfile
	@echo -e "Remember to do 'module load cuda' and 'module load gcc'\n"
	gcc -O3 -lm -lOpenCL mcmcGPU.c common/epsilons.c -o mcmcGPU

makeSeqfile: makeSeqfile.c
	gcc -O3 makeSeqfile.c -o makeSeqfile

mcmcCPU: mcmcCPU.c common/epsilons.c common/epsilons.h
	gcc -O3 -lm mcmcCPU.c -o mcmcCPU common/epsilons.c -o mcmcCPU
