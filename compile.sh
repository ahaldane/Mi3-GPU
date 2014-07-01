#!/usr/bin/bach

module load cuda
module load gcc
gcc -O3 -lm -lOpenCL mcmcGPU.c epsilons.c -o mcmcGPU
