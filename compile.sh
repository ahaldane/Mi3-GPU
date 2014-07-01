#!/bin/bash

module load cuda
module load gcc
gcc -O3 -lm -lOpenCL mcmcGPU.c common/epsilons.c -o mcmcGPU
