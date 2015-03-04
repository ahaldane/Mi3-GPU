#!/usr/bin/env bash
#PBS -q batch
#PBS -l nodes=cb2hpc1:gpus=4
#PBS -N MCMCseq
#PBS -e stderr
#PBS -o stdout

cd $PBS_O_WORKDIR

stdbuf -i0 -o0 -e0 ./mcmcGPU.py bimarg.npy 0.0004 100 32768 512 16 32 ABCDEFGH -restart logscore -pc 1e-5 -pcdamping 0.1 -regularizationScale 0.1 -nsteps 16 -perturbSteps 64 -trackequil 16 -o outdir >log
