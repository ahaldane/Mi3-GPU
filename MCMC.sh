#!/usr/bin/env bash
#PBS -q batch
#PBS -l nodes=cb2hpc2:gpus=4
#PBS -N MCMCseq
#PBS -e stderr
#PBS -o stdout

cd $PBS_O_WORKDIR

stdbuf -i0 -o0 -e0 ./IvoGPU.py inverseIsing --bimarg f2.npy --gamma 0.0004 --mcsteps 100 --nwalkers 65536 --equiltime 2048 --nsamples 64 --sampletime 64 --alpha ABCDEFGH --seqmodel reducedA2/run_14/ --damping 0.001 --nsteps 2048 --newtonsteps 128 --trackequil 16 --outdir reducedA3 >logReducedA3
