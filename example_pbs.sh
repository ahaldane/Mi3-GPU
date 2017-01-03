#!/bin/bash
#PBS -q gpu
#PBS -N potts_inference
#PBS -l walltime=6:00:00
#PBS -l nodes=1:ppn=4
#PBS -o stdout
#PBS -e stderr

cd $PBS_O_WORKDIR

execpath=./IvoGPU.py

margfile="example_bimarg_pc.npy"
logfile="hiv_pr_inference.log"
outdir="hiv_pr_inference"

stdbuf -i0 -o0 -e0 $execpath inverseIsing \
  --seqmodel    logscore \
  --bimarg      $margfile \
  --alpha       ABCD \
  --gamma       0.0004 \
  --mcsteps     100 \
  --nwalkers    8192 \
  --equiltime   1024 \
  --nsamples    64 \
  --sampletime  64 \
  --damping     0.001 \
  --nsteps      2048 \
  --newtonsteps 64 \
  --trackequil  16 \
  --outdir $outdir >$logfile

# The --seqmodel option may be set to a directory such as $outdir/run_24
# to continue a previous run.
