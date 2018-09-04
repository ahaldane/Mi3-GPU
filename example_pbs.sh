#!/bin/bash
#PBS -q gpu
#PBS -N potts_inference
#PBS -l walltime=6:00:00

cd $PBS_O_WORKDIR

execpath=~/IvoGPU/IvoGPU.py

margfile="example_bimarg_pc.npy"
logfile="hiv_pr_inference.log"
outdir="hiv_pr_inference"

stdbuf -i0 -o0 -e0 $execpath inverseIsing \
  --seqmodel    independent \
  --bimarg      $margfile \
  --alpha       ABCD \
  --gamma       0.0004 \
  --mcsteps     64 \
  --nwalkers    65536 \
  --reseed      single_indep \
  --damping     0.01 \
  --nsteps      2048 \
  --newtonsteps 256 \
  --outdir $outdir >$logfile

# The --seqmodel option may be set to a directory such as $outdir/run_24
# to continue a previous run.
