#!/bin/bash
#PBS -q gpu
#PBS -N potts_hiv_inference
#PBS -l walltime=2:00:00

cd $PBS_O_WORKDIR

execpath=../../IvoGPU.py

margfile="example_bimarg_pc.npy"
logfile="hiv_pr_inference.log"
outdir="hiv_pr_inference"

python -u $execpath inverseIsing \
  --bimarg      $margfile \
  --alpha       ABCD \
  --nwalkers    262144 \
  --damping     0.01 \
  --outdir $outdir >$logfile

# The --seqmodel option may be set to a directory such as $outdir/run_24
# to continue a previous run.
