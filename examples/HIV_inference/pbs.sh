#!/bin/bash

margfile="example_bimarg_pc.npy"
logfile="hiv_pr_inference.log"
outdir="hiv_pr_inference"

Mi3.py infer \
  --init_model  independent \
  --bimarg      $margfile \
  --alpha       ABCD \
  --nwalkers    262144 \
  --damping     0.01 \
  --mcsteps     64 \
  --reg         l1z:0.0001 \
  --outdir $outdir >$logfile

# It can be good to run a second round of inference with a very
# large number of walkers, to obtain a very statistically accurate model.
# The commented lines below illustrate how to continue from the last run
# of the command above.

#Mi3.py infer \
#  --init_model  $outdir/run_63 \
#  --bimarg      $margfile \
#  --alpha       ABCD \
#  --nwalkers    1048576 \
#  --damping     0.01 \
#  --mcsteps     64 \
#  --reg         l1z:0.0001 \
#  --outdir ${outdir}_2 >${logfile}_2
