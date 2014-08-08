#!/usr/bin/env python2
from scipy import *
import load
import sys

nseqs = int(sys.argv[2])
alpha = sys.argv[3] if len(sys.argv) == 4 else None
seqs, info = load.loadSites(sys.argv[1], names=alpha)
if alpha == None:
    alpha = info[2]['alpha']

nreps = nseqs/seqs.shape[0]
nrem = nseqs%seqs.shape[0]

load.writeSites(sys.stdout, concatenate([seqs for i in range(nreps)] + [seqs[:nrem]]), alpha, param={'alpha': alpha})
