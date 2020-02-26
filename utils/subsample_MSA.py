#!/usr/bin/env python3
import numpy as np
import seqload, seqtools
from numpy.random import randint
import sys, os

s = seqload.loadSeqs(sys.argv[1])[0]
cutoff = 1-float(sys.argv[2])
L = s.shape[1]

inds = []
out_seq = []
while s.shape[0] != 0:
    ind = randint(s.shape[0])
    out_seq.append(s[ind].copy()) # no ref to s
    s = s[np.sum(s == s[ind,:], axis=1)/float(L) < cutoff,:]
    print(s.shape, file=sys.stderr)

with os.fdopen(sys.stdout.fileno(), 'wb', closefd=False) as fp:
    seqload.writeSeqs(fp, np.array(out_seq), noheader=True)
