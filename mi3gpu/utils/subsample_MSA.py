#!/usr/bin/env python
import numpy as np
import sys
from mi3gpu.utils.seqload import loadSeqs, writeSeqs
from mi3gpu.utils.seqtools import filtersim

rng = np.random.default_rng()

s = loadSeqs(sys.argv[1])[0]
s = s[rng.permutation(s.shape[0])]
fs = filtersim(s, int(float(sys.argv[2])*s.shape[1]))
fs = fs[rng.permutation(fs.shape[0])]
writeSeqs(sys.stdout, fs)
