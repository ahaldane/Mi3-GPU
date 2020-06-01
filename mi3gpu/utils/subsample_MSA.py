#!/usr/bin/env python3
import numpy as np
from numpy.random import randint
import sys, os, argparse

import mi3gpu.utils.seqload as seqload
import mi3gpu.utils.seqtools as seqtools

parser = argparse.ArgumentParser(
    description='remove sequences too similar to another sequence')
parser.add_argument('seqs')
parser.add_argument('cutoff', type=float)
parser.add_argument('--ind', help='save indices')
args = parser.parse_args()

s, ids, _ = seqload.loadSeqs(args.seqs)
cutoff = 1-float(args.cutoff)
N, L = s.shape

inds = None
remaining_inds = None
if args.ind is not None:
    inds = []
    remaining_inds = np.arange(N)

out_seq = []
out_ids = []
while s.shape[0] != 0:
    ind = randint(s.shape[0])
    keep = np.sum(s == s[ind,:], axis=1)/float(L) < cutoff

    out_seq.append(s[ind].copy()) # no ref to s
    s = s[keep,:]

    if ids is not None:
        out_ids.append(ids[ind])
        ids = ids[keep]

    if inds is not None:
        inds.append(remaining_inds[ind])
        remaining_inds = remaining_inds[keep]
    print(s.shape, file=sys.stderr)

with os.fdopen(sys.stdout.fileno(), 'wb', closefd=False) as fp:
    seqload.writeSeqs(fp, np.array(out_seq), ids=out_ids or None)

if inds is not None:
    np.save(args.ind, inds)
