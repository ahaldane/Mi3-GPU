#!/usr/bin/env python3
import numpy as np
from numpy.random import randint
import sys, os, argparse
from mi3gpu.utils.seqload import loadSeqs, writeSeqs
from mi3gpu.utils.seqtools import filtersim

def subsample_MSA(s, ids, cutoff):
    N, L = s.shape
    rng = np.random.default_rng()

    # shuffle input sequences (since filtersim is order-dependent)
    inord = rng.permutation(N)
    s = s[inord]

    s, inds = filtersim(s, int((1-cutoff)*s.shape[1]), return_inds=True)
    inds = inord[inds]
    if ids is not None:
        ids = ids[inds]

    # shuffle output sequences (since filtersim order is not random)
    outord = rng.permutation(s.shape[0])
    s = s[outord]
    inds = inds[outord]
    if ids is not None:
        ids = ids[outord]
    return s, ids

def main():
    parser = argparse.ArgumentParser(
        description='remove sequences too similar to another sequence')
    parser.add_argument('seqs')
    parser.add_argument('cutoff', type=float)
    parser.add_argument('--inds', help='save indices to this file')
    args = parser.parse_args()

    s, ids, _ = loadSeqs(args.seqs)
    s, ids = subsample_MSA(s, ids, cutoff)

    # write out result
    writeSeqs(sys.stdout, s, ids=ids)
    if args.inds:
        np.save(args.inds, inds)

if __name__ == '__main__':
    main()
