#!/usr/bin/env python3
#
#Copyright 2020 Allan Haldane.

#This file is part of Mi3-GPU.

#Mi3-GPU is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#Mi3-GPU is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with Mi3-GPU.  If not, see <http://www.gnu.org/licenses/>.

#Contact: allan.haldane _AT_ gmail.com
import numpy as np
import sys
import argparse

from mi3gpu.utils.potts_common import alpha20
import mi3gpu.utils.seqload as seqload

def getMarginals(seqs, q, weights=None, nrmlz=True):
    nSeq, L = seqs.shape

    if q > 16: # the x + q*y operation below may overflow for u1
        seqs = seqs.astype('i4')

    if nrmlz:
        nrmlz = lambda x: x/np.sum(x, axis=-1, keepdims=True)
    else:
        nrmlz = lambda x: x

    def counts(s, bins):
        return np.bincount(s, minlength=bins, weights=weights)

    f = nrmlz(np.array([counts(seqs[:,i], q) for i in range(L)]))
    ff = nrmlz(np.array([counts(seqs[:,j] + q*seqs[:,i], q*q)
                         for i in range(L-1) for j in range(i+1, L)]))
    return f, ff

class BiCounter:
    def __init__(self, weights):
        self.weights = weights
        self.pos = 0

    def __call__(self, counts, seqs, ids, headers, alpha):
        nSeq, L = seqs.shape
        q = len(alpha)
        nbins = q*q

        if q > 16: # the x + q*y operation below may overflow for u1
            seqs = seqs.astype('i4')

        if self.weights is not None:
            if not isinstance(counts, np.ndarray) and counts == 0:
                counts = np.zeros((L*(L-1)//2, q*q), dtype='f8')
            w = self.weights[self.pos:self.pos+nSeq]
            self.pos += nSeq
        else:
            if not isinstance(counts, np.ndarray) and counts == 0:
                counts = np.zeros((L*(L-1)//2, q*q), dtype='i4')
            w = None

        n = 0
        for i in range(L):
            nsi = q*seqs[:,i]
            for j in range(i+1, L):
                sj = seqs[:,j]
                counts[n,:] += np.bincount(sj + nsi, minlength=nbins, weights=w)
                n += 1
        return counts

class UniCounter:
    def __init__(self, weights):
        self.weights = weights
        self.pos = 0

    def __call__(self, counts, seqs, ids, headers, alpha):
        nSeq, L = seqs.shape
        q = len(alpha)
        nbins = q

        if self.weights is not None:
            if not isinstance(counts, np.ndarray) and counts == 0:
                counts = np.zeros((L,q), dtype='f8')
            w = self.weights[self.pos:self.pos+nSeq]
            self.pos += nSeq
        else:
            if not isinstance(counts, np.ndarray) and counts == 0:
                counts = np.zeros((L,q), dtype='i4')
            w = None

        n = 0
        for i in range(L):
            si = seqs[:,i]
            counts[n,:] += np.bincount(si, minlength=nbins, weights=w)
            n += 1
        return counts

def main():
    parser = argparse.ArgumentParser(description='Compute Bivariate Marginals')
    parser.add_argument('--alpha', default='protgap')
    parser.add_argument('--weights')
    parser.add_argument('--counts', action='store_true')
    parser.add_argument('--uni', action='store_true')
    parser.add_argument('--dtype', default='f4')
    parser.add_argument('seqfile')
    parser.add_argument('outfile')

    args = parser.parse_args(sys.argv[1:])

    alphabets = {'protein': alpha20,
                 'protgap': '-' + alpha20,
                 'charge': '0+-',
                 'nuc': "ACGT"}

    letters = alphabets.get(args.alpha, args.alpha)

    if args.weights:
        try:
            weights = np.load(args.weights)
        except:
            weights = np.loadtxt(args.weights)
    else:
        weights = None

    if args.uni:
        counter = UniCounter(weights)
    else:
        counter = BiCounter(weights)

    counts = seqload.reduceSeqs(args.seqfile, counter, 0, letters)[0]
    if args.counts:
        np.save(args.outfile, counts)
    else:
        counts = counts.astype(args.dtype)
        ff = counts/np.sum(counts[0,:])
        np.save(args.outfile, ff)

if __name__ == '__main__':
    main()
