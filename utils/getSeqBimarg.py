#!/usr/bin/env python2
from scipy import *
import numpy as np
import seqload
import sys
import argparse
from Bio.Alphabet import IUPAC

def getMarginals(seqs, nBases): 
    nSeq, seqLen = seqs.shape
    nrmlz = lambda x: x/sum(x,axis=1)[:,newaxis]
    freqs = lambda s,bins: histogram(s, bins)[0].astype(float64)

    bins = arange(nBases+1, dtype='int')
    f = nrmlz(array([freqs(seqs[:,i], bins) for i in range(seqLen)]))

    bins = arange(nBases*nBases+1, dtype='int')
    ff = nrmlz(array([ freqs(seqs[:,j] + nBases*seqs[:,i], bins)
                                             for i in range(seqLen) 
                                             for j in range(i+1, seqLen)]))
    return f, ff

class Counter:
    def __init__(self, weights):
        self.weights = weights
        self.pos = 0

    def __call__(self, counts, seqs, (param, headers)): 
        nSeq, L = seqs.shape
        nB = len(param['alpha'])
        nbins = nB*nB

        
        if self.weights is not None:
            if counts is 0:
                counts = zeros((L*(L-1)/2, nB*nB), dtype='f8')
            w = self.weights[self.pos:self.pos+nSeq]
            self.pos += nSeq
        else:
            if counts is 0:
                counts = zeros((L*(L-1)/2, nB*nB), dtype='i4')
            w = None

        n = 0
        for i in range(L):
            nsi = nB*seqs[:,i].astype('i8')
            for j in range(i+1, L):
                sj = seqs[:,j].astype('i8')
                counts[n,:] += bincount(sj + nsi, minlength=nbins, weights=w)
                n += 1
        return counts

def main():
    parser = argparse.ArgumentParser(description='Compute Bivariate Marginals')
    parser.add_argument('-alpha', default='protgap')
    parser.add_argument('-weights')
    parser.add_argument('-counts', action='store_true')
    parser.add_argument('seqfile')
    parser.add_argument('outfile')

    args = parser.parse_args(sys.argv[1:])
    
    alphabets = {'protein': IUPAC.protein.letters, 
                 'protgap': '-' + IUPAC.protein.letters, 
                 'charge': '0+-', 
                 'nuc': "ACGT"}

    letters = alphabets.get(args.alpha, args.alpha)

    if args.weights:
        try:
            weights = load(args.weights)
        except:
            weights = loadtxt(args.weights)
    else:
        weights = None

    counter = Counter(weights)

    counts = seqload.reduceSeqs(args.seqfile, counter, 0, letters)[0]
    if args.counts:
        save(args.outfile, counts)
    else:
        ff = counts.astype('f4')/np.float32(sum(counts[0,:]))
        save(args.outfile, ff)

if __name__ == '__main__':
    main()
