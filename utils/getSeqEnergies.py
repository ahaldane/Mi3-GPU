#!/usr/bin/env python2
import scipy
from scipy import *
import seqload
import sys, argparse
from Bio.Alphabet import IUPAC

def getLq(J):
    L = int(((1+sqrt(1+8*J.shape[0]))/2) + 0.5)
    q = int(sqrt(J.shape[1]) + 0.5)
    return L, q

def energies(s, J):
    L, q = getLq(J)
    pairenergy = zeros(s.shape[0])
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        pairenergy += couplings[n,s[:,j] + q*s[:,i]]
    return pairenergy

def main():
    parser = argparse.ArgumentParser(description='Compute Sequence Energies')
    parser.add_argument('seqs')
    parser.add_argument('couplings')
    parser.add_argument('-o', '--out')
    parser.add_argument('-alpha', default='protgap')

    args = parser.parse_args(sys.argv[1:])
    
    alphabets = {'protein': IUPAC.protein.letters, 
                 'protgap': '-' + IUPAC.protein.letters, 
                 'charge': '0+-', 
                 'nuc': "ACGT"}
    try:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:int(args.alpha)]
    except ValueError:
        letters = alphabets.get(args.alpha, args.alpha)

    couplings = scipy.load(args.couplings)

    def chunkE(seqs, param):
        return energies(seqs, couplings)
    
    # process the file in chunks for speed
    e = seqload.mapSeqs(args.seqs, letters, chunkE)[0]
    if args.out:
        save(args.out, e)
    else:
        savetxt(sys.stdout, e)

if __name__ == '__main__':
    main()
    
