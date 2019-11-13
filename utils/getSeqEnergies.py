#!/usr/bin/env python
#
#Copyright 2019 Allan Haldane.

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
    if q > 16: # the x + q*y operation below may overflow for i1
        s = s.astype('i4')
    pairenergy = zeros(s.shape[0], dtype='f8')
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        pairenergy += J[n,s[:,j] + q*s[:,i]]
    return pairenergy

def main():
    parser = argparse.ArgumentParser(description='Compute Sequence Energies')
    parser.add_argument('seqs')
    parser.add_argument('couplings')
    parser.add_argument('-o', '--out')
    parser.add_argument('--alpha', default='protgap')

    args = parser.parse_args(sys.argv[1:])
    
    alphabets = {'protein': IUPAC.protein.letters, 
                 'protgap': '-' + IUPAC.protein.letters, 
                 'charge': '0+-', 
                 'nuc': "ACGT"}
    try:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:int(args.alpha)]
    except ValueError:
        letters = alphabets.get(args.alpha, args.alpha)

    couplings = scipy.load(args.couplings).astype('f8')

    def chunkE(seqs, param):
        return energies(seqs, couplings)
    
    # process the file in chunks for speed
    e = seqload.mapSeqs(args.seqs, chunkE, letters)[0]
    if args.out:
        save(args.out, e)
    else:
        savetxt(sys.stdout, e)

if __name__ == '__main__':
    main()
    
