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
import sys, argparse

from mi3gpu.utils.potts_common import getLq, alpha20
import mi3gpu.utils.seqload as seqload

def potts_energies(s, J):
    L, q = getLq(J)
    N, Ls = s.shape
    assert(L == Ls)

    s = s.T.copy() # transpose for speed
    if q > 16: # the x + q*y operation below may overflow for i1
        s = s.astype('i4')

    pairenergy = np.zeros(N, dtype='f8')
    for n,(i,j) in enumerate((i,j) for i in range(L-1) for j in range(i+1,L)):
        pairenergy += J[n,s[j,:] + q*s[i,:]]
    return pairenergy

def indep_energies(s, h):
    L, q = h.shape
    return np.sum(h[np.arange(L), s], axis=1)

def potts_energies_decomposed(s, J):
    L, q = getLq(J)
    N, Ls = s.shape
    assert(L == Ls)

    s = s.T.copy() # transpose for speed
    if q > 16: # the x + q*y operation below may overflow for i1
        s = s.astype('i4')

    cpl = np.zeros((N, L*(L-1)//2), dtype='f8')
    for n,(i,j) in enumerate((i,j) for i in range(L-1) for j in range(i+1,L)):
        cpl[:,n] = J[n,s[j,:] + q*s[i,:]]
    return cpl

def main():
    parser = argparse.ArgumentParser(description='Compute Sequence Energies')
    parser.add_argument('seqs')
    parser.add_argument('couplings')
    parser.add_argument('-o', '--out')
    parser.add_argument('--alpha', default='protgap')

    args = parser.parse_args(sys.argv[1:])
    
    alphabets = {'protein': alpha20,
                 'protgap': '-' + alpha20,
                 'charge': '0+-', 
                 'nuc': "ACGT"}
    try:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:int(args.alpha)]
    except ValueError:
        letters = alphabets.get(args.alpha, args.alpha)

    couplings = np.load(args.couplings).astype('f8')

    def chunkE(seqs, param):
        return potts_energies(seqs, couplings)
    
    # process the file in chunks for speed
    e = seqload.mapSeqs(args.seqs, chunkE, letters)[0]
    if args.out:
        np.save(args.out, e)
    else:
        np.savetxt(sys.stdout, e)

if __name__ == '__main__':
    main()
    
