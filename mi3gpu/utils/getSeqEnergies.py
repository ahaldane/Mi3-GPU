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
import functools

from mi3gpu.utils.potts_common import getLq, alpha20
import mi3gpu.utils.seqload as seqload

@functools.lru_cache
def nij_inds(L):
    i,j = np.triu_indices(L,k=1)
    n = np.arange(L*(L-1)//2)
    return n,i,j

# reorder the memory for optimized energy calculation.
# do this once beforehand if calling E_potts many times.
def MSA_prepmem(s, q):
    return np.require(s, dtype='i4' if q > 16 else None, requirements='F')

def E_potts(s, J):
    L, q = getLq(J)
    assert(L == s.shape[-1])
    # the x + q*y operation below may overflow for i1 if q>16, so fix if so.
    # Also, make the ij index the fast axis, and transpose to help bcasting
    s = MSA_prepmem(s, q)

    if s.ndim == 1: # a single sequence
        n,i,j = nij_inds(L)
        return np.sum(J[n,q*s[i] + s[j]], dtype='f8')
    
    N = s.shape[0]
    qsi = np.empty(N, dtype=s.dtype) # preallocated scratch
    qqs = np.empty(N, dtype=s.dtype) # preallocated scratch
    pairenergy = np.zeros(N, dtype='f8')
    n = 0
    for i in range(L-1):
        np.multiply(q, s[:,i], out=qsi)
        for j in range(i+1,L):
            np.add(qsi, s[:,j], out=qqs)
            pairenergy += J[n,qqs]
            n += 1
    return pairenergy

def E_indep(s, h):
    L, q = h.shape
    return np.sum(h[np.arange(L), s], axis=-1)

def E_potts_decomposed(s, J):
    L, q = getLq(J)
    assert(L == s.shape[-1])
    # the x + q*y operation below may overflow for i1 if q>16, so fix if so.
    # Also, make the ij index the fast axis
    s = MSA_prepmem(s, q)
    n,i,j = nij_inds(L)
    # note this returns the f4 dtype of J. To perform sums, convert to f8 first
    return J[n[:,None],q*s[:,i] + s[:,j]]

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
        return E_potts(seqs, couplings)
    
    # process the file in chunks for speed
    e = seqload.mapSeqs(args.seqs, chunkE, letters)[0]
    if args.out:
        np.save(args.out, e)
    else:
        np.savetxt(sys.stdout, e)

if __name__ == '__main__':
    main()
    
