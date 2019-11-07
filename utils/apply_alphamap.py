#!/usr/bin/env python
#
#Copyright 2018 Allan Haldane.

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
import sys, os, argparse
from Bio.Alphabet import IUPAC
import seqload
import numpy as np
from alphabet_reduction import getLq

def indmap(oldalpha, amap):
    def ind(x):
        for i in range(len(amap)):
            if x in amap[i]:
                return i
        return 0

    return  np.array([ind(let) for let in oldalpha])

def reduceSeqAlphaPerpos(seqs, newalphas, oldalpha, out=None):
    rseqs = np.empty(seqs.shape, dtype=int)
    for n,a in enumerate(newalphas):
        conv = indmap(oldalpha, a)
        rseqs[:,n] = conv[seqs[:,n]]

    if out is None:
        out = sys.stdout
    seqload.writeSeqs(out, rseqs, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", noheader=True)

def reduceBimAlphaPerpos(bimarg, newalphas, oldalpha, out):
    if out is None:
        raise ValueError('out argument required for bimarg reduction')

    L, q = getLq(bimarg)
    qout = len(newalphas[0])
    nPairs = L*(L-1)//2

    # strategy: compute index mapping array to use in add.at
    maps = np.array([indmap(oldalpha, a) for a in newalphas])
    qqmap = np.array([np.add.outer(qout*maps[i], maps[j]).ravel()
                      for i in range(L-1) for j in range(i+1,L)])
    nmap = np.broadcast_to(np.arange(nPairs, dtype=int)[:,None], (nPairs, q*q))

    # allocate new bimarg
    newbim = np.zeros((nPairs, qout*qout), dtype='f8')

    # do the accumulation with `at`
    np.add.at(newbim, (nmap, qqmap), bimarg)

    # renormalize
    newbim /= np.sum(newbim, axis=1, keepdims=True)

    np.save(out, newbim.astype('f4'))

def main():
    parser = argparse.ArgumentParser(
                                description='Apply alphabet reduction to MSA')
    parser.add_argument('file', help='either seq file or bimarg file')
    parser.add_argument('alphamap')
    parser.add_argument('--alpha', default='protgap')
    parser.add_argument('--out')

    args = parser.parse_args(sys.argv[1:])
    alphabets = {'protein': IUPAC.protein.letters,
                 'protgap': '-' + IUPAC.protein.letters,
                 'charge': '0+-',
                 'nuc': "ACGT"}
    alpha = alphabets.get(args.alpha, args.alpha)

    with open(args.alphamap) as f:
        # assumed to be a file containing the output of alphabet reduction, but
        # only for one reduction level.  Each line should look like:
        # ALPHA8 -DNAGSQFMYCI E HWP K L R T V
        newalphas = [a.split()[1:] for a in f.readlines()]

    try:
        bimarg = np.load(args.file)
    except ValueError:
        seqs = seqload.loadSeqs(args.file, alpha)[0]
        reduceSeqAlphaPerpos(seqs, newalphas, alpha, args.out)
    else:
        reduceBimAlphaPerpos(bimarg, newalphas, alpha, args.out)

if __name__ == '__main__':
    main()

