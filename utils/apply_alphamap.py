#!/usr/bin/env python
#
#Copyright 2018 Allan Haldane.

#This file is part of IvoGPU.

#IvoGPU is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#IvoGPU is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with IvoGPU.  If not, see <http://www.gnu.org/licenses/>.

#Contact: allan.haldane _AT_ gmail.com
import sys, os, argparse
from Bio.Alphabet import IUPAC
import seqload
import numpy as np

def ind(x, alpha):
    for i in range(len(alpha)):
        if x in alpha[i]:
            return i
    return 0

def reduceSeqAlphaPerpos(seqs, newalphas, oldalpha):
    def ind(x, alpha):
        for i in range(len(alpha)):
            if x in alpha[i]:
                return i
        return 0
    
    rseqs = np.empty(seqs.shape, dtype=int)
    for n,a in enumerate(newalphas):
        conv = np.array([ind(let,a) for let in oldalpha])
        rseqs[:,n] = conv[seqs[:,n]]
            
    seqload.writeSeqs(sys.stdout, rseqs, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 
                      noheader=True)

def reduceBimAlphaPerpos(bimarg, newalphas, oldalpha):

def main():
    parser = argparse.ArgumentParser(description='Apply alphabet reduction to MSA')
    parser.add_argument('seqs_bimarg', help='seq file or bimarg file')
    parser.add_argument('alphamap')
    parser.add_argument('-alpha', default='protgap')

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

    
    bimarg, seqs = None, None
    try:
        bimarg = np.load(args.seq_bimarg)
        reduceBimAlphaPerpos(seqs, newalphas, alpha)
    except ValueError:
        seqs = seqload.loadSeqs(args.seqs, alpha)[0]
        reduceSeqAlphaPerpos(seqs, newalphas, alpha)

if __name__ == '__main__':
    main()

