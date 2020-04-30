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
np.set_printoptions(linewidth=200, suppress=True)

import mi3gpu.utils.seqload as seqload
import mi3gpu.utils.seqtools as seqtools
from mi3gpu.utils.potts_common import alpha20

def main():
    parser = argparse.ArgumentParser(description='Compute phylogenetic weights')
    parser.add_argument('--alpha', default='protgap')
    parser.add_argument('sim', default='none', help="Similarity Threshold")
    parser.add_argument('seqfile')
    parser.add_argument('outfile')

    args = parser.parse_args(sys.argv[1:])
    
    alphabets = {'protein': alpha20,
                 'protgap': '-' + alpha20,
                 'charge': '0+-', 
                 'nuc': "ACGT"}
    letters = alphabets.get(args.alpha, args.alpha)
    nBases = len(letters)

    seqs = seqload.loadSeqs(args.seqfile, letters)[0]
    nSeq, seqLen = seqs.shape

    if args.sim == 'none':
        sim = 0
    elif args.sim == 'unique':
        sim = 0.5/seqLen
    elif args.sim.startswith('m'):
        sim = (float(args.sim[1:])+0.5)/seqLen
    else:
        sim = float(args.sim)

    sim = 1-sim
    if sim < 0 or sim > 1:
        raise Exception("Similarity threshold must be between 0 and 1")

    if sim != 1.0:
        similarityCutoff = int(np.ceil((1-sim)*seqLen))
        print("Identity cutoff:", similarityCutoff, file=sys.stderr)
        weights = 1.0/seqtools.nsim(seqs, similarityCutoff)
    else:
        weights = np.ones(seqs.shape[0])
    M_eff = np.sum(weights)
    print(M_eff)

    np.save(args.outfile, weights)

if __name__ == '__main__':
    main()
