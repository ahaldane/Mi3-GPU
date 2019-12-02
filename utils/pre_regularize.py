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
import numpy as np
import sys, argparse
from scipy.special import rel_entr
from potts_common import indepF

np.seterr(all='ignore')

def KL(f1, f2):
    return np.sum(rel_entr(f1, f2), axis=-1)

def regularize_bayesian_KL(fab, N):
    fafb = indepF(fab)
    Np = (N-1)/(2*N*N)

    lambdas = np.zeros(fab.shape[0])
    for l in np.linspace(0,1,100):
        f = (1-l)*fab + l*fafb
        fp = f + (1-f)/N
        expected_KL = np.sum(f*(np.log(fp/f) - Np*f*(1-f)/fp**2), axis=1)
        observed_KL = KL(fab, f)

        lambdas[observed_KL < expected_KL] = l
    
    lambdas = lambdas[:,None]
    return (1-lambdas)*fab + lambdas*fafb, lambdas

def main():
    parser = argparse.ArgumentParser(description='Regularizer Pre-processing')
    parser.add_argument('bimarg')
    parser.add_argument('N', type=float, help="MSA depth")
    parser.add_argument('out')
    parser.add_argument('--pseudocount', type=float, default=0,
                        help="pseudocount to avoid div by 0")
    parser.add_argument('--outformat', default='f4')
    parser.add_argument('--lambda_out')

    args = parser.parse_args(sys.argv[1:])

    try:
        fab = np.load(args.bimarg)
    except:
        fab = np.loadtxt(args.bimarg)
    fab = fab.astype('f8') + args.pseudocount
    fab = fab/np.sum(fab, axis=1, keepdims=True)

    fab_biased, lam = regularize_bayesian_KL(fab, args.N)

    np.save(args.out, fab_biased.astype(args.outformat))

    if args.lambda_out:
        np.save(args.lambda_out, lam)

if __name__ == '__main__':
    main()

