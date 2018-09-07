#!/usr/bin/env python2
#
#Copyright 2016 Allan Haldane.

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
from scipy import *
import numpy as np
import sys, argparse
from scipy.special import rel_entr

np.seterr(all='ignore')

def indepF(fab):
    L = int( (1+sqrt(1+8*fab.shape[0]))/2 + 0.5)
    nB = int(sqrt(fab.shape[1]) + 0.5)

    fabx = fab.reshape((fab.shape[0], nB, nB))
    fa1, fb2 = sum(fabx,axis=2), sum(fabx,axis=1)
    fafb = array([outer(fa, fb).flatten() for fa,fb in zip(fa1, fb2)])
    return fafb

def KL(f1, f2):
    return sum(rel_entr(f1, f2), axis=-1)

def ExpEnt(f, N):
    Np = (N-1)/(2*N*N)
    fp = f + (1-f)/N
    terms = log(fp) - Np*f*(1-f)/fp**2
    return sum(f*terms, axis=-1)

def regularize_bayesian_KL(fab, N):
    fafb = indepF(fab)
    Np = (N-1)/(2*N*N)

    lambdas = zeros(fab.shape[0])
    for l in linspace(0,1,100):
        f = (1-l)*fab + l*fafb
        fp = f + (1-f)/N
        expected_KL = sum(f*(log(fp/f) - Np*f*(1-f)/fp**2), axis=1)
        observed_KL = KL(fab, f)

        lambdas[observed_KL < expected_KL] = l
    
    lambdas = lambdas[:,newaxis]
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
        fab = load(args.bimarg)
    except:
        fab = loadtxt(args.bimarg)
    fab = fab.astype('f8') + args.pseudocount
    fab = fab/sum(fab, axis=1, keepdims=True)

    fab_biased, lam = regularize_bayesian_KL(fab, args.N)

    save(args.out, fab_biased.astype(args.outformat))

    if args.lambda_out:
        save(args.lambda_out, lam)

if __name__ == '__main__':
    main()

