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
import sys, argparse

def getCouplingMatrix(couplings):
    #compute the blocks that make up Ciajb, that is, compute the block Cij
    L = int( (1+sqrt(1+8*couplings.shape[0]))/2 + 0.5)
    nB = int(sqrt(couplings.shape[1]) + 0.5)
    coupleinds = [(a,b) for a in range(L-1) for b in range(a+1, L)]

    C = empty((L,nB,L,nB))
    C.fill(nan)
    for n,(i,j) in enumerate(coupleinds): 
        block = couplings[n].reshape(nB,nB)
        C[i,:,j,:] = block
        C[j,:,i,:] = block.T
    return C

def weightedGauge(hs, Js, weights=None):
    if weights == None:
        raise Exception("weights must be supplied to get weighted gauge")

    L, nB = hs.shape
    weightsx = weights.reshape((L*(L-1)/2, nB, nB))
    Jx = Js.reshape((L*(L-1)/2, nB, nB))
    #weightsxC = nan_to_num(getCouplingMatrix(weights))
    #JxC = nan_to_num(getCouplingMatrix(Js))

    J0 = (Jx - average(Jx, weights=weightsx, axis=1)[:,newaxis,:] 
             - average(Jx, weights=weightsx, axis=2)[:,:,newaxis] 
             + average(Jx, weights=weightsx, axis=(1,2))[:,newaxis,newaxis])
    #computation of hs has not been checked.. probably wrong
    #h0 = hs + sum(average(JxC, weights=weightsxC, axis=1), axis=0)
    h0 = hs
    J0 = J0.reshape((J0.shape[0], nB**2))
    return h0, J0

def zeroGauge(hs, Js, weights=None):
    #sets the mean h-rows and J rows/cols to 0
    #this is a fully constrained gauge
    if any(isinf(Js)) or any(isinf(hs)):
        raise Exception("Error: Cannot convert to zero gauge because "
                        "of infinities")

    L, nB = hs.shape
    Jx = Js.reshape((L*(L-1)/2, nB, nB))
    JxC = nan_to_num(getCouplingMatrix(Js))

    J0 = (Jx - mean(Jx, axis=1)[:,newaxis,:] 
             - mean(Jx, axis=2)[:,:,newaxis] 
             + mean(Jx, axis=(1,2))[:,newaxis,newaxis])
    h0 = hs + sum(mean(JxC, axis=1), axis=0)
    h0 = h0 - mean(h0, axis=1)[:,newaxis]
    J0 = J0.reshape((J0.shape[0], nB**2))
    return h0, J0

def zeroJGauge(hs, Js, weights=None): 
    #only set mean J to 0, but choose fields so sequence energies do not change
    # (this keeps the overall sequence energy the same)
    if any(isinf(Js)) or any(isinf(hs)):
        raise Exception("Error: Cannot convert to zero gauge because "
                        "of infinities")

    L, nB = hs.shape
    Jx = Js.reshape((L*(L-1)/2, nB, nB))

    J0 = (Jx - mean(Jx, axis=1)[:,newaxis,:] 
             - mean(Jx, axis=2)[:,:,newaxis] 
             + mean(Jx, axis=(1,2))[:,newaxis,newaxis])

    JxC = nan_to_num(getCouplingMatrix(Js))
    h0 = hs+(sum(mean(JxC, axis=1), axis=0) - 
          (sum(mean(JxC, axis=(1,3)), axis=0)/2)[:,newaxis])
    J0 = J0.reshape((J0.shape[0], nB**2))
    return h0, J0


def fieldlessGaugeDistributed(hs, Js, weights=None): #convert to a fieldless gauge
    #This function tries to distribute the fields evenly
    #but does not first re-zero the fields/couplings
    L, nB = hs.shape
    J0 = Js.copy()
    hd = hs/(L-1)
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        J0[n,:] += repeat(hd[i,:], nB)
        J0[n,:] += tile(hd[j,:], nB)
    return zeros(hs.shape), J0

def fieldlessGaugeEven(hs, Js, weights=None): #convert to a fieldless gauge
    #construct a fieldless gauge, but try to distribute the energy evenly
    # among the fields by first converting to the zero gauge
    return fieldlessGaugeDistributed(*zeroGauge(hs, Js))

def fieldlessGauge(hs, Js, weights=None):
    #note: Fieldless gauge is not fully constrained: There are many possible 
    #choices that are fieldless, this just returns one of them
    seqLen, nBases = hs.shape
    J0 = Js.copy()
    J0[0,:] += repeat(hs[0,:], nBases)
    for i in range(seqLen-1):
        J0[i,:] += tile(hs[i+1,:], nBases)
    return zeros(hs.shape), J0

def tryload(fn):
    try:
        return load(fn)
    except:
        return loadtxt(fn)

def main():
    parser = argparse.ArgumentParser(
        description='Convert Potts parameters from one gauge to another')
    parser.add_argument('gauge', choices=['fieldless', 'fieldlessEven', 
                                 'weighted', 'zero', 'minJ', 'zeroJ'])
    parser.add_argument('-hin')
    parser.add_argument('-Jin')
    parser.add_argument('-hout')
    parser.add_argument('-Jout')
    parser.add_argument('-weights')
    parser.add_argument('--bin', action='store_true', 
                                 help='save in binary format')
    args = parser.parse_args()

    if args.hin:
        h0 = tryload(args.hin)
        hL,hnB = h0.shape
    else:
        h0 = None

    if args.Jin:
        J0 = tryload(args.Jin)
        jL = int((1+sqrt(1+8*J0.shape[0]))/2 + 0.5)
        jnB = int(sqrt(J0.shape[1]) + 0.5)
    else:
        J0 = None

    if args.hin == None and args.Jin == None:
        parser.error("Must supply either hin or Jin (or both)")

    if args.hin == None:
        print "No h supplied, assuming h = 0"
        L,nB = jL,jnB
        h0 = zeros((L,nB))
    elif args.Jin == None:
        print "No J supplied, assuming J = 0"
        L,nB = hL,hnB
        J0 = zeros((L*(L-1)/2,nB*nB))
    else:
        if hL != jL or hnB != jnB:
            raise Exception("Error: Size of h does not match size of J")
        L,nB = jL,jnB
    weights = None
    if args.weights != None:
        weights = load(args.weights)

    gfuncs = {'fieldless':     fieldlessGauge,
              'fieldlessEven': fieldlessGaugeEven,
              'weighted':      weightedGauge,
              'zero':          zeroGauge,
              'zeroJ':         zeroJGauge}
    h1, J1 = gfuncs[args.gauge](h0, J0, weights)

    savefunc = save if args.bin else savetxt
    if args.hout:
        savefunc(args.hout, h1)
    if args.Jout:
        savefunc(args.Jout, J1)

if __name__ == '__main__':
    main()

