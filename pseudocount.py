#!/usr/bin/env python2
from scipy import *
import numpy as np
import argparse, sys

nrmlz = lambda x: x/sum(x,axis=1)[:,newaxis]

def constant(ff, pc):
    return nrmlz(ff + pc)

def prior(ff, pc):
    L = int(((1+sqrt(1+8*ff.shape[0]))/2) + 0.5) 
    nB = int(sqrt(ff.shape[1]) + 0.5)
    mu = float(pc)

    ffs = ff.reshape(ff.shape[0], nB, nB)
    f = array([sum(ffs[0],axis=1)] + 
              [sum(ffs[n],axis=0) for n in range(L-1)])
    f = f/(sum(f,axis=1)[:,newaxis]) # correct any fp errors

    sf = array([np.add.outer(f[i],f[j]).flatten() 
                   for i in range(L-1) for j in range(i+1,L)])
    
    # nrmlz only needed to correct fp error
    return nrmlz((1-mu)**2*ff + (1-mu)*mu*sf/nB + (mu/nB)**2)

def onuchic(ff, pc):
    nB = int(sqrt(ff.shape[1]) + 0.5)

    if pc < 0 or pc > 1:
        raise Exception("pc must be between 0 and 1")
    pc = pc/(1-pc) #to make it the same as in reference impl.
    return nrmlz(ff + pc/(nB*nB))

def main():
    parser = argparse.ArgumentParser(description='Add pseudocount')
    parser.add_argument('margfile')
    parser.add_argument('pc', type=float)
    parser.add_argument('-mode', choices=['constant', 'prior', 'onuchic'], 
                        default='prior')
    parser.add_argument('-o', default='outpc', help="Output file")

    args = parser.parse_args(sys.argv[1:])
    ff = load(args.margfile)

    pc = args.pc
    if args.mode == 'constant':
        print >>sys.stderr, "Using the simple pseudocount"
        ff = constant(ff, pc)
    elif args.mode == 'prior':
        print >>sys.stderr, "Using Priors pseudocount"
        ff = prior(ff, pc)
    elif args.mode == 'onuchic':
        print >>sys.stderr, "Using Onuchic pseudocount"
        ff = onuchic(ff, pc)

    save(args.o, ff.astype('<f4'))

if __name__ == '__main__':
    main()
