#!/usr/bin/env python
import numpy as np
import argparse, sys

nrmlz = lambda x: x/np.sum(x,axis=1)[:,None]

def getLq(J):
    L = int(((1+np.sqrt(1+8*J.shape[0]))//2) + 0.5)
    q = int(np.sqrt(J.shape[1]) + 0.5)
    return L, q

def main():
    parser = argparse.ArgumentParser(
        description='Apply a pseudocount to a set of bivariate marginals.',
        epilog='The "pc" argument has different meanings based on "mode". For '
        '"constant" it is the value added to all bimarg. For "jeffreys" and '
        '"bayes" it is the MSA depth N. For "meanfield" it is the fraction '
        'of dataset size (0 to 1) as described in Morcos et al PNAS 2011.')
    parser.add_argument('margfile')
    parser.add_argument('pc', type=float)
    parser.add_argument('--mode', choices=['jeffreys', 'bayes', 'meanfield',
                                           'constant'], default='jeffreys')
    parser.add_argument('-o', '--out', default='outpc', help="Output file")

    args = parser.parse_args(sys.argv[1:])
    ff = np.load(args.margfile)
    L, q = getLq(ff)

    pc = args.pc
    if args.mode == 'constant':
        print("Using a flat pseudocount", file=sys.stderr)
        ff = nrmlz(ff + pc)
    elif args.mode in ['jeffreys', 'bayes'] :

        ffs = ff.reshape(ff.shape[0], q, q)
        f = np.array([np.sum(ffs[0],axis=1)] + 
                     [np.sum(ffs[n],axis=0) for n in range(L-1)])
        f = f/(np.sum(f,axis=1)[:,None]) # correct any fp errors

        fifj = np.array([np.add.outer(f[i],f[j]).flatten() 
                         for i in range(L-1) for j in range(i+1,L)])
        
        if args.mode == 'jeffreys':
            print("Using C-preserving Jeffreys pseudocount", file=sys.stderr)
            mu = q/(2*pc + q)
        else:
            print("Using C-preserving Bayes pseudocount", file=sys.stderr)
            mu = q/(pc + q)

        # nrmlz only needed to correct fp error
        ff = nrmlz((1-mu)**2*ff + (1-mu)*mu*fifj/q + (mu/q)**2)

    elif args.mode == 'meanfield':
        print("Using meanfield pseudocount", file=sys.stderr)
        if pc < 0 or pc > 1:
            raise Exception("pc must be between 0 and 1")
        pc = pc/(1-pc)
        ff = nrmlz(ff + pc/(q*q))

    np.save(args.out, ff.astype('<f4'))

if __name__ == '__main__':
    main()
