#!/usr/bin/env python3
import numpy as np
import argparse, sys
from potts_common import getLq, getUnimarg, indepF

nrmlz = lambda x: x/np.sum(x,axis=1)[:,None]

def mutation_pc(ff, N, mode='jeffreys'):
    L, q = getLq(ff)

    if mode == 'jeffreys':
        print("Using C-preserving Jeffreys pseudocount", file=sys.stderr)
        mu = q/(2*N + q)
    else:
        print("Using C-preserving Bayes pseudocount", file=sys.stderr)
        mu = q/(N + q)

    f = getUnimarg(ff)
    fifj = np.array([np.add.outer(f[i],f[j]).flatten()
                 for i in range(L-1) for j in range(i+1,L)])

    # nrmlz only needed to correct fp error
    ff = nrmlz((1-mu)**2*ff + (1-mu)*mu*fifj/q + (mu/q)**2)
    return ff


def main():
    parser = argparse.ArgumentParser(
        description='Apply a pseudocount to a set of bivariate marginals.',
        epilog='The "pc" argument has different meanings based on "mode". For '
        '"constant" it is the value added to all bimarg. For "jeffreys" and '
        '"bayes" it is the MSA depth N. For "meanfield" it is the fraction '
        'of dataset size (0 to 1) as described in Morcos et al PNAS 2011.')
    parser.add_argument('margfile')
    parser.add_argument('pc', nargs='*', type=float)
    parser.add_argument('--mode', default='jeffreys', choices=['jeffreys',
                        'bayes', 'meanfield', 'unijmix', 'constant', 'tst'])
    parser.add_argument('-o', '--out', default='outpc', help="Output file")

    args = parser.parse_args(sys.argv[1:])
    ff = ff_orig = np.load(args.margfile)
    L, q = getLq(ff)

    pc = args.pc
    if args.mode == 'constant':
        print("Using a flat pseudocount", file=sys.stderr)
        if len(pc) != 1 or pc[0] < 0 or pc[0] > 1:
            raise ValueError("pc should be a single value 0 <= x <= 1")
        ff = nrmlz(ff + pc[0])
    elif args.mode in ['jeffreys', 'bayes'] :
        if len(pc) != 1:
            raise ValueError("pc should be a single value representing Neff")
        pc, = pc

        ff = mutation_pc(ff, pc, args.mode)
    elif args.mode == 'unijmix':
        if len(pc) != 2:
            raise ValueError("pc should be two values representing (pc, w)"
                             "where the weight is 0 <= w <= 1 and pc is "
                             "a pseudocount on the univariate marginals, eg "
                             "put 0.5/Neff for Jeffrey's prior.")
        pc, l = pc
        f = getUnimarg(ff)
        f = nrmlz(f + pc)
        fifj = np.array([np.outer(f[i],f[j]).flatten()
                         for i in range(L-1) for j in range(i+1,L)])
        ff = nrmlz((1-l)*ff + l*fifj)

    elif args.mode == 'meanfield':
        print("Using meanfield pseudocount", file=sys.stderr)
        if len(pc) != 1 or pc[0] < 0 or pc[0] > 1:
            raise ValueError("pc should be a single value 0 <= x <= 1")
        pc = pc[0]/(1-pc[0])
        ff = nrmlz(ff + pc/(q*q))

    elif args.mode == 'test':
        if len(pc) != 2:
            raise ValueError("pc should be two values representing (pc, w)"
                             "where the weight is 0 <= w <= 1 and pc is "
                             "a pseudocount on the univariate marginals, eg "
                             "put 0.5/Neff for Jeffrey's prior.")
        pc, l = pc
        f = getUnimarg(ff)
        fpc = nrmlz(f + pc)
        z = (fpc - (1-l)*f)/l

        zizj = np.array([np.outer(z[i],z[j]).flatten()
                         for i in range(L-1) for j in range(i+1,L)])
        ff = nrmlz((1-l)*ff + l*zizj)

    ssr = np.sum((ff - ff_orig)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = (np.abs(ff_orig - ff)/ff_orig)
        ferr = np.mean(rel_err[ff_orig > 0.01])
    print("Difference in pseudocounted marginals relative to original:")
    print("SSR: {:.2f}   Ferr: {:.4f}".format(ssr, ferr))


    np.save(args.out, ff.astype('<f4'))

if __name__ == '__main__':
    main()
